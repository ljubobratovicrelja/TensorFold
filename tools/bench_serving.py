#!/usr/bin/env python3
"""Flash-Next serving benchmark — engine-agnostic (vLLM or TensorFold).

Designed so the same script produces comparable numbers on two different
engines. Everything goes through the OpenAI-compatible API.

Protocol
--------
* Every measured row is preceded by an unmeasured run of the SAME prompts, so
  kernels are hot and (for long context) the prefix is already in the prompt
  cache. There are no cold measured rows.
* Decode speed is (completion_tokens - 1) / (time of last streamed token -
  time of first), so TTFT is excluded. That is the sparkDash definition.
* completion_tokens comes from the usage block when the server sends one, and
  otherwise from the requested max_tokens when finish_reason == "length".
* Prompts are prose ("explain step by step how a hash map works") to match
  MiaAI-Lab's published prose rows. Concurrent streams get distinct prompts.
* Filler text is generated once per seed and reused, so the warm pass and the
  measured pass share a prefix.

Usage
-----
  BASE_URL=http://127.0.0.1:8005/v1 MODEL=qwen3.8-flash-next \
      OUT=results/vllm.json python3 bench.py

  PHASES=short,long,prefill   # subset to run
"""
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8005/v1")
MODEL = os.environ.get("MODEL", "qwen3.8-flash-next")
OUT = os.environ.get("OUT", "results/run.json")
LABEL = os.environ.get("LABEL", "unlabelled")
PHASES = set(os.environ.get("PHASES", "short,long,prefill").split(","))
SHORT_TOKENS = int(os.environ.get("SHORT_TOKENS", "600"))
LONG_OUT = int(os.environ.get("LONG_OUT", "256"))
DEEP_OUT = int(os.environ.get("DEEP_OUT", "128"))

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
         "november oscar papa quebec romeo sierra tango uniform victor whiskey xray "
         "yankee zulu scheduler kernel latency throughput quantize tensor gradient cache "
         "pointer buffer register pipeline parallel entropy manifold").split()
WORDS_PER_TOKEN = 1.3665

PROSE = [
    "Explain step by step how a hash map works.",
    "Explain step by step how a B-tree stays balanced on insert.",
    "Explain step by step how TCP handles a lost packet.",
    "Explain step by step how a garbage collector finds unreachable objects.",
    "Explain step by step how a compiler turns source into machine code.",
    "Explain step by step how a GPU hides memory latency.",
]

RESULTS = []
LOCK = threading.Lock()


def record(row):
    with LOCK:
        RESULTS.append(row)
        print(json.dumps(row), flush=True)


def build_long(seed, approx_tokens):
    rng = random.Random(seed)
    body = " ".join(rng.choice(WORDS) for _ in range(int(approx_tokens / WORDS_PER_TOKEN)))
    return (f"Document {seed}. Below is a log excerpt.\n\n{body}\n\n"
            "List the ten most frequent words in the excerpt above, one per line with counts.")


def call(prompt, max_tokens, temperature):
    """One streaming request. Returns per-request metrics."""
    req = {"model": MODEL,
           "messages": [{"role": "user", "content": prompt}],
           "max_tokens": max_tokens, "temperature": temperature, "stream": True,
           "stream_options": {"include_usage": True}}
    data = json.dumps(req).encode()
    r = urllib.request.Request(BASE + "/chat/completions", data=data,
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    t_first = t_last = None
    chunks = 0
    usage = {}
    finish = None
    err = None
    try:
        with urllib.request.urlopen(r, timeout=7200) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data: "):
                    continue
                p = line[6:]
                if p == "[DONE]":
                    break
                try:
                    obj = json.loads(p)
                except json.JSONDecodeError:
                    continue
                if obj.get("error"):
                    err = str(obj["error"])[:300]
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                        now = time.time()
                        if t_first is None:
                            t_first = now
                        t_last = now
                        chunks += 1
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:300]}"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    t_end = time.time()

    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    if ct is None:
        ct = max_tokens if finish == "length" else chunks
    ttft = (t_first - t0) if t_first else None
    dec = (ct - 1) / (t_last - t_first) if t_first and t_last and ct and ct > 1 else None
    return {"prompt_tokens": pt, "completion_tokens": ct, "chunks": chunks,
            "finish": finish, "ttft": ttft, "decode": dec, "wall": t_end - t0,
            "error": err}


def run_parallel(prompts, max_tokens, temperature):
    out = [None] * len(prompts)

    def w(i):
        out[i] = call(prompts[i], max_tokens, temperature)

    ts = [threading.Thread(target=w, args=(i,)) for i in range(len(prompts))]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out, time.time() - t0


def phase_short():
    for c in (1, 2, 4):
        prompts = [PROSE[i % len(PROSE)] for i in range(c)]
        run_parallel(prompts, 128, 0.7)                      # warm, unmeasured
        rows, wall = run_parallel(prompts, SHORT_TOKENS, 0.7)  # measured
        for i, r in enumerate(rows):
            record({"phase": "short", "concurrency": c, "stream": i,
                    "ctx_target": 0, **r})
        ok = [r for r in rows if r.get("decode")]
        if ok:
            record({"phase": "short", "concurrency": c, "summary": True,
                    "aggregate": sum(r["decode"] for r in ok),
                    "per_stream": statistics.median([r["decode"] for r in ok]),
                    "ttft_median": statistics.median([r["ttft"] for r in ok if r["ttft"]]),
                    "wall": wall})


def phase_long():
    for ctx in (32768, 258000):
        for c in ((1, 2, 4) if ctx == 32768 else (1, 4)):
            out_tok = LONG_OUT if ctx == 32768 else DEEP_OUT
            prompts = [build_long(4000 + c * 100 + i, ctx) for i in range(c)]
            warm, _ = run_parallel(prompts, 32, 0.7)         # warm prefix, unmeasured
            bad = [r for r in warm if r.get("error") or not r.get("decode")]
            if bad and any(r.get("error") for r in warm):
                record({"phase": "long", "concurrency": c, "ctx_target": ctx,
                        "summary": True, "error": f"warm pass failed: {bad[0].get('error')}"})
                continue
            rows, wall = run_parallel(prompts, out_tok, 0.7)  # measured
            for i, r in enumerate(rows):
                record({"phase": "long", "concurrency": c, "stream": i,
                        "ctx_target": ctx, **r})
            ok = [r for r in rows if r.get("decode")]
            if ok:
                record({"phase": "long", "concurrency": c, "ctx_target": ctx,
                        "summary": True,
                        "aggregate": sum(r["decode"] for r in ok),
                        "per_stream": statistics.median([r["decode"] for r in ok]),
                        "ttft_median": statistics.median([r["ttft"] for r in ok if r["ttft"]]),
                        "prompt_tokens": statistics.median(
                            [r["prompt_tokens"] for r in ok if r["prompt_tokens"]] or [0]),
                        "wall": wall})


def phase_prefill():
    for ctx in (4096, 32768, 131072):
        p = build_long(9000 + ctx, ctx)
        r = call(p, 16, 0)                                  # kernels already warm
        record({"phase": "prefill", "ctx_target": ctx,
                "prompt_tokens": r.get("prompt_tokens"), "ttft": r.get("ttft"),
                "prefill_tps": (r["prompt_tokens"] / r["ttft"])
                                if r.get("prompt_tokens") and r.get("ttft") else None,
                "error": r.get("error")})


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    print(f"# {LABEL} | {BASE} | model={MODEL} | phases={sorted(PHASES)}", flush=True)
    t0 = time.time()
    # global warmup: three short requests, discarded
    for i in range(3):
        call(PROSE[i], 128, 0.7)
    print("# warmup done", flush=True)
    if "short" in PHASES:
        phase_short()
    if "long" in PHASES:
        phase_long()
    if "prefill" in PHASES:
        phase_prefill()
    meta = {"label": LABEL, "base_url": BASE, "model": MODEL,
            "phases": sorted(PHASES), "short_tokens": SHORT_TOKENS,
            "long_out": LONG_OUT, "deep_out": DEEP_OUT,
            "elapsed_s": round(time.time() - t0, 1),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(OUT, "w") as f:
        json.dump({"meta": meta, "rows": RESULTS}, f, indent=1)
    print(f"# wrote {OUT} ({len(RESULTS)} rows, {meta['elapsed_s']}s)", flush=True)
