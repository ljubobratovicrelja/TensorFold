#!/usr/bin/env python3
"""The standard Flash-Next serving matrix: N agents at a clear 20k or 100k context.

One request per agent, each with its own filled prompt (no shared prefix), so
every cell measures a cold prefill of that fill and then decode under N-way
concurrency. No warm replay: a cell is one pass, which is what makes the whole
matrix minutes, not an hour.

  python3 tools/bench_matrix.py                        # 1,2,4,6 x 20000,100000, :8005
  python3 tools/bench_matrix.py --agents 1,2,4 --fill 20000 --out /tmp/m.json

Per stream: decode tok/s = (completion_tokens - 1) / (last piece - first piece),
so TTFT is excluded. Aggregate tok/s = all streams' tokens after their first
piece / (last piece of any stream - first piece of any stream). A stream that
errors or answers without a usage block is reported, never counted.

The point of 6 on a 4-lane server is the queue: 4 requests run, 2 wait, and the
cell's wall and per-stream decode show it. Distinct prompts are also prefilled one
at a time, so the Nth agent's TTFT is roughly N prefill times; `ttft spread` reports
that, and `cell/s` is the throughput you actually get including it.
"""

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
         "november oscar papa quebec romeo sierra tango uniform victor whiskey xray "
         "yankee zulu scheduler kernel latency throughput quantize tensor gradient cache "
         "pointer buffer register pipeline parallel entropy manifold").split()
WORDS_PER_TOKEN = 1.3665
ASK = "List the ten most frequent words in the excerpt above, one per line with counts."


def build(seed: int, approx_tokens: int) -> str:
    rng = random.Random(seed)
    body = " ".join(rng.choice(WORDS) for _ in range(int(approx_tokens / WORDS_PER_TOKEN)))
    return f"Document {seed}. Below is a log excerpt.\n\n{body}\n\n{ASK}"


def call(base: str, model: str, prompt: str, max_tokens: int, temperature: float, timeout: float) -> dict:
    req = {"model": model,
           "messages": [{"role": "user", "content": prompt}],
           "max_tokens": max_tokens,
           "temperature": temperature,
           "chat_template_kwargs": {"enable_thinking": False},
           "stream": True,
           "stream_options": {"include_usage": True}}
    r = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                               json.dumps(req).encode(),
                               {"Content-Type": "application/json"})
    t0 = time.time()
    t_first = t_last = None
    pieces = 0
    usage, finish, err = {}, None, None
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
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
                        t_first = t_first or now
                        t_last = now
                        pieces += 1
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:300]}"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    t_end = time.time()

    ct = usage.get("completion_tokens")
    if ct is None:
        ct = max_tokens if finish == "length" else pieces
    return {"prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": ct,
            "finish": finish,
            "ttft": (t_first - t0) if t_first else None,
            "decode": (ct - 1) / (t_last - t_first) if t_first and t_last and ct and ct > 1 else None,
            "wall": t_end - t0,
            "first": t_first, "last": t_last,
            "error": err}


def cell(base: str, model: str, fill: int, agents: int, tokens: int, temperature: float, timeout: float) -> dict:
    prompts = [build(fill * 1000 + agents * 100 + i, fill) for i in range(agents)]
    rows = [None] * agents
    barrier = threading.Barrier(agents)

    def work(i):
        barrier.wait()
        rows[i] = call(base, model, prompts[i], tokens, temperature, timeout)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(agents)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    ok = [r for r in rows if r and not r.get("error") and r.get("decode")]
    bad = [r for r in rows if r and r.get("error")]
    first = min((r["first"] for r in ok), default=None)
    last = max((r["last"] for r in ok), default=None)
    aggregate = None
    if ok and first is not None and last is not None and last > first:
        aggregate = sum(r["completion_tokens"] - 1 for r in ok) / (last - first)
    ttfts = [r["ttft"] for r in ok if r.get("ttft")]
    prefills = [r["prompt_tokens"] / r["ttft"] for r in ok if r.get("ttft") and r.get("prompt_tokens")]
    # The cell's own throughput, prefill included: what an agent operator actually waits through.
    throughput = (sum(r["completion_tokens"] for r in ok) / wall) if ok else None
    return {"fill": fill, "agents": agents,
            "prompt_tokens": statistics.median([r["prompt_tokens"] for r in ok if r.get("prompt_tokens")] or [0]),
            "prefill_tps": statistics.median(prefills or [0]),
            "ttft_median": statistics.median(ttfts or [0]),
            "ttft_min": min(ttfts or [0]),
            "ttft_max": max(ttfts or [0]),
            "per_stream": statistics.median([r["decode"] for r in ok] or [0]),
            "aggregate": aggregate,
            "throughput": throughput,
            "wall": wall,
            "failed": len(bad),
            "errors": [r["error"] for r in bad][:3],
            "streams": [{k: r.get(k) for k in ("prompt_tokens", "completion_tokens", "ttft", "decode", "wall", "error")} for r in rows if r]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default=os.environ.get("BASE_URL", "http://127.0.0.1:8005/v1"))
    ap.add_argument("--model", default=os.environ.get("MODEL", "qwen3.8-flash-next"))
    ap.add_argument("--agents", default="1,2,4,6")
    ap.add_argument("--fill", default="20000,100000")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--label", default="matrix")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    levels = [int(x) for x in a.agents.split(",")]
    fills = [int(x) for x in a.fill.split(",")]
    out = a.out or f"bench_matrix-{a.label}.json"

    print(f"# {a.label} | {a.base} | model={a.model} | agents={levels} | fill={fills} | tokens={a.tokens}", flush=True)
    for i in range(2):  # kernels hot before the first measured cell
        call(a.base, a.model, build(i, 512), 16, a.temperature, a.timeout)

    rows = []
    t0 = time.time()
    print(f"{'fill':>7} {'agents':>6} {'prompt':>8} {'pref/s':>8} {'ttft_s':>7} {'spread':>7} "
          f"{'tok/s':>7} {'agg':>7} {'cell/s':>7} {'wall_s':>7} {'fail':>4}", flush=True)
    for fill in fills:
        for agents in levels:
            c = cell(a.base, a.model, fill, agents, a.tokens, a.temperature, a.timeout)
            rows.append(c)
            print(f"{c['fill']:>7} {c['agents']:>6} {c['prompt_tokens']:>8} {c['prefill_tps']:>8.0f} {c['ttft_median']:>7.2f} "
                  f"{c['ttft_max'] - c['ttft_min']:>7.2f} {c['per_stream']:>7.1f} {(c['aggregate'] or 0):>7.1f} "
                  f"{(c['throughput'] or 0):>7.1f} {c['wall']:>7.1f} {c['failed']:>4}", flush=True)

    meta = {"label": a.label, "base_url": a.base, "model": a.model,
            "agents": levels, "fill": fills, "tokens": a.tokens, "temperature": a.temperature,
            "elapsed_s": round(time.time() - t0, 1),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump({"meta": meta, "rows": rows}, open(out, "w"), indent=1)
    print(f"# wrote {out} ({meta['elapsed_s']}s)", flush=True)
    return 1 if any(r["failed"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())