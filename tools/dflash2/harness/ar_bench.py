#!/usr/bin/env python3
"""Autoregressive-only throughput and output-agreement bench on the served path.

One server per arm, no drafter (`--no-drafts`, no `--drafter`), greedy decode, the 16 public GSM8K
test prompts in `prompts/gsm8k_16.json`, `prompts/gsm8k_gold.json` beside them for answer scoring.

    # arm running on :8099, server log at /tmp/ar/serve.log
    ar_bench.py --arm baseline --log /tmp/ar/serve.log --out runs/ar/baseline.json

    # after swapping the quantization on and restarting
    ar_bench.py --arm head_fp8 --log /tmp/ar/serve.log --out runs/ar/head_fp8.json

    ar_bench.py --compare runs/ar/baseline.json runs/ar/head_fp8.json

The artifact is one JSON object per arm: config, git rev, per-prompt rows, aggregate. Rows carry the
generated text and a sha256 of it, so an arm that changes answers is visible without a rerun, and the
GSM8K `\\boxed{}` answer is extracted and graded where the gold is present.

Greedy sampling is deliberate: two arms then differ only by the weights, and any difference in text is
a fidelity signal rather than sampling noise. Speed is measured as wall time over the HTTP call, which
includes the terminal download and the sampler; the server's own `done chatcmpl` line is parsed too so
the two can be cross-checked.
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request

MODEL = "qwen3.8-flash-next"
BOXED = re.compile(r"\\boxed\{([^}]*)\}")
NUM = re.compile(r"-?\d+(?:[\.,]\d+)?")


def post(base, prompt, max_tokens, seed):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "draft": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    return out, time.time() - t0


def answer(text):
    m = BOXED.findall(text or "")
    if not m:
        return None
    v = NUM.findall(m[-1].replace(",", ""))
    return v[-1] if v else None


def gold_of(answer_text):
    m = re.search(r"####\s*(-?[\d.,]+)", answer_text or "")
    return m.group(1).replace(",", "") if m else None


def done_line(log, n):
    """The n-th `done chatcmpl` line, which is what the engine itself measured."""
    try:
        out = subprocess.run(["grep", "-a", "done chatcmpl", log],
                             capture_output=True, text=True).stdout.strip().splitlines()
    except Exception:
        return None
    if n >= len(out):
        return None
    line = out[n]
    g = lambda p: (float(m.group(1)) if (m := re.search(p, line)) else None)
    return {"tps": g(r"tok/s=([\d.]+)"), "rounds": g(r"rounds=(\d+)"),
            "accepted": g(r"accepted=(\d+)"),
            "drafted": (float(m.group(1)) if (m := re.search(r"accepted=\d+/(\d+)", line)) else None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="arm")
    ap.add_argument("--base", default="http://127.0.0.1:8099/v1")
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "prompts/gsm8k_16.json"))
    ap.add_argument("--gold", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "prompts/gsm8k_gold.json"))
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()

    if a.compare:
        return compare(*a.compare)

    prompts = json.load(open(a.prompts))
    golds = []
    if os.path.exists(a.gold):
        rows = json.load(open(a.gold))["rows"]
        golds = [gold_of(r["row"].get("answer")) for r in rows]
    if a.limit:
        prompts = prompts[: a.limit]
        golds = golds[: a.limit]

    rows, t_start = [], time.time()
    for i, pr in enumerate(prompts):
        body, wall = post(a.base, pr, a.max_tokens, seed=1234 + i)
        text = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        usage = body.get("usage") or {}
        toks = usage.get("completion_tokens")
        row = {
            "i": i,
            "prompt_sha256": hashlib.sha256(pr.encode()).hexdigest()[:16],
            "tokens": toks,
            "wall_s": round(wall, 4),
            "tok_s": round(toks / wall, 3) if toks and wall else None,
            "finish": (body.get("choices") or [{}])[0].get("finish_reason"),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "chars": len(text),
            "pred": answer(text),
            "gold": golds[i] if i < len(golds) else None,
            "text": text,
        }
        row["correct"] = None if row["pred"] is None or row["gold"] is None else row["pred"] == row["gold"]
        row["server"] = done_line(a.log, i) if a.log else None
        rows.append(row)
        print(f"[{i:2d}] {toks} tok in {wall:6.2f}s = {row['tok_s']:6.2f} tok/s  "
              f"pred={row['pred']} gold={row['gold']} finish={row['finish']}", flush=True)

    ok = [r for r in rows if r["tok_s"]]
    graded = [r for r in rows if r["correct"] is not None]
    rev = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                          "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    art = {
        "arm": a.arm,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_rev": rev,
        "config": {"model": MODEL, "max_tokens": a.max_tokens, "temperature": 0.0,
                   "draft": False, "drafter": None, "prompts": os.path.basename(a.prompts),
                   "thinking": False},
        "aggregate": {
            "prompts": len(rows),
            "tokens": sum(r["tokens"] or 0 for r in rows),
            "wall_s": round(time.time() - t_start, 2),
            "tok_s_mean": round(statistics.mean(r["tok_s"] for r in ok), 3) if ok else None,
            "tok_s_median": round(statistics.median(r["tok_s"] for r in ok), 3) if ok else None,
            "tok_s_min": min((r["tok_s"] for r in ok), default=None),
            "tok_s_max": max((r["tok_s"] for r in ok), default=None),
            "graded": len(graded),
            "correct": sum(1 for r in graded if r["correct"]),
        },
        "rows": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(art, open(a.out, "w"), indent=1)
    ag = art["aggregate"]
    print(f"\n=== {a.arm}: {ag['prompts']} prompts, {ag['tokens']} tokens, "
          f"{ag['tok_s_mean']:.2f} tok/s mean / {ag['tok_s_median']:.2f} median, "
          f"{ag['correct']}/{ag['graded']} correct -> {a.out} ===")


def compare(path_a, path_b):
    A, B = json.load(open(path_a)), json.load(open(path_b))
    ra = {r["i"]: r for r in A["rows"]}
    rb = {r["i"]: r for r in B["rows"]}
    idx = sorted(set(ra) & set(rb))
    if not idx:
        sys.exit("no common prompts")
    d = [rb[i]["tok_s"] - ra[i]["tok_s"] for i in idx if ra[i]["tok_s"] and rb[i]["tok_s"]]
    same = sum(1 for i in idx if ra[i]["text_sha256"] == rb[i]["text_sha256"])
    n = len(d)
    mean = statistics.mean(d)
    sd = statistics.stdev(d) if n > 1 else 0.0
    se = sd / (n ** 0.5) if n else 0.0
    t = {8: 2.365, 12: 2.201, 16: 2.131, 20: 2.093, 24: 2.069, 32: 2.040}.get(n - 1, 1.96)
    lo, hi = mean - t * se, mean + t * se
    rel = 100 * mean / statistics.mean(ra[i]["tok_s"] for i in idx)
    ca = sum(1 for i in idx if ra[i].get("correct"))
    cb = sum(1 for i in idx if rb[i].get("correct"))
    print(f"paired prompts        {n}")
    print(f"  {A['arm']:14s} {statistics.mean(ra[i]['tok_s'] for i in idx):7.2f} tok/s   correct {ca}")
    print(f"  {B['arm']:14s} {statistics.mean(rb[i]['tok_s'] for i in idx):7.2f} tok/s   correct {cb}")
    print(f"paired diff (B-A)     {mean:+.2f} tok/s  sd {sd:.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"relative              {rel:+.2f}%   CI [{100*(lo/statistics.mean(ra[i]['tok_s'] for i in idx)):+.2f}%, "
          f"{100*(hi/statistics.mean(ra[i]['tok_s'] for i in idx)):+.2f}%]")
    print(f"identical outputs     {same}/{n}")
    verdict = ("real GAIN" if lo > 0 else "real LOSS" if hi < 0 else "null (CI spans 0)")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
