#!/usr/bin/env python3
"""Paired acceptance A/B between two drafter checkpoints, on the served path.

The question this answers is not "is fp8 faster" — that is a stage table — but "does fp8 buy
throughput end to end, or does it pay for it in acceptance". That needs the metric the drafter is
judged on: accept length per round, per prompt, on the same prompts, with the two arms measured in
the same session so box state cancels.

Usage, one server per arm (the box serves one model at a time):

    # arm A running on :8099, log at /tmp/srv-bf16.log
    accept_ab.py --arm bf16=/tmp/srv-bf16.log --out runs/ab-bf16.jsonl
    # restart the server on the other checkpoint, then:
    accept_ab.py --arm fp8=/tmp/srv-fp8.log --out runs/ab-fp8.jsonl --append
    accept_ab.py --compare runs/ab-bf16.jsonl runs/ab-fp8.jsonl

`--append` adds rows to an existing file, so the two arms can land in one file and `--compare` can
pair them by prompt index. Numbers come from the server's own `done chatcmpl` line, never from the
HTTP body: the body reports usage, the log line reports what the drafter actually did.

A verdict is refused when n is too small to resolve the effect it is asked about. That refusal is
the point of the script: 16 prompts cannot see a 2% accept-length change, and reporting one anyway
is how a quantization arm gets shipped on noise.
"""
import argparse
import json
import math
import re
import subprocess
import sys
import time
import urllib.request
from typing import Any

MODEL = "qwen3.8-flash-next"
DONE = "done chatcmpl"

# two-sided 95% t quantiles for the small samples this harness runs at
T975 = {8: 2.365, 12: 2.201, 16: 2.131, 20: 2.093, 24: 2.069, 32: 2.040, 64: 1.998, 128: 1.980}


def t975(n):
    """Nearest tabulated two-sided 95% t for n-1 degrees of freedom, 1.96 beyond the table."""
    df = max(n - 1, 1)
    return T975.get(df, 1.96) if df < 128 else 1.96


def post(base, prompt, seed, max_tokens):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "seed": seed,
        "draft": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    req = urllib.request.Request(
        base, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r), time.time() - t0


def done_lines(log):
    out = subprocess.run(["grep", "-aE", DONE, log], capture_output=True, text=True).stdout
    return out.strip().splitlines()


def parse(line: str) -> dict[str, Any]:
    g = lambda pat, f=str: (f(m.group(1)) if (m := re.search(pat, line)) else None)
    return {
        "tokens": g(r"tokens=(\d+)", int),
        "rounds": g(r"rounds=(\d+)", int),
        "accepted": g(r"accepted=(\d+)", int),
        "drafted": g(r"accepted=\d+/(\d+)", int),
        "tps": g(r"tok/s=([\d.]+)", float),
        "finish": g(r"finish=(\w+)"),
        "sha": g(r"sha=(\w+)"),
    }


def run_arm(a):
    name, log = a.arm.split("=", 1)
    prompts = json.load(open(a.prompts))
    if a.limit:
        prompts = prompts[: a.limit]
    rows: list[dict[str, Any]] = []
    n_before = len(done_lines(log))
    for i, pr in enumerate(prompts):
        seed = a.seed_base + i
        try:
            _, wall = post(a.base, pr, seed, a.max_tokens)
        except Exception as e:  # a failed request is a row, not a crash
            print(f"[{i:3d}] ERROR {e}")
            rows.append({"arm": name, "i": i, "error": str(e)})
            continue
        lines = done_lines(log)
        new = lines[n_before:]
        n_before = len(lines)
        if not new:
            print(f"[{i:3d}] no done line")
            rows.append({"arm": name, "i": i, "error": "no done line"})
            continue
        d: dict[str, Any] = parse(new[-1])
        d.update({"arm": name, "i": i, "seed": seed, "wall_s": round(wall, 2)})
        d["accept_len"] = (d["tokens"] / d["rounds"]) if d["tokens"] and d["rounds"] else None
        rows.append(d)
        if d["accept_len"]:
            print(f"[{i:3d}] tok/s={d['tps']:6} accept_len={d['accept_len']:.3f} "
                  f"acc={d['accepted']}/{d['drafted']} rounds={d['rounds']} finish={d['finish']}")
    mode = "a" if a.append else "w"
    with open(a.out, mode) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in rows if r.get("accept_len")]
    print(f"\n=== {name}: {len(ok)}/{len(prompts)} prompts, rows -> {a.out} ===")
    if ok:
        print(f"mean accept_len  {sum(r['accept_len'] for r in ok) / len(ok):.3f} tok/round")
        print(f"mean tok/s       {sum(r['tps'] for r in ok) / len(ok):.1f}")
        print(f"aggregate rate   {sum(r['accepted'] for r in ok) / sum(r['drafted'] for r in ok):.3f}")
        print(f"total tokens     {sum(r['tokens'] for r in ok)}")


def load_rows(path):
    out = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("accept_len"):
            out[(r["arm"], r["i"])] = r
    return out


def compare(path_a, path_b, effect):
    rows = load_rows(path_a)
    rows.update(load_rows(path_b))
    arms = sorted({a for a, _ in rows})
    if len(arms) != 2:
        sys.exit(f"need exactly 2 arms in the files, found {arms}")
    a, b = arms
    idx = sorted({i for _, i in rows if (a, i) in rows and (b, i) in rows})
    if len(idx) < 3:
        sys.exit(f"only {len(idx)} paired prompts; nothing to compare")
    xa = [rows[(a, i)]["accept_len"] for i in idx]
    xb = [rows[(b, i)]["accept_len"] for i in idx]
    ta = [rows[(a, i)]["tps"] for i in idx]
    tb = [rows[(b, i)]["tps"] for i in idx]
    dif = [y - x for x, y in zip(xa, xb)]  # b relative to a
    n = len(dif)
    mean = sum(dif) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in dif) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    half = t975(n) * se
    print(f"paired prompts   {n}")
    print(f"{a:>12s}  accept_len {sum(xa) / n:.3f}   tok/s {sum(ta) / n:.1f}")
    print(f"{b:>12s}  accept_len {sum(xb) / n:.3f}   tok/s {sum(tb) / n:.1f}")
    print(f"paired diff ({b} - {a})  {mean:+.3f} tok/round   sd {sd:.3f}   "
          f"95% CI [{mean - half:+.3f}, {mean + half:+.3f}]")
    base = sum(xa) / n
    print(f"relative          {100 * mean / base:+.2f}%   "
          f"CI [{100 * (mean - half) / base:+.2f}%, {100 * (mean + half) / base:+.2f}%]")
    # the sample this test needs before it can call an effect of `effect` (relative, e.g. 0.02)
    delta = effect * base
    need = math.ceil((t975(n) * sd / delta) ** 2) if delta and sd else None
    if need:
        print(f"to resolve {100 * effect:.1f}% ({delta:.3f} tok/round) at this sd: "
              f"n >= {need} paired prompts")
    if half > delta:
        print(f"VERDICT: UNDERPOWERED — the CI (±{half:.3f}) is wider than the "
              f"{delta:.3f} effect asked about. Do not call this arm parity or a regression.")
    elif mean < -delta:
        print("VERDICT: REGRESSION — the whole CI is below the effect asked about.")
    elif mean > delta:
        print("VERDICT: GAIN in acceptance, not a cost.")
    else:
        print(f"VERDICT: PARITY within ±{delta:.3f} tok/round at n={n}.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", help="name=server_log_path; run that arm against the live server")
    p.add_argument("--out", help="jsonl to write rows to")
    p.add_argument("--append", action="store_true", help="append to --out instead of truncating")
    p.add_argument("--compare", nargs=2, metavar=("ROWS_A", "ROWS_B"))
    p.add_argument("--effect", type=float, default=0.02,
                   help="relative accept-length effect to resolve (default 2%%)")
    p.add_argument("--prompts", default="/tmp/gsm8k_16.json")
    p.add_argument("--base", default="http://127.0.0.1:8099/v1/chat/completions")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    if a.compare:
        compare(a.compare[0], a.compare[1], a.effect)
    elif a.arm and a.out:
        run_arm(a)
    else:
        p.error("either --arm NAME=LOG --out FILE, or --compare A B")


if __name__ == "__main__":
    main()
