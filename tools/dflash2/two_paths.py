#!/usr/bin/env python3
"""The two-path equivalence check: one prompt, greedy, one token sha per speculation path.

The whole point of `--drafter` is that it is additive: the same binary must serve MTP, no-drafts
and (later) a DFlash2 drafter, and every path must produce the same tokens, because speculative
decoding verifies against the target's own argmax. So the gate is not "does it run" but "does it
return the same tensorfold.token_sha256 as the MTP build it replaced".

Usage:
    python3 tools/dflash2/two_paths.py --label mtp  --port 8099
    python3 tools/dflash2/two_paths.py --label nodrafts --port 8099
    python3 tools/dflash2/two_paths.py --label dflash2 --port 8099 --drafter <dir>

Each run prints one JSON line; pass the same --out to compare them, which is what the acceptance
gate does. Greedy (temperature 0) because a sampled path may legitimately differ.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

# A prompt with a long deterministic answer: the model stops at ~25 tokens on open-ended asks, and
# a short completion makes decode rate and acceptance meaningless.
PROMPT = (
    "Write a Python function `binary_search(nums, target)` returning the index of target in a "
    "sorted list or -1. Output only Python source, no comments, no fences. Then add the tests."
)


def post(port: int, path: str, body: dict, timeout: float = 900.0) -> dict:
    url = f"http://127.0.0.1:{port}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run(args) -> dict:
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    out = post(args.port, "/v1/chat/completions", body)
    tf = out.get("tensorfold") or {}
    row = {
        "label": args.label,
        "port": args.port,
        "model": out.get("model"),
        "token_sha256": tf.get("token_sha256"),
        "completion_tokens": out.get("usage", {}).get("completion_tokens"),
        "tokens_per_second": tf.get("tokens_per_second"),
        "time_to_first_token": tf.get("time_to_first_token"),
        "cached_tokens": (out.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
        "finish_reason": (out.get("choices") or [{}])[0].get("finish_reason"),
    }
    # The drafter's own accounting, when the build reports it.
    for key in ("drafts", "accepted", "acceptance_length", "verify_rounds"):
        if key in tf:
            row[key] = tf[key]
    return row


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--model", default="qwen3.8-flash-next")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--drafter", default=None, help="informational: which drafter this run used")
    p.add_argument("--out", default=None, help="append the row here and compare with prior rows")
    a = p.parse_args()

    row = run(a)
    print(json.dumps(row))
    if a.out:
        path = Path(a.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        shas = {r["label"]: r["token_sha256"] for r in rows if r.get("token_sha256")}
        if len(shas) > 1:
            same = len(set(shas.values())) == 1
            print(
                f"{'OK' if same else 'MISMATCH'}: {len(shas)} paths, "
                f"token_sha256 {'identical' if same else 'DIFFER'} -> {shas}",
                file=sys.stderr,
            )
            return 0 if same else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
