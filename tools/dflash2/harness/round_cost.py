#!/usr/bin/env python3
"""Same server, same prompt, drafts on and off: where the round time actually goes.

DFlash2 costs 0.3 ms a round in the drafter and yet a drafting request runs eight times slower
than MTP. Either the target's verify pass over the block is the bill, or the tap hooks this branch
added are. The only way to tell them apart is one server, one binary, `draft` toggled per request.
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 160
PROMPT = "Write a short paragraph about the engineering of a GPU inference server."


def run(draft: bool, label: str) -> dict:
    body = json.dumps(
        {
            "model": "qwen3.8-flash-next",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": TOKENS,
            "temperature": 0.0,
            "draft": draft,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    wall = time.perf_counter() - t0
    n = out["usage"]["completion_tokens"]
    return {
        "label": label,
        "draft": draft,
        "tokens": n,
        "wall_s": round(wall, 3),
        "tok_s": round(n / wall, 2),
    }


for draft, label in ((False, "nodrafts"), (True, "drafting"), (False, "nodrafts-again")):
    print(json.dumps(run(draft, label)), flush=True)
