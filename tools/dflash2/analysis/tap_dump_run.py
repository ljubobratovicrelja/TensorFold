#!/usr/bin/env python3
"""One prefill against a TF_FLASHNEXT_TAP_DUMP server, then a look at what landed.

The reader of this is `tools/dflash2/analysis/tap_accept.py`: it wants a teacher-forced sequence of
tokens long enough to fill the drafter's 2048-row window, plus the engine's own tap rows for
that same sequence. So this sends a long, deterministic prompt and asks for a single token —
the prefill is the measurement, the completion is not — and then prints the dump's geometry.

The prompt text comes from the design doc beside it, so a rerun tokenizes to the same ids.

    python3 tools/dflash2/analysis/tap_dump_run.py --dump-dir /tmp/tapdump-readout
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path


def prompt_text(path: Path, words: int) -> str:
    """A deterministic slice of a doc as prose: words, not tokens, so it is stable across runs."""
    text = path.read_text(encoding="utf-8")
    body = " ".join(line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#"))
    return " ".join(body.split()[:words])


def post(base: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8099/v1")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--dump-dir", required=True, help="the directory the server was started with")
    ap.add_argument("--words", type=int, default=700, help="prompt words (~1.3 tokens a word)")
    ap.add_argument("--doc", default=str(Path(__file__).resolve().parents[3] / "docs/design/flashnext-dflash2-drafter.md"))
    a = ap.parse_args()

    text = prompt_text(Path(a.doc), a.words)
    payload = {
        "model": a.model,
        "messages": [{"role": "user", "content": "Continue this text in your own words:\n\n" + text}],
        "max_tokens": 1,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.time()
    out = post(a.base_url, payload)
    usage = out.get("usage", {})
    print(f"prefill: prompt_tokens={usage.get('prompt_tokens')} wall={time.time() - started:.2f}s")

    tf = out.get("tensorfold") or {}
    if tf:
        print(f"tensorfold: prefill_seconds={tf.get('prefill_seconds')} token_sha256={tf.get('token_sha256')}")

    d = Path(a.dump_dir)
    if not d.is_dir():
        print(f"NO DUMP: {d} does not exist (did the server start with TF_FLASHNEXT_TAP_DUMP?)")
        return 1
    files = sorted(p.name for p in d.iterdir())
    print(f"dump files: {files}")
    man_path = d / "manifest.json"
    if not man_path.exists():
        print("NO MANIFEST yet: the pass may have exceeded one chunk, or the capture refused")
        return 1
    man = json.loads(man_path.read_text())
    print("manifest: " + json.dumps(man, sort_keys=True))
    ring = (d / man["ring_file"]).stat().st_size
    ids = (d / man["ids_file"]).stat().st_size // 4
    print(f"ring={ring} bytes ({ring / 1e6:.1f} MB), ids={ids} tokens, prompt_tokens={usage.get('prompt_tokens')}")
    if ids != usage.get("prompt_tokens"):
        print("WARNING: ids in the dump differ from usage.prompt_tokens (chunking or a template change)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
