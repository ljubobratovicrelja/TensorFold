#!/usr/bin/env python3
"""Harvest DFlash2 drafter training fixtures from a corpus, and measure the rate.

One fixture = one prefill pass. The engine writes the tap ring (`ring.bin`, 2064 slots x 5 taps
x 2560 bf16) plus the ids it actually read (`ids.bin`) whenever `TF_FLASHNEXT_TAP_DUMP` names a
directory; this script feeds the corpus in one-chunk passes and moves each dump out of the way
before the next request overwrites it. `draft-check` already consumes exactly this format, so
what lands here is training data, not a new format.

The pass must fit one chunk (`cuda_prompt.zig`: a window's hooks take one chunk, solo 8192 rows),
so `--chunk-tokens` stays well under that. The dump is the whole ring whatever the prompt length,
so short chunks waste bytes: 2048 tokens per pass is the efficient size.

    TF_FLASHNEXT_TAP_DUMP=/tmp/harv/dump ./tensorfold-native serve T --port 8101 --no-drafts &
    python3 tools/dflash2/harvest.py --dump-dir /tmp/harv/dump --out /tmp/harv/out --corpus DOCS...

Bytes: a full ring is 52,838,400 B per pass = 25.8 KB a token at 2048 tokens.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

def corpus_chunks(paths: list[Path], tokens: int, tokenizer_dir: str | None) -> list[str]:
    """Split the corpus into prompts of at most `tokens` tokens, on paragraph boundaries.

    Words are a bad proxy here: markdown and code tokenise near 2 tokens a word, so a word budget
    overruns the pass limit. The checkpoint's own tokenizer counts exactly when it is available.
    """
    enc = None
    if tokenizer_dir:
        from tokenizers import Tokenizer  # local to the harvest, not a repo dependency

        enc = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))

    def count(s: str) -> int:
        if enc is not None:
            return len(enc.encode(s, add_special_tokens=False).ids)
        return int(len(s) / 3.0)

    out: list[str] = []
    buf = ""
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="replace")
        for para in text.split("\n\n"):
            if not para.strip():
                continue
            cand = f"{buf}\n\n{para}" if buf else para
            if count(cand) > tokens and buf:
                out.append(buf)
                buf, cand = "", para
            # one paragraph longer than a chunk is split on characters until it fits
            while count(cand) > tokens:
                lo, hi = 1, len(cand)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if count(cand[:mid]) <= tokens:
                        lo = mid
                    else:
                        hi = mid - 1
                out.append(cand[:lo])
                cand = cand[lo:]
            buf = cand
    if buf.strip():
        out.append(buf)
    return out


def post(base: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8101/v1")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--dump-dir", required=True, help="the directory the server was started with")
    ap.add_argument("--out", required=True, help="where chunk-XXXX/ fixtures are kept")
    ap.add_argument("--corpus", nargs="+", required=True, help="text files to harvest")
    ap.add_argument("--chunk-tokens", type=int, default=1900, help="target prompt tokens a pass")
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = all")
    ap.add_argument("--budget-gb", type=float, default=0.0, help="stop before the output passes this")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--tokenizer-dir", default="", help="a checkpoint dir holding tokenizer.json, for exact chunking")
    a = ap.parse_args()

    dump = Path(a.dump_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    chunks = corpus_chunks([Path(p) for p in a.corpus], a.chunk_tokens, a.tokenizer_dir or None)
    if a.max_chunks:
        chunks = chunks[: a.max_chunks]
    print(f"corpus: {len(chunks)} chunks of ~{a.chunk_tokens} tokens -> {out}")

    rows: list[dict] = []
    wrote = 0
    t_start = time.time()
    for i, text in enumerate(chunks):
        if a.budget_gb and wrote >= a.budget_gb * 1e9:
            print(f"budget reached at chunk {i} ({wrote / 1e9:.2f} GB)")
            break
        payload = {
            "model": a.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.time()
        try:
            resp = post(a.base_url, payload, a.timeout)
        except urllib.error.HTTPError as e:  # a refused pass is data too, not a crash
            print(f"[{i:04d}] FAILED HTTP {e.code}: {e.read().decode()[:400]}")
            break
        except Exception as e:
            print(f"[{i:04d}] FAILED: {type(e).__name__}: {e}")
            break
        wall = time.time() - t0
        usage = resp.get("usage") or {}
        tf = resp.get("tensorfold") or {}
        tokens = usage.get("prompt_tokens") or 0
        prefill_s = tf.get("prefill_seconds")

        man = dump / "manifest.json"
        if not man.exists():
            print(f"[{i:04d}] NO DUMP in {dump} (server started without TF_FLASHNEXT_TAP_DUMP?)")
            return 1
        dest = out / f"chunk-{i:04d}"
        dest.mkdir(exist_ok=True)
        ring_bytes = 0
        for name in ("ring.bin", "ids.bin", "manifest.json"):
            src = dump / name
            if src.exists():
                shutil.move(str(src), str(dest / name))
                ring_bytes += (dest / name).stat().st_size
        wrote += ring_bytes
        m = json.loads((dest / "manifest.json").read_text())
        rows.append(
            {
                "chunk": i,
                "tokens": tokens,
                "prefill_s": prefill_s,
                "wall_s": round(wall, 3),
                "bytes": ring_bytes,
                "pos_end": m.get("pos_end"),
                "ids_in_dump": (dest / "ids.bin").stat().st_size // 4,
            }
        )
        rate = f"{tokens / prefill_s:,.0f} tok/s" if prefill_s else "?"
        print(
            f"[{i:04d}] tokens={tokens:5d} prefill={prefill_s}s ({rate}) wall={wall:.2f}s "
            f"bytes={ring_bytes / 1e6:.1f} MB pos_end={m.get('pos_end')}"
        )

    if not rows:
        print("nothing harvested")
        return 1
    total_tokens = sum(r["tokens"] for r in rows)
    engine_s = sum(r["prefill_s"] or 0 for r in rows)
    elapsed = time.time() - t_start
    print("\n=== harvest summary ===")
    print(f"chunks            {len(rows)}")
    print(f"tokens            {total_tokens:,}")
    print(f"engine prefill    {engine_s:.1f} s -> {total_tokens / engine_s:,.0f} tok/s" if engine_s else "")
    print(f"wall              {elapsed:.1f} s -> {total_tokens / elapsed:,.0f} tok/s")
    print(f"bytes written     {wrote / 1e9:.2f} GB -> {wrote / max(total_tokens, 1) / 1024:.1f} KB/token")
    print(f"per hour at this rate: {total_tokens / elapsed * 3600:,.0f} tokens/h")
    (out / "harvest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
