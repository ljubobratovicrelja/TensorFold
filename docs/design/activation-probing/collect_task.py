#!/usr/bin/env python3
"""One task through the served model: request, wall-clock cap, response and timings on disk.

    python3 collect_task.py <category> <prompt-file> <out-dir> <max_tokens> [port]

Writes <out-dir>/prompt.md, request.json, response.json and timing.json. Never raises on a
truncated or aborted generation: the timing file says what happened.
"""
import json
import sys
import time
import urllib.request
import urllib.error

def main() -> int:
    cat, prompt_file, out_dir, max_tokens = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    port = int(sys.argv[5]) if len(sys.argv) > 5 else 8099
    deadline = float(sys.argv[6]) if len(sys.argv) > 6 else 170.0  # seconds of generation
    prompt = open(prompt_file, "r", encoding="utf-8").read().strip()

    body = {
        "model": "qwen3.8-flash-next",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    open(f"{out_dir}/prompt.md", "w", encoding="utf-8").write(prompt + "\n")
    json.dump(body, open(f"{out_dir}/request.json", "w"), indent=1)

    t0 = time.time()
    status, payload, note = "unknown", None, ""
    try:
        with urllib.request.urlopen(req, timeout=deadline + 60) as r:
            payload = json.loads(r.read())
        status = "ok"
    except urllib.error.HTTPError as e:
        status, note = "http_error", e.read().decode(errors="replace")[:2000]
    except Exception as e:  # a socket timeout past the cap, a reset by the kill
        status, note = "client_error", f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0

    text = ""
    if payload:
        ch = (payload.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        text = (msg.get("content") or "")
        reasoning = msg.get("reasoning_content") or ""
        json.dump(payload, open(f"{out_dir}/response.json", "w"), indent=1)
        open(f"{out_dir}/output.md", "w", encoding="utf-8").write(text + "\n")
        if reasoning:
            open(f"{out_dir}/reasoning.md", "w", encoding="utf-8").write(reasoning + "\n")
    timing = {
        "category": cat,
        "status": status,
        "note": note,
        "elapsed_s": round(elapsed, 2),
        "max_tokens": max_tokens,
        "usage": (payload or {}).get("usage"),
        "finish_reason": ((payload or {}).get("choices") or [{}])[0].get("finish_reason"),
        "tensorfold": (payload or {}).get("tensorfold"),
        "chars": len(text),
        "words": len(text.split()),
    }
    json.dump(timing, open(f"{out_dir}/timing.json", "w"), indent=1)
    print(json.dumps(timing))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
