#!/usr/bin/env bash
# Autoregressive-only bench run, one arm, one server, by pidfile.
#
#   ./ar_run.sh <arm-name> <out.json> [extra serve flags...]
#
# The server is this worktree's binary with --no-drafts and no --drafter: plain autoregressive decode,
# which is the only thing a quantization arm should be judged on. Port 8099, refused if already in use.
# Logs under /tmp/ar-<arm>/.
set -uo pipefail
arm="${1:?arm name}"; out="${2:?out json}"; shift 2
BIN="$HOME/Projects/tf-wt/fp8alloc/zig-out/native/bin/tensorfold-native"
CKPT="$HOME/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300"
PORT=8099
LOG="/tmp/ar-$arm"
PIDFILE="$LOG/server.pid"
mkdir -p "$LOG"

if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "port $PORT already serving — refusing" >&2; exit 2
fi
[ -x "$BIN" ] || { echo "no binary at $BIN — build first" >&2; exit 2; }

"$BIN" serve "$CKPT" --host 127.0.0.1 --port "$PORT" --name qwen3.8-flash-next \
  --context 16384 --parallel 1 --kv-cache fp8 --no-drafts "$@" > "$LOG/serve.log" 2>&1 &
echo $! > "$PIDFILE"

for i in $(seq 1 120); do
  curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo "server died:" >&2; tail -20 "$LOG/serve.log" >&2; exit 1; }
  sleep 3
done
curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || { echo "never became ready" >&2; tail -20 "$LOG/serve.log" >&2; exit 1; }

python3 "$(dirname "$0")/ar_bench.py" --arm "$arm" --log "$LOG/serve.log" --out "$out" \
  --max-tokens "${MAX_TOKENS:-1024}"
rc=$?

kill "$(cat "$PIDFILE")" 2>/dev/null
for _ in $(seq 1 30); do kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break; sleep 1; done
kill -9 "$(cat "$PIDFILE")" 2>/dev/null || true
rm -f "$PIDFILE"
echo "server stopped (rc=$rc)"
exit $rc
