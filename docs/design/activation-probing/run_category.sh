#!/usr/bin/env bash
# One category: boot the engine with the probes pointed at this category's directory, serve exactly
# one request, stop the engine. Usage: run_category.sh <category> <max_tokens> [deadline_s]
set -u
CAT="$1"; MAXTOK="$2"; DEADLINE="${3:-170}"
ROOT="/home/relja/Projects/Qwen3.8-Flash-Next-DFlash2-GB10/activations"
ENGINE="/home/relja/Projects/tf-wt/dflash2"
SNAP="$HOME/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300"
OUT="$ROOT/data/$CAT"
LOG="$ROOT/logs/$CAT.log"
PORT=8099

mkdir -p "$OUT" "$ROOT/logs"
rm -rf "$OUT/experts" "$OUT/hidden"
cd "$ENGINE" || exit 1

TF_FLASHNEXT_WINDOW_GRAPH=0 \
TF_FLASHNEXT_EXPERT_DUMP="$OUT/experts" \
TF_FLASHNEXT_HIDDEN_DUMP="$OUT/hidden" \
TF_FLASHNEXT_ROUTER_LOGITS=128 \
  ./zig-out/native/bin/tensorfold-native serve "$SNAP" \
    --port "$PORT" --parallel 1 --context 8192 --kv-cache fp8 > "$LOG" 2>&1 &
PID=$!
echo "engine pid $PID, log $LOG"

ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$PID" 2>/dev/null; then echo "engine died"; tail -20 "$LOG"; exit 1; fi
  if curl -s -m 2 -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then ready=1; break; fi
  sleep 3
done
[ "$ready" = 1 ] || { echo "engine never answered"; tail -20 "$LOG"; kill "$PID"; exit 1; }
echo "engine ready"

python3 "$ROOT/collect_task.py" "$CAT" "$OUT/prompt_source.md" "$OUT" "$MAXTOK" "$PORT" "$DEADLINE"
RC=$?

sleep 1
kill "$PID" 2>/dev/null
for _ in $(seq 1 20); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
kill -9 "$PID" 2>/dev/null
echo "engine stopped (rc=$RC)"
du -sh "$OUT/experts" "$OUT/hidden" 2>/dev/null
ls "$OUT/experts" | wc -l
exit "$RC"
