#!/usr/bin/env bash
# One generated fixture round through the engine's own drafter, dumped stage by stage.
#
# The path is the production one, not a second forward: the server loads the served checkpoint, a
# one-token prompt makes the prefill capture a tap ring, and the draft-check hook fires at the end
# of that pass. TF_FLASHNEXT_DRAFT_FIXTURE swaps the captured ring for ours, so the round the
# engine runs is `fwd.Draft.draft` — the real kernels over the real 81-tensor checkpoint — on bytes
# the reference side also read. Nothing about the prompt reaches the round.
#
#   ./run_engine.sh <fixture-case-dir> <out-dir>
#
# Exit: 0 when <out-dir>/stages.json was written, 1 otherwise. Only this script's own server pid is
# killed, by pid, on the way out.
set -uo pipefail

WT=/home/relja/Projects/tf-wt/probe-layer
BIN=$WT/zig-out/native/bin/tensorfold-native
CKPT=$HOME/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300
DRF=$HOME/.cache/huggingface/hub/models--incoai--Qwen3.8-Flash-Next-DFlash2/snapshots/a22a1744e85c17364dcc54a6a420af7135d4c7f6
PORT=8099

FX=$1
OUT=$2
LOG=$OUT/logs
PIDF=$LOG/server.pid
mkdir -p "$LOG"
rm -f "$OUT/stages.json"

export TF_FLASHNEXT_TAP_DUMP=$LOG/tapdump      # the capture the hook hangs off
export TF_FLASHNEXT_CHUNK=2048                 # one chunk, or the capture refuses the pass
export TF_FLASHNEXT_DRAFT_CHECK=1
export TF_FLASHNEXT_DRAFT_FIXTURE=$FX
export TF_FLASHNEXT_DRAFT_STAGES=$OUT
export TF_FLASHNEXT_STAGE_CTX_TENSOR=1         # the fuse's own rows as a tensor, not just a summary
export TF_FLASHNEXT_DRAFTER_DIR=$DRF
export TF_ALIGN_CHECKPOINT=$CKPT
export TF_ALIGN_COMMAND="run_engine.sh $FX $OUT"

if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "refusing: $(cat "$PIDF") is already running" >&2
  exit 1
fi

nohup "$BIN" serve "$CKPT" --host 127.0.0.1 --port $PORT \
  --name qwen3.8-flash-next --context 4096 --parallel 1 > "$LOG/server.log" 2>&1 &
echo $! > "$PIDF"
echo "server pid $(cat "$PIDF"), log $LOG/server.log"

ready=0
for _ in $(seq 1 300); do
  if grep -q '] serving ' "$LOG/server.log" 2>/dev/null; then ready=1; break; fi
  if ! kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "server died before serving" >&2; break; fi
  sleep 1
done
if [ "$ready" != 1 ]; then
  echo "NOT READY; tail of the log:" >&2
  tail -30 "$LOG/server.log" >&2
  kill "$(cat "$PIDF")" 2>/dev/null
  exit 1
fi
echo "serving"

curl -s -m 900 -o "$LOG/response.json" -w 'http_status=%{http_code}\n' \
  "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0}'

rc=1
if [ -s "$OUT/stages.json" ]; then rc=0; fi
echo "stages.json: $([ $rc = 0 ] && echo written || echo MISSING)"

kill "$(cat "$PIDF")" 2>/dev/null
for _ in $(seq 1 30); do kill -0 "$(cat "$PIDF")" 2>/dev/null || break; sleep 1; done
kill -9 "$(cat "$PIDF")" 2>/dev/null
rm -f "$PIDF"
echo "server stopped"
exit $rc
