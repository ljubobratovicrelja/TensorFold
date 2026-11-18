#!/usr/bin/env bash
# Start/stop the dflash2 worktree's server on a test port, by pidfile.
#
# Never `pkill -f tensorfold-native`: the pattern matches the caller's own `bash -c` and kills the
# shell running the command. A pidfile is the only safe handle.
#
#   ./tfsrv.sh start <port> [extra serve flags...]
#   ./tfsrv.sh stop
#   ./tfsrv.sh wait <port>     # poll /v1/models until ready
set -uo pipefail
BIN="$HOME/Projects/tf-wt/dflash2/zig-out/native/bin/tensorfold-native"
CKPT="$HOME/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300"
PIDFILE=/tmp/tfsrv-df2.pid
LOGDIR=/tmp/tfsrv-df2

case "${1:-}" in
start)
  port="${2:?port}"
  shift 2
  mkdir -p "$LOGDIR"
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "already running: $(cat "$PIDFILE")"; exit 1
  fi
  nohup "$BIN" serve "$CKPT" --host 127.0.0.1 --port "$port" \
    --name qwen3.8-flash-next --context 32768 --parallel 2 "$@" \
    > "$LOGDIR/serve-$port.log" 2>&1 &
  echo $! > "$PIDFILE"
  echo "started pid $(cat "$PIDFILE") on :$port -> $LOGDIR/serve-$port.log"
  ;;
stop)
  if [ -f "$PIDFILE" ]; then
    pid="$(cat "$PIDFILE")"
    if kill -0 "$pid" 2>/dev/null; then kill "$pid"; fi
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$PIDFILE"
    echo "stopped $pid"
  else
    echo "no pidfile"
  fi
  ;;
wait)
  port="${2:?port}"
  for i in $(seq 1 60); do
    if curl -s -m 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      echo "ready after $((i*3))s"; exit 0
    fi
    sleep 3
  done
  echo "NOT ready after 180s"; tail -5 "$LOGDIR/serve-$port.log"; exit 1
  ;;
*)
  echo "usage: $0 {start <port> [flags] | stop | wait <port>}"; exit 2 ;;
esac
