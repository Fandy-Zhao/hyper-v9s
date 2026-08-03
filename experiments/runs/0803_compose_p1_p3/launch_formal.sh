#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhaozhuofan/Hyper-LlaVA
OUT="$ROOT/outputs/compose_p1_p3_20260803T090000Z"
cd "$ROOT"
if [[ -s "$OUT/logs/formal_scheduler.pid" ]]; then
  EXISTING_PID="$(cat "$OUT/logs/formal_scheduler.pid")"
  if kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "scheduler already running as PID $EXISTING_PID" >&2
    exit 1
  fi
fi
echo "$$" > "$OUT/logs/formal_scheduler.pid"
exec env OUTPUT_ROOT="$OUT" bash scripts/run_compose_p1_p3_4gpu.sh \
  > "$OUT/logs/formal_scheduler.stdout.log" \
  2> "$OUT/logs/formal_scheduler.stderr.log" \
  < /dev/null
