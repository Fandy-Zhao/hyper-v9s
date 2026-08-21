#!/usr/bin/env bash
set -euo pipefail

PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
REPO=/home/zhaozhuofan/Hyper-LlaVA
ROOT="$REPO/experiments/runs/task0_capacity_chain_seed42"
MODULE=compose.experiments.task0_capacity_chain
LOG="$ROOT/watch.log"
SUSTAINED_MIN="${SUSTAINED_MIN:-6}"
INTERVAL="${INTERVAL:-60}"
cd "$REPO"
mkdir -p "$ROOT"

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

gpu_free() {
  local gpu="$1" used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
  [[ "$used" -lt 1000 ]]
}

all_free() {
  for gpu in 4 5 6 7; do gpu_free "$gpu" || return 1; done
}

existing_multi_r8() {
  pgrep -af 'run_task0_multi_r8_watch\.sh|run_task0_multi_r8\.sh|task0_multi_r8_train|task0_multi_r8_eval' | grep -v 'grep' >/dev/null 2>&1
}

claim() {
  local free=0
  while :; do
    if existing_multi_r8; then
      free=0; say "existing multi-r8 task is still active; capacity chain waiting"
    elif all_free; then
      free=$((free + 1)); say "GPU 4-7 free for ${free}/${SUSTAINED_MIN} checks"
      if [[ "$free" -ge "$SUSTAINED_MIN" ]]; then return 0; fi
    else
      free=0; say "GPU 4-7 not free; resetting claim window"
    fi
    sleep "$INTERVAL"
  done
}

parallel_phase() {
  local phase="$1"; shift
  local -a pids=()
  local -a names=()
  local name gpu
  for name in "$@"; do
    gpu=$($PY - "$name" <<'PY'
import sys
print({'single_r8': 4, 'two_r8': 5, 'four_r8': 6, 'single_r16': 7}[sys.argv[1]])
PY
)
    if [[ "$phase" == smoke && -f "$ROOT/$name/SMOKE_DONE" ]] || [[ "$phase" == train && -f "$ROOT/$name/DONE" ]]; then
      say "skip $phase/$name (marker already exists)"; continue
    fi
    if [[ "$phase" == smoke && -f "$ROOT/$name/SMOKE_FAILED" ]]; then
      say "skip formal/$name because smoke failed"; continue
    fi
    say "launch $phase/$name on GPU $gpu"
    if [[ "$phase" == smoke ]]; then
      nohup "$PY" -m "$MODULE" train --root "$ROOT" --config "$name" --gpu "$gpu" --smoke >"$ROOT/$name/smoke_driver.log" 2>&1 &
    else
      nohup "$PY" -m "$MODULE" train --root "$ROOT" --config "$name" --gpu "$gpu" >"$ROOT/$name/train_driver.log" 2>&1 &
    fi
    pids+=("$!"); names+=("$name")
  done
  local i status=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then say "$phase/${names[$i]} complete"; else say "$phase/${names[$i]} failed; continuing other configs"; status=1; fi
  done
  return 0
}

say '=== Task0 capacity-chain watcher start ==='
if [[ ! -f "$ROOT/reference_hyperllava_config.json" ]]; then
  "$PY" -m "$MODULE" prepare --root "$ROOT"
fi
claim
parallel_phase smoke single_r8 two_r8 four_r8 single_r16

# Retry the claim after smoke so no external job can be overlapped with the
# long formal phase.  The existing multi-r8 watcher is intentionally included
# in the guard for the whole lifecycle.
claim
parallel_phase train single_r8 two_r8 four_r8 single_r16

claim
"$PY" -m "$MODULE" assemble --root "$ROOT" >>"$ROOT/assemble.log" 2>&1 || say 'assembly had one or more failures; continuing'
"$PY" -m "$MODULE" rms --root "$ROOT" >>"$ROOT/rms.log" 2>&1 || say 'RMS diagnostics failed; continuing'
claim
"$PY" -m "$MODULE" evaluate --root "$ROOT" >>"$ROOT/evaluate.log" 2>&1 || say 'capacity evaluation returned nonzero; inspect per-config markers'
claim
"$PY" -m "$MODULE" hyper-eval --root "$ROOT" >>"$ROOT/hyper_eval.log" 2>&1 || say 'Hyper reference evaluation failed; report will preserve the failure'
"$PY" -m "$MODULE" report --root "$ROOT" >>"$ROOT/report.log" 2>&1 || say 'report generation failed'
say "=== Task0 capacity-chain watcher complete: $ROOT/summary/performance_chain.md ==="
