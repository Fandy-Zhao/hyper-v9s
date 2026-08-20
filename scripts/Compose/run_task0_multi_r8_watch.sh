#!/usr/bin/env bash
# Staggered-share watcher for the Task0 multi-r8 experiment.
#
# Physical GPUs 4-7 are shared with the v62 adaptive-aggregation experiment
# (its scheduler keeps all four GPUs saturated).  This watcher polls GPU
# memory and, once all of GPU 4-7 have been free (<1 GiB) for a sustained
# window (default 6 min), claims them and runs the full pipeline:
#   smoke-train + smoke-eval (GPU 7) -> formal training (4 GPUs, parallel)
#   -> assign -> assemble -> rms -> gen (+base) -> nll -> delta-stats
#   -> summary -> report
# Every stage is resumable (markers / complete files); if another job
# re-occupies a GPU mid-pipeline, the failing stage backs off and the
# watcher waits for a fresh sustained-free window before retrying.
set -euo pipefail

PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
REPO=/home/zhaozhuofan/Hyper-LlaVA
ROOT="$REPO/experiments/runs/task0_multi_r8_seed42"
DRIVER="bash $REPO/scripts/Compose/run_task0_multi_r8.sh"
WATCH_LOG="$REPO/experiments/runs/task0_multi_r8_seed42/logs/watch.log"
SUSTAINED_MIN="${SUSTAINED_MIN:-6}"
cd "$REPO"
mkdir -p "$ROOT/logs"

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >>"$WATCH_LOG"; echo "[watch] $*"; }

gpu_free() {
  local gpu="$1" used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
  [[ "$used" -lt 1000 ]]
}

all_gpus_free() {
  for gpu in 4 5 6 7; do
    gpu_free "$gpu" || return 1
  done
}

any_gpu_free() {
  for gpu in 4 5 6 7; do
    gpu_free "$gpu" && return 0
  done
  return 1
}

# ---- claim: wait for a sustained free window ---------------------------
claim() {
  local free_since=0
  say "waiting for sustained GPU 4-7 free window (${SUSTAINED_MIN} min)"
  while :; do
    if all_gpus_free; then
      free_since=$((free_since + 1))
      say "GPUs free for ${free_since} min"
      if [[ "$free_since" -ge "$SUSTAINED_MIN" ]]; then
        say "claiming GPUs 4-7"
        return 0
      fi
    else
      free_since=0
      say "GPUs busy; resetting free counter"
    fi
    sleep 60
  done
}

# ---- guarded stage runner: back off when a GPU gets re-occupied ---------
run_stage() {
  local name="$1"; shift
  say "stage: ${name}"
  while :; do
    if "$@"; then
      say "stage ok: ${name}"
      return 0
    fi
    say "stage failed (likely GPU contention): ${name}; backing off"
    claim
  done
}

# Launch can die mid-way (one job failure kills its GPU's launch process and
# the remaining jobs stay PENDING), so wait_training is resumable: if no
# train_compose process is alive but jobs remain PENDING, claim the GPUs and
# relaunch -- completed jobs are skipped via complete.txt markers.
wait_training() {
  local stuck=0 pending running
  say "waiting for training (resumable)"
  while :; do
    pending=$("$PY" -m compose.experiments.task0_multi_r8_train status --root "$ROOT" 2>/dev/null | grep -c PENDING || true)
    running=$(pgrep -fc "train_compose" || true)
    if [[ "$pending" == "0" ]]; then
      say "training complete (0 pending)"
      return 0
    fi
    if [[ "$running" -gt 0 ]]; then
      stuck=0
      say "training in progress (${pending} pending, ${running} procs)"
    else
      stuck=$((stuck + 1))
      say "no training proc alive; ${pending} pending (stuck=${stuck}/3)"
      if [[ "$stuck" -ge 3 ]]; then
        say "training stalled; reclaiming GPUs and relaunching"
        claim
        run_stage "train-relaunch" $DRIVER train
        stuck=0
      fi
    fi
    sleep 300
  done
}

# -------------------------------------------------------------------------
say "=== watcher start ==="
claim

# 1. smoke on GPU 7 (tiny subset; other GPUs stay idle while we validate)
say "smoke: train (GPU 7)"
while :; do
  if gpu_free 7 && $DRIVER smoke-train; then break; fi
  say "smoke-train failed or GPU 7 busy; backing off"
  claim
done
say "smoke: eval (GPU 7)"
while :; do
  if gpu_free 7 && $DRIVER smoke-eval; then break; fi
  say "smoke-eval failed or GPU 7 busy; backing off"
  claim
done
say "SMOKE COMPLETE"

# 2. formal training on all four GPUs (driver's train phase re-checks GPU
# freeness itself before launching; completed jobs are skipped)
run_stage "train-launch" $DRIVER train
wait_training
say "TRAINING COMPLETE"

# 3. evaluation chain (resumable)
run_stage "assign" "$PY" -m compose.experiments.task0_multi_r8_eval assign --root "$ROOT"
run_stage "assemble" "$PY" -m compose.experiments.task0_multi_r8_eval assemble --root "$ROOT"
run_stage "rms" $DRIVER rms
run_stage "gen" $DRIVER gen
run_stage "gen-base" $DRIVER gen-base
run_stage "nll" $DRIVER nll
run_stage "delta-stats" $DRIVER delta-stats
run_stage "summary" $DRIVER summary
run_stage "report" "$PY" -m compose.experiments.task0_multi_r8_report --root "$ROOT"

say "=== watcher complete: report at $ROOT/reports/task0_multi_r8_report.md ==="
