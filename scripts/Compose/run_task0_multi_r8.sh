#!/usr/bin/env bash
# Task0 multi rank-8 expert experiment driver (spec §1-§19).
# Physical GPUs 4,5,6,7 only (never 0-3).
#   run_task0_multi_r8.sh prep|validate|train|train-status|assign|assemble|rms|gen|gen-base|nll|delta-stats|summary
#   run_task0_multi_r8.sh smoke-prep|smoke-train|smoke-eval   (GPU 7, tiny subset)
set -euo pipefail

PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
REPO=/home/zhaozhuofan/Hyper-LlaVA
ROOT="$REPO/experiments/runs/task0_multi_r8_seed42"
SMOKE="$REPO/experiments/runs/task0_multi_r8_seed42_smoke"
cd "$REPO"

phase="${1:?usage: see header}"
shift || true

gpu_free() {
  local gpu="$1" used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
  [[ "$used" -lt 1000 ]] || { echo "GPU $gpu busy (${used} MiB)" >&2; return 1; }
}

case "$phase" in
  prep)
    "$PY" -m compose.experiments.task0_multi_r8_prep prepare --root "$ROOT"
    "$PY" -m compose.experiments.task0_multi_r8_prep validate --root "$ROOT"
    ;;
  train)
    # GPU4 -> single_r8 (1 job), GPU5 -> two_r8 (2 sequential), GPU6 -> four_r8 (4 sequential), GPU7 -> rank48 (1 job)
    gpu_free 4; gpu_free 5; gpu_free 6; gpu_free 7
    nohup "$PY" -m compose.experiments.task0_multi_r8_train launch --root "$ROOT" --gpu 4 >"$ROOT/logs/train_gpu4.log" 2>&1 &
    nohup "$PY" -m compose.experiments.task0_multi_r8_train launch --root "$ROOT" --gpu 5 >"$ROOT/logs/train_gpu5.log" 2>&1 &
    nohup "$PY" -m compose.experiments.task0_multi_r8_train launch --root "$ROOT" --gpu 6 >"$ROOT/logs/train_gpu6.log" 2>&1 &
    nohup "$PY" -m compose.experiments.task0_multi_r8_train launch --root "$ROOT" --gpu 7 >"$ROOT/logs/train_gpu7.log" 2>&1 &
    echo "training launched: GPU4 single_r8, GPU5 two_r8, GPU6 four_r8, GPU7 rank48"
    ;;
  train-status)
    "$PY" -m compose.experiments.task0_multi_r8_train status --root "$ROOT"
    ;;
  assign)
    "$PY" -m compose.experiments.task0_multi_r8_eval assign --root "$ROOT"
    ;;
  assemble)
    "$PY" -m compose.experiments.task0_multi_r8_eval assemble --root "$ROOT"
    ;;
  rms)
    "$PY" -m compose.experiments.task0_multi_r8_eval rms --root "$ROOT" --gpus 7
    ;;
  gen)
    "$PY" -m compose.experiments.task0_multi_r8_eval gen --root "$ROOT" --gpus 4,5,6,7 "$@"
    ;;
  gen-base)
    "$PY" -m compose.experiments.task0_multi_r8_eval gen --root "$ROOT" --gpus 4,5,6,7 --base
    ;;
  nll)
    "$PY" -m compose.experiments.task0_multi_r8_eval nll --root "$ROOT" --gpus 4,5,6,7
    ;;
  delta-stats)
    "$PY" -m compose.experiments.task0_multi_r8_eval delta-stats --root "$ROOT"
    ;;
  summary)
    "$PY" -m compose.experiments.task0_multi_r8_eval summary --root "$ROOT"
    ;;
  # ---- smoke (GPU 7 only, tiny subset, separate root) ----
  smoke-prep)
    "$PY" -m compose.experiments.task0_multi_r8_prep prepare --root "$SMOKE"
    "$PY" -m compose.experiments.task0_multi_r8_prep validate --root "$SMOKE"
    ;;
  smoke-train)
    gpu_free 7
    for cfg in single_r8 two_r8 four_r8 rank48; do
      "$PY" -m compose.experiments.task0_multi_r8_train launch \
        --root "$SMOKE" --gpu 7 --config "$cfg" --max-steps 3
    done
    ;;
  smoke-eval)
    "$PY" -m compose.experiments.task0_multi_r8_eval assign --root "$SMOKE"
    "$PY" -m compose.experiments.task0_multi_r8_eval assemble --root "$SMOKE"
    "$PY" -m compose.experiments.task0_multi_r8_eval rms --root "$SMOKE" --gpus 7 --max-samples 12
    "$PY" -m compose.experiments.task0_multi_r8_eval gen --root "$SMOKE" --gpus 7 \
      --config single_r8 --label single:0 --max-samples 24
    "$PY" -m compose.experiments.task0_multi_r8_eval nll --root "$SMOKE" --gpus 7 \
      --config single_r8 --max-samples 12 --batch-size 4
    "$PY" -m compose.experiments.task0_multi_r8_eval delta-stats --root "$SMOKE"
    "$PY" -m compose.experiments.task0_multi_r8_eval summary --root "$SMOKE" --max-samples 24
    ;;
  *)
    echo "unknown phase: $phase" >&2; exit 2
    ;;
esac
