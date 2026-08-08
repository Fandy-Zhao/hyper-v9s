#!/usr/bin/env bash
# Compose UCIT resume (spec §25): re-invoke the unified runner on the same
# run root. Stage .done markers make every task idempotent, so the run
# continues from the first incomplete stage of FIRST_TASK.
#
#   usage: resume_run.sh [TASK_ID]   (default: 0 = whole run)
set -euo pipefail

REPO=/home/zhaozhuofan/Hyper-LlaVA
PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
RUN_ROOT="${RUN_ROOT:-${REPO}/experiments/runs/compose_ucit}"
GPUS="${GPUS:-0}"
FIRST_TASK="${1:-0}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

cd "${REPO}"
exec "${PY}" compose/experiments/task_run.py \
  --config configs/compose_ucit.yaml \
  --root "${RUN_ROOT}" \
  --first-task "${FIRST_TASK}" \
  --last-task 5 \
  --gpus "${GPUS}"
