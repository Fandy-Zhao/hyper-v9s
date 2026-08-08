#!/usr/bin/env bash
# Compose UCIT official 6-task continual learning run (spec §25).
#
# Tasks 0..5 in the official sequence (ImageNet-R, ArxivQA, VizWiz,
# IconQA, CLEVR, Flickr30k) via the unified runner
# compose/experiments/task_run.py with configs/compose_ucit.yaml.
#
# The runner is stage-level resumable: interrupted tasks resume from the
# next incomplete stage. Resume the whole run with resume_run.sh.
#
#   usage: six_task_run.sh [--gpus 0,1,2,3]
set -euo pipefail

REPO=/home/zhaozhuofan/Hyper-LlaVA
PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
RUN_ROOT="${RUN_ROOT:-${REPO}/experiments/runs/compose_ucit}"
GPUS="${GPUS:-0}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

cd "${REPO}"
exec "${PY}" compose/experiments/task_run.py \
  --config configs/compose_ucit.yaml \
  --root "${RUN_ROOT}" \
  --first-task 0 \
  --last-task 5 \
  --gpus "${GPUS}"
