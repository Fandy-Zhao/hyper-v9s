#!/usr/bin/env bash
# Compose P1-Real: one-process-per-GPU dynamic queue on authorized GPUs 4-7.
# See outputs/compose_p1_real_20260803T120000Z/reports/p1_real_dataset_audit.md
# for the GPU authorization record.
set -euo pipefail

ROOT=/home/zhaozhuofan/Hyper-LlaVA
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/outputs/compose_p1_real_20260803T120000Z}
MANIFEST=${MANIFEST:-${OUTPUT_ROOT}/configs/compose_p1_real_tasks.jsonl}
PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

cd "${ROOT}"
# GPUs 0-3 are occupied by an unrelated user's long-running jobs; the user
# explicitly authorized physical GPUs 4-7 for this task on 2026-08-03.
exec "${PY}" -m compose.experiments.scheduler \
  --manifest "${MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --devices "${DEVICES:-4,5,6,7}" \
  --min-free-mib "${MIN_FREE_MIB:-8000}" \
  --poll-seconds "${POLL_SECONDS:-30}"
