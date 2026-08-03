#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhaozhuofan/Hyper-LlaVA
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/outputs/compose_p1_p3_20260803T090000Z}
MANIFEST=${MANIFEST:-${OUTPUT_ROOT}/configs/tasks.jsonl}
PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

cd "${ROOT}"
exec "${PY}" -m compose.experiments.scheduler \
  --manifest "${MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --devices auto \
  --min-free-mib "${MIN_FREE_MIB:-18000}" \
  --poll-seconds "${POLL_SECONDS:-30}"
