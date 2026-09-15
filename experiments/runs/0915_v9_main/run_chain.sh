#!/usr/bin/env bash
# The formal V9-S continual sequence: ImageNet-R -> ArxivQA -> VizWiz -> IconQA
# -> CLEVR -> Flickr30k, 0..5, in that order, one task per run_task.sh.
#
#   run_chain.sh              # 0..5, skipping anything already committed
#   run_chain.sh 3 5          # a range, for restarting after a failure
#
# 52.0 h of optimizer steps in total at the measured 28.36 s/step on one RTX
# 4090 (6601 steps over 211244 samples), plus per-task startup, query-cache
# encoding and the §30 calibration.  Restarting the script resumes the task that
# was interrupted and skips the ones that committed.
set -euo pipefail

BASE=/root/autodl-tmp/Hyper-LlaVA/experiments/runs/0915_v9_main
GPUS="0"
if [ "${1:-}" = "--gpus" ]; then GPUS="${2:?--gpus requires a comma-separated list}"; shift 2; fi
FROM="${1:-0}"
TO="${2:-5}"

for TASK in $(seq "$FROM" "$TO"); do
  echo "[chain] ===== $(date -Is) starting task $TASK ====="
  bash "$BASE/run_task.sh" "$TASK" "$GPUS"
done
echo "[chain] ===== $(date -Is) sequence complete ====="
