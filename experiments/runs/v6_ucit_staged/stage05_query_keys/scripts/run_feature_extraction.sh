#!/usr/bin/env bash
set -euo pipefail

cd /home/zhaozhuofan/Hyper-LlaVA
export PYTHONPATH=/home/zhaozhuofan/Hyper-LlaVA
ROOT=experiments/runs/v6_ucit_staged/stage05_query_keys
PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
EXTRACT=compose/cli/extract_query_features.py
IMAGES=/data/dataset/zhaozhuofan/UCIT/datasets
OLD_DATA=/data/dataset/zhaozhuofan/v6_ucit_stage04/full_seed42
NEW_DATA=/data/dataset/zhaozhuofan/v6_ucit_stage05
OLD_CACHE=/data/ckpt/zhaozhuofan/v6_ucit_staged/stage04_oracle_teacher/cache/full_seed42/historical_only
OLD_POST_CACHE=/data/ckpt/zhaozhuofan/v6_ucit_staged/stage04_oracle_teacher/cache/full_seed42/post_task_diagnostic
NEW_CACHE=/data/ckpt/zhaozhuofan/v6_ucit_staged/stage05_query_keys/oracle_direct
FEATURES=/data/ckpt/zhaozhuofan/v6_ucit_staged/stage05_query_keys/features/partial

run_task() {
  local gpu="$1" task_id="$2" task_name="$3"
  mkdir -p "$FEATURES/train" "$FEATURES/validation" "$ROOT/logs/features"
  if [[ ! -f "$FEATURES/train/${task_name}_reused.pt" ]]; then CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$EXTRACT" --questions "$OLD_DATA/$task_name.json" \
    --oracle-cache "$OLD_CACHE/$task_name/direct.json" --images "$IMAGES" --task-id "$task_id" --task-name "$task_name" \
    --post-task-oracle-cache "$OLD_POST_CACHE/$task_name/direct.json" \
    --split train --output "$FEATURES/train/${task_name}_reused.pt" --device cuda:0 \
    > "$ROOT/logs/features/${task_name}_reused.log" 2>&1; fi
  if [[ ! -f "$FEATURES/train/${task_name}_missing.pt" ]]; then CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$EXTRACT" --questions "$NEW_DATA/train_missing/$task_name.json" \
    --oracle-cache "$NEW_CACHE/train_missing/historical_only/$task_name/direct.json" --images "$IMAGES" --task-id "$task_id" --task-name "$task_name" \
    --post-task-oracle-cache "$NEW_CACHE/train_missing/post_task_diagnostic/$task_name/direct.json" \
    --split train --output "$FEATURES/train/${task_name}_missing.pt" --device cuda:0 \
    > "$ROOT/logs/features/${task_name}_missing.log" 2>&1; fi
  if [[ ! -f "$FEATURES/validation/$task_name.pt" ]]; then CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$EXTRACT" --questions "$NEW_DATA/validation/$task_name.json" \
    --oracle-cache "$NEW_CACHE/validation/historical_only/$task_name/direct.json" --images "$IMAGES" --task-id "$task_id" --task-name "$task_name" \
    --post-task-oracle-cache "$NEW_CACHE/validation/post_task_diagnostic/$task_name/direct.json" \
    --split validation --output "$FEATURES/validation/$task_name.pt" --device cuda:0 \
    > "$ROOT/logs/features/${task_name}_validation.log" 2>&1; fi
}

mode="${1:-full}"
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  nvidia-smi
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
} > "$ROOT/manifests/gpu_preflight_feature_extraction_${mode}.txt"

status=0
if [[ "$mode" == early ]]; then
  (run_task 5 0 ImageNet-R; run_task 5 1 ArxivQA) & p5=$!
  (run_task 7 2 VizWiz; run_task 7 3 IconQA) & p7=$!
  for pid in "$p5" "$p7"; do wait "$pid" || status=1; done
elif [[ "$mode" == full ]]; then
  (run_task 4 0 ImageNet-R; run_task 4 4 CLEVR) & p4=$!
  (run_task 5 1 ArxivQA) & p5=$!
  (run_task 6 2 VizWiz; run_task 6 5 Flickr30k) & p6=$!
  (run_task 7 3 IconQA) & p7=$!
  for pid in "$p4" "$p5" "$p6" "$p7"; do wait "$pid" || status=1; done
else
  echo "usage: $0 [early|full]" >&2; exit 2
fi
printf '%s\n' "$status" > "$ROOT/logs/feature_extraction_${mode}.exit"
exit "$status"
