#!/usr/bin/env bash
set -euo pipefail

cd /home/zhaozhuofan/Hyper-LlaVA
export PYTHONPATH=/home/zhaozhuofan/Hyper-LlaVA
ROOT=experiments/runs/v6_ucit_staged/stage05_query_keys
DATA=/data/dataset/zhaozhuofan/v6_ucit_stage05
CACHE=/data/ckpt/zhaozhuofan/v6_ucit_staged/stage05_query_keys/oracle_direct
STAGE04=experiments/runs/v6_ucit_staged/stage04_oracle_teacher
BASE=/data/ckpt/zhaozhuofan/models/llava-v1.5-7b
FULL=/data/ckpt/zhaozhuofan/v6_ucit_staged/stage01_baseline/checkpoints
IMAGES=/data/dataset/zhaozhuofan/UCIT/datasets
PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
GEN="$STAGE04/scripts/oracle_generate.py"
DIRECT_HASH=10f2cdce7726e9f5cecdcebbd056555cd98393415e9665a5a80a4ce1f6b99fe5

visible_for() {
  local count="$1" out="" i
  for ((i=0; i<count; i++)); do [[ -z "$out" ]] || out+=,; out+="$i"; done
  printf '%s' "$out"
}

run_one() {
  local gpu="$1" task_id="$2" task_name="$3" subset="$4" scope="$5"
  local kind=hyper checkpoint visible questions split
  questions="$DATA/$subset/$task_name.json"
  split=train
  [[ "$subset" == validation ]] && split=validation
  if [[ "$scope" == historical_only ]]; then
    visible="$(visible_for "$task_id")"
    if ((task_id == 0)); then kind=base; checkpoint="$BASE"; else checkpoint="$FULL/full_gb24_task${task_id}_llava_lora_ours"; fi
  else
    visible="$(visible_for "$((task_id + 1))")"
    checkpoint="$FULL/full_gb24_task$((task_id + 1))_llava_lora_ours"
  fi
  local out="$ROOT/oracle/$subset/$scope/$task_name"
  local cache="$CACHE/$subset/$scope/$task_name/direct.json"
  mkdir -p "$out" "$(dirname "$cache")"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$GEN" \
    --checkpoint-kind "$kind" --checkpoint "$checkpoint" --questions "$questions" --images "$IMAGES" \
    --task-id "$task_id" --task-name "$task_name" --split "$split" --temporal-scope "$scope" \
    --visible-experts "$visible" --config "$STAGE04/configs/oracle_direct.yaml" --config-hash "$DIRECT_HASH" \
    --cache-file "$cache" --summary-file "$out/direct_summary.json" --batch-size 2 \
    > "$out/direct.log" 2>&1
}

run_task() {
  local gpu="$1" task_id="$2" task_name="$3" subset scope
  for subset in train_missing validation; do
    for scope in historical_only post_task_diagnostic; do
      run_one "$gpu" "$task_id" "$task_name" "$subset" "$scope"
    done
  done
}

mkdir -p "$ROOT/manifests" "$ROOT/logs"
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  nvidia-smi
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
} > "$ROOT/manifests/gpu_preflight_oracle_expansion.txt"

status=0
(run_task 4 0 ImageNet-R; run_task 4 4 CLEVR) & p4=$!
(run_task 5 1 ArxivQA) & p5=$!
(run_task 6 2 VizWiz; run_task 6 5 Flickr30k) & p6=$!
(run_task 7 3 IconQA) & p7=$!
for pid in "$p4" "$p5" "$p6" "$p7"; do wait "$pid" || status=1; done
printf '%s\n' "$status" > "$ROOT/logs/oracle_expansion.exit"
exit "$status"
