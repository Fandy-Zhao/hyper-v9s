#!/usr/bin/env bash
set -euo pipefail

cd /home/zhaozhuofan/Hyper-LlaVA
export PYTHONPATH=/home/zhaozhuofan/Hyper-LlaVA
ROOT=experiments/runs/v6_ucit_staged/stage04_oracle_teacher
DATA=/data/dataset/zhaozhuofan/v6_ucit_stage04
CACHE=/data/ckpt/zhaozhuofan/v6_ucit_staged/stage04_oracle_teacher/cache
RMS=/data/ckpt/zhaozhuofan/v6_ucit_staged/stage04_oracle_teacher/rms
BASE=/data/ckpt/zhaozhuofan/models/llava-v1.5-7b
FULL=/data/ckpt/zhaozhuofan/v6_ucit_staged/stage01_baseline/checkpoints
MINI=/data/ckpt/zhaozhuofan/v6_ucit_staged/stage03_composition/checkpoints
IMAGES=/data/dataset/zhaozhuofan/UCIT/datasets
PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
SCRIPT="$ROOT/scripts/oracle_generate.py"
DIRECT_HASH=10f2cdce7726e9f5cecdcebbd056555cd98393415e9665a5a80a4ce1f6b99fe5
RMS_HASH=12fa7623df4d392c748a61d8bf980c9610f1dcde25276eaae7b47701a24e0a8d
REQUEST="${1:?usage: run_ucit_oracles.sh smoke|mini2|full_seed42|full_seed42_workerN}"
SUITE="$REQUEST"
if [[ "$REQUEST" == full_seed42_* ]]; then SUITE=full_seed42; fi

case "$REQUEST" in
  smoke|mini2|full_seed42|full_seed42_worker4|full_seed42_worker5|full_seed42_worker6|full_seed42_worker7) ;;
  *) echo "unsupported suite: $REQUEST" >&2; exit 2 ;;
esac

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  nvidia-smi
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
} > "$ROOT/manifests/${REQUEST}_preflight.txt"

visible_for() {
  local count="$1" out="" i
  for ((i=0; i<count; i++)); do
    if [[ -n "$out" ]]; then out+=,; fi
    out+="$i"
  done
  printf '%s' "$out"
}

run_boundary() {
  local gpu="$1" task_id="$2" task_name="$3" scope="$4" kind="$5" checkpoint="$6" visible="$7" questions="$8" calibration="$9" split="${10:-train}"
  local out="$ROOT/$SUITE/$scope/$task_name"
  local cache="$CACHE/$SUITE/$scope/$task_name"
  local rms="$RMS/$SUITE/$scope/$task_name.json"
  mkdir -p "$out" "$cache" "$(dirname "$rms")"
  [[ -e "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; return 1; }
  [[ -f "$questions" ]] || { echo "missing questions: $questions" >&2; return 1; }
  [[ -f "$calibration" ]] || { echo "missing calibration: $calibration" >&2; return 1; }

  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT" \
    --checkpoint-kind "$kind" --checkpoint "$checkpoint" --questions "$questions" \
    --images "$IMAGES" --task-id "$task_id" --task-name "$task_name" --split "$split" \
    --temporal-scope "$scope" --visible-experts "$visible" \
    --config "$ROOT/configs/oracle_direct.yaml" --config-hash "$DIRECT_HASH" \
    --cache-file "$cache/direct.json" --summary-file "$out/direct_summary.json" --batch-size 2 \
    > "$out/direct.log" 2>&1

  local -a rms_args=()
  local visible_count=0
  if [[ -n "$visible" ]]; then
    local commas="${visible//[^,]/}"
    visible_count=$((1 + ${#commas}))
  fi
  if ((visible_count >= 2)); then
    rms_args=(--calibration-questions "$calibration" --rms-statistics "$rms")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT" \
    --checkpoint-kind "$kind" --checkpoint "$checkpoint" --questions "$questions" \
    --images "$IMAGES" --task-id "$task_id" --task-name "$task_name" --split "$split" \
    --temporal-scope "$scope" --visible-experts "$visible" \
    --config "$ROOT/configs/oracle_rms.yaml" --config-hash "$RMS_HASH" \
    "${rms_args[@]}" --cache-file "$cache/rms.json" --summary-file "$out/rms_summary.json" --batch-size 2 \
    > "$out/rms.log" 2>&1
}

run_task_pair() {
  local gpu="$1" task_id="$2" task_name="$3" questions="$4" checkpoint_prefix="$5"
  local historical_kind=hyper historical_checkpoint post_checkpoint
  if ((task_id == 0)); then
    historical_kind=base
    historical_checkpoint="$BASE"
  else
    historical_checkpoint="${checkpoint_prefix}$((task_id))_llava_lora_ours"
  fi
  post_checkpoint="${checkpoint_prefix}$((task_id + 1))_llava_lora_ours"
  run_boundary "$gpu" "$task_id" "$task_name" historical_only "$historical_kind" "$historical_checkpoint" "$(visible_for "$task_id")" "$questions" "$questions"
  run_boundary "$gpu" "$task_id" "$task_name" post_task_diagnostic hyper "$post_checkpoint" "$(visible_for "$((task_id + 1))")" "$questions" "$questions"
}

status=0
if [[ "$SUITE" == smoke ]]; then
  questions="$DATA/smoke/ImageNet-R_validation.json"
  calibration="$DATA/smoke/ImageNet-R_train.json"
  run_boundary 5 0 ImageNet-R historical_only base "$BASE" "" "$questions" "$calibration" validation & p1=$!
  run_boundary 6 0 ImageNet-R post_task_diagnostic hyper "$FULL/full_gb24_task1_llava_lora_ours" 0 "$questions" "$calibration" validation & p2=$!
  for pid in "$p1" "$p2"; do wait "$pid" || status=1; done
elif [[ "$SUITE" == mini2 ]]; then
  image_questions="$DATA/mini2/ImageNet-R.json"
  arxiv_questions="$DATA/mini2/ArxivQA.json"
  run_boundary 7 0 ImageNet-R historical_only base "$BASE" "" "$image_questions" "$image_questions"
  run_boundary 7 0 ImageNet-R post_task_diagnostic hyper "$MINI/mini2_gb24_task1_llava_lora_ours" 0 "$image_questions" "$image_questions"
  run_boundary 7 1 ArxivQA historical_only hyper "$MINI/mini2_gb24_task1_llava_lora_ours" 0 "$arxiv_questions" "$arxiv_questions"
  run_boundary 7 1 ArxivQA post_task_diagnostic hyper "$MINI/mini2_gb24_task2_llava_lora_ours" 0,1 "$arxiv_questions" "$arxiv_questions"
else
  prefix="$FULL/full_gb24_task"
  worker4() { run_task_pair 4 0 ImageNet-R "$DATA/full_seed42/ImageNet-R.json" "$prefix"; run_boundary 4 4 CLEVR historical_only hyper "${prefix}4_llava_lora_ours" "$(visible_for 4)" "$DATA/full_seed42/CLEVR.json" "$DATA/full_seed42/CLEVR.json"; }
  worker5() { run_task_pair 5 1 ArxivQA "$DATA/full_seed42/ArxivQA.json" "$prefix"; run_boundary 5 4 CLEVR post_task_diagnostic hyper "${prefix}5_llava_lora_ours" "$(visible_for 5)" "$DATA/full_seed42/CLEVR.json" "$DATA/full_seed42/CLEVR.json"; }
  worker6() { run_task_pair 6 2 VizWiz "$DATA/full_seed42/VizWiz.json" "$prefix"; run_boundary 6 5 Flickr30k historical_only hyper "${prefix}5_llava_lora_ours" "$(visible_for 5)" "$DATA/full_seed42/Flickr30k.json" "$DATA/full_seed42/Flickr30k.json"; }
  worker7() { run_task_pair 7 3 IconQA "$DATA/full_seed42/IconQA.json" "$prefix"; run_boundary 7 5 Flickr30k post_task_diagnostic hyper "${prefix}6_llava_lora_ours" "$(visible_for 6)" "$DATA/full_seed42/Flickr30k.json" "$DATA/full_seed42/Flickr30k.json"; }
  case "$REQUEST" in
    full_seed42_worker4) worker4 ;;
    full_seed42_worker5) worker5 ;;
    full_seed42_worker6) worker6 ;;
    full_seed42_worker7) worker7 ;;
    *)
      worker4 & p1=$!; worker5 & p2=$!; worker6 & p3=$!; worker7 & p4=$!
      for pid in "$p1" "$p2" "$p3" "$p4"; do wait "$pid" || status=1; done
      ;;
  esac
fi
exit "$status"
