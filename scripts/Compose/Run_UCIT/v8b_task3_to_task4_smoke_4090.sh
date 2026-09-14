#!/usr/bin/env bash
set -euo pipefail

REPO=/home/xukunlun/CODE/CIT/zzf
RUN=/data/ckpt/xukunlun/CIT/zzf_runs/v8b_task3_to_task4_smoke_20260914
PYTHON=/data/dataset/xukunlun/anaconda3/envs/hyper/bin/python
TEACHER_PID=${TEACHER_PID:-766004}

while kill -0 "${TEACHER_PID}" 2>/dev/null; do
  sleep 15
done

if [[ ! -f "${RUN}/teacher/task4/COMPLETE.json" ]]; then
  printf '%s\n' "teacher phase failed; V8-B training was not started" > "${RUN}/V8B_BLOCKED"
  exit 1
fi

cd "${REPO}"
export CUDA_VISIBLE_DEVICES=1
export COMPOSE_SELECTION_PLAN=1

"${PYTHON}" -m compose.experiments.v8b_task_run \
  --task 4 \
  --previous-checkpoint /data/ckpt/xukunlun/CIT/zzf_runs/v8_exact_full6_20260912/task3/committed \
  --teacher-result "${RUN}/teacher/task4/teacher_result.json" \
  --train-json /data/ckpt/xukunlun/CIT/zzf_runs/v8_exact_full6_20260912/task4/data/train_full.json \
  --val-json /data/ckpt/xukunlun/CIT/zzf_runs/v8_exact_full6_20260912/task4/data/val_full.json \
  --query-cache-root /home/xukunlun/CODE/CIT/zzf/assets/v8/v7_fixed_query_cache_gpu01_20260903 \
  --model-path /home/xukunlun/CODE/CIT/zzf/assets/models/llava-v1.5-7b \
  --vision-tower /home/xukunlun/CODE/CIT/zzf/assets/models/clip-vit-large-patch14-336 \
  --projector-path /home/xukunlun/CODE/CIT/zzf/assets/models/llava-v1.5-7b/mm_projector.bin \
  --image-folder /home/xukunlun/CODE/CIT/zzf/assets/UCIT/datasets \
  --output-dir "${RUN}/v8b" \
  --device cuda:0 \
  --epochs 1 \
  --batch-size 1 \
  --val-limit 8 \
  --max-new-tokens 128
