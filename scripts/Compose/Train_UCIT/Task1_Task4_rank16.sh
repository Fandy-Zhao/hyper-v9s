#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/ckpt/zhaozhuofan/models/llava-v1.5-7b}"
VISION_TOWER="${VISION_TOWER:-/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336}"
DATA_PATH="${DATA_PATH:-/data/ckpt/zhaozhuofan/compose/inputs/task1_task4_train_53857.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/data/dataset/zhaozhuofan/UCIT/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/ckpt/zhaozhuofan/compose/UCIT/Task1_Task4_rank16_seed42}"
MAX_STEPS="${MAX_STEPS:-}"

for required in "$MODEL_PATH" "$VISION_TOWER" "$DATA_PATH" "$IMAGE_FOLDER" "$MODEL_PATH/mm_projector.bin"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing non-empty output directory: $OUTPUT_DIR" >&2
  exit 1
fi
record_count="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$DATA_PATH")"
if [[ "$record_count" != "53857" ]]; then
  echo "Unexpected mixed-task record count: $record_count" >&2
  exit 1
fi
extra_steps=()
if [[ -n "$MAX_STEPS" ]]; then
  extra_steps=(--max_steps "$MAX_STEPS" --save_strategy no)
else
  extra_steps=(--num_train_epochs 1 --save_strategy epoch)
fi
printf 'Compose mixed rank16: output=%s rank=16 alpha=32 scale=2 seed=42 records=%s global_batch=64 max_steps=%s\n' \
  "$OUTPUT_DIR" "$record_count" "${MAX_STEPS:-full}"

deepspeed --include localhost:0,1,2,3 --master_port "${MASTER_PORT:-29618}" \
  --module compose.train.train_compose \
  --deepspeed ./scripts/zero2.json \
  --model_name_or_path "$MODEL_PATH" \
  --pretrain_mm_mlp_adapter "$MODEL_PATH/mm_projector.bin" \
  --version v1 \
  --data_path "$DATA_PATH" \
  --image_folder "$IMAGE_FOLDER" \
  --vision_tower "$VISION_TOWER" \
  --compose_rank 16 \
  --compose_alpha 32 \
  --compose_dropout 0 \
  --compose_expert_ids 0 \
  --compose_expert_name task1-task4-mixed-rank16 \
  --compose_origin_task_id UCIT/ImageNet-R+IconQA \
  --compose_expert_tags functional-proxy,capacity-control \
  --compose_gates 1 \
  --compose_gate_normalization none \
  --expected_adapter_parameters 39976960 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --evaluation_strategy no \
  --learning_rate 2e-4 \
  --weight_decay 0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --report_to none \
  --seed 42 \
  "${extra_steps[@]}"
