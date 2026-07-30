#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/ckpt/zhaozhuofan/models/llava-v1.5-7b}"
VISION_TOWER="${VISION_TOWER:-/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336}"
DATA_PATH="${DATA_PATH:-/data/dataset/zhaozhuofan/UCIT/instructions/ImageNet-R/train.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/data/dataset/zhaozhuofan/UCIT/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/ckpt/zhaozhuofan/compose/smoke/task1_two_step_0730}"

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
printf 'Compose Task1 smoke: output=%s rank=8 alpha=16 dropout=0 expert=0 gate=1 normalization=none seed=42 max_steps=2\n' "$OUTPUT_DIR"

deepspeed --include "localhost:${GPU_ID:-0}" --master_port "${MASTER_PORT:-29612}" \
  --module compose.train.train_compose \
  --deepspeed ./scripts/zero2.json \
  --model_name_or_path "$MODEL_PATH" \
  --pretrain_mm_mlp_adapter "$MODEL_PATH/mm_projector.bin" \
  --version v1 \
  --data_path "$DATA_PATH" \
  --image_folder "$IMAGE_FOLDER" \
  --vision_tower "$VISION_TOWER" \
  --compose_rank 8 \
  --compose_alpha 16 \
  --compose_dropout 0 \
  --compose_expert_ids 0 \
  --compose_gates 1 \
  --compose_gate_normalization none \
  --expected_adapter_parameters 19988480 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --bf16 True \
  --output_dir "$OUTPUT_DIR" \
  --max_steps 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy no \
  --save_strategy no \
  --learning_rate 2e-4 \
  --logging_steps 1 \
  --model_max_length 1024 \
  --gradient_checkpointing True \
  --dataloader_num_workers 0 \
  --report_to none \
  --seed 42
