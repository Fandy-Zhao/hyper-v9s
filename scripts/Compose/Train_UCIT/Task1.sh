#!/usr/bin/env bash
set -euo pipefail

PROMPT_VERSION="${PROMPT_VERSION:-v1}"
MODEL_PATH="${MODEL_PATH:-/data/ckpt/zhaozhuofan/models/llava-v1.5-7b}"
VISION_TOWER="${VISION_TOWER:-/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336}"
DATA_PATH="${DATA_PATH:-/data/dataset/zhaozhuofan/UCIT/instructions/ImageNet-R/train.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/data/dataset/zhaozhuofan/UCIT/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/ckpt/zhaozhuofan/compose/UCIT/Task1}"

deepspeed --include localhost:0,1,2,3 --master_port "${MASTER_PORT:-29611}" \
  --module compose.train.train_compose \
  --deepspeed ./scripts/zero2.json \
  --model_name_or_path "$MODEL_PATH" \
  --pretrain_mm_mlp_adapter "$MODEL_PATH/mm_projector.bin" \
  --version "$PROMPT_VERSION" \
  --data_path "$DATA_PATH" \
  --image_folder "$IMAGE_FOLDER" \
  --vision_tower "$VISION_TOWER" \
  --compose_rank 8 \
  --compose_alpha 16 \
  --compose_expert_ids 0 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --evaluation_strategy no \
  --save_strategy epoch \
  --learning_rate 2e-4 \
  --weight_decay 0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --report_to none
