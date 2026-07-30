#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/ckpt/zhaozhuofan/models/llava-v1.5-7b}"
VISION_TOWER="${VISION_TOWER:-/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336}"
DATA_PATH="${DATA_PATH:-/data/dataset/zhaozhuofan/UCIT/instructions/ImageNet-R/train.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/data/dataset/zhaozhuofan/UCIT/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/ckpt/zhaozhuofan/compose/baselines/peft_task1_rank8_seed42}"
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
if [[ "$record_count" != "23998" ]]; then
  echo "Unexpected Task1 record count: $record_count" >&2
  exit 1
fi
extra_steps=()
if [[ -n "$MAX_STEPS" ]]; then
  extra_steps=(--max_steps "$MAX_STEPS" --save_strategy no)
else
  extra_steps=(--num_train_epochs 1 --save_strategy epoch)
fi
printf 'PEFT Task1 config: output=%s rank=8 alpha=16 dropout=0 seed=42 records=%s global_batch=64 max_steps=%s\n' "$OUTPUT_DIR" "$record_count" "${MAX_STEPS:-full}"

deepspeed --include localhost:0,1,2,3 --master_port "${MASTER_PORT:-29613}" \
  --module compose.train.train_peft \
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
  --expected_adapter_parameters 19988480 \
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
