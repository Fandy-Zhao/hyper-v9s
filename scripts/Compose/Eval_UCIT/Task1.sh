#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/ckpt/zhaozhuofan/models/llava-v1.5-7b}"
VISION_TOWER="${VISION_TOWER:-/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336}"
QUESTION_FILE="${QUESTION_FILE:-/data/dataset/zhaozhuofan/UCIT/instructions/ImageNet-R/test_3000.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/data/dataset/zhaozhuofan/UCIT/datasets}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a Compose or PEFT checkpoint}"
ADAPTER_KIND="${ADAPTER_KIND:-compose}"
RESULT_DIR="${RESULT_DIR:-/data/ckpt/zhaozhuofan/compose/eval/task1_${ADAPTER_KIND}_rank8_seed42}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

for required in "$MODEL_PATH" "$VISION_TOWER" "$QUESTION_FILE" "$IMAGE_FOLDER" "$MODEL_PATH/mm_projector.bin" "$CHECKPOINT_DIR"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done
if [[ -d "$RESULT_DIR" ]] && [[ -n "$(find "$RESULT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing non-empty result directory: $RESULT_DIR" >&2
  exit 1
fi
IFS=',' read -r -a gpus <<< "$GPU_LIST"
chunks="${#gpus[@]}"
mkdir -p "$RESULT_DIR"
printf 'Compose Task1 eval: kind=%s checkpoint=%s result=%s chunks=%s metric=case-insensitive-exact-match\n' "$ADAPTER_KIND" "$CHECKPOINT_DIR" "$RESULT_DIR" "$chunks"

pids=()
for ((index=0; index<chunks; index++)); do
  CUDA_VISIBLE_DEVICES="${gpus[$index]}" python -m compose.eval.eval_task \
    --adapter-kind "$ADAPTER_KIND" \
    --model-path "$MODEL_PATH" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --projector-path "$MODEL_PATH/mm_projector.bin" \
    --vision-tower "$VISION_TOWER" \
    --question-file "$QUESTION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --answers-file "$RESULT_DIR/${chunks}_${index}.jsonl" \
    --run-summary-file "$RESULT_DIR/${chunks}_${index}_run.json" \
    --expert-id 0 \
    --gate 1 \
    --normalization none \
    --num-chunks "$chunks" \
    --chunk-idx "$index" \
    --device cuda:0 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
: > "$RESULT_DIR/predictions.jsonl"
for ((index=0; index<chunks; index++)); do
  cat "$RESULT_DIR/${chunks}_${index}.jsonl" >> "$RESULT_DIR/predictions.jsonl"
done
python -m compose.eval.metrics \
  --annotation-file "$QUESTION_FILE" \
  --predictions-file "$RESULT_DIR/predictions.jsonl" \
  --output-file "$RESULT_DIR/metrics.json"
