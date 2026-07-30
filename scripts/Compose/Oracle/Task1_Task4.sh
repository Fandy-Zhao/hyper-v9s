#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/ckpt/zhaozhuofan/models/llava-v1.5-7b}"
VISION_TOWER="${VISION_TOWER:-/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/ckpt/zhaozhuofan/compose/UCIT/Task1_Task4_experts01_rank8_seed42}"
QUESTION_FILE="${QUESTION_FILE:-/data/ckpt/zhaozhuofan/compose/inputs/task1_task4_test_6000.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/data/dataset/zhaozhuofan/UCIT/datasets}"
RESULT_DIR="${RESULT_DIR:-/data/ckpt/zhaozhuofan/compose/oracle/task1_task4_experts01_rank8_seed42}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"

for required in "$MODEL_PATH" "$VISION_TOWER" "$MODEL_PATH/mm_projector.bin" \
  "$CHECKPOINT_DIR/compose_experts.json" "$CHECKPOINT_DIR/compose_experts.bin" \
  "$QUESTION_FILE" "$IMAGE_FOLDER"; do
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
extra=()
if [[ -n "$MAX_SAMPLES" ]]; then
  extra=(--max-samples "$MAX_SAMPLES")
fi
printf 'Compose Oracle: checkpoint=%s samples=%s chunks=%s batch=%s max_samples=%s\n' \
  "$CHECKPOINT_DIR" "$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$QUESTION_FILE")" \
  "$chunks" "$BATCH_SIZE" "${MAX_SAMPLES:-full}"

pids=()
for ((index=0; index<chunks; index++)); do
  CUDA_VISIBLE_DEVICES="${gpus[$index]}" python -m compose.oracle.evaluator \
    --model-path "$MODEL_PATH" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --projector-path "$MODEL_PATH/mm_projector.bin" \
    --vision-tower "$VISION_TOWER" \
    --question-file "$QUESTION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --output-file "$RESULT_DIR/${chunks}_${index}.jsonl" \
    --summary-file "$RESULT_DIR/${chunks}_${index}_summary.json" \
    --task-id Task1+Task4 \
    --batch-size "$BATCH_SIZE" \
    --num-chunks "$chunks" \
    --chunk-idx "$index" \
    --device cuda:0 \
    "${extra[@]}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
: > "$RESULT_DIR/oracle_cache.jsonl"
for ((index=0; index<chunks; index++)); do
  cat "$RESULT_DIR/${chunks}_${index}.jsonl" >> "$RESULT_DIR/oracle_cache.jsonl"
done
python -m compose.oracle.aggregate \
  --input-file "$RESULT_DIR/oracle_cache.jsonl" \
  --output-file "$RESULT_DIR/metrics.json"
