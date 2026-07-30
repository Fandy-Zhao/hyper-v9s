#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/ckpt/zhaozhuofan/models/llava-v1.5-7b}"
VISION_TOWER="${VISION_TOWER:-/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR}"
QUESTION_FILE="${QUESTION_FILE:-/data/ckpt/zhaozhuofan/compose/inputs/task1_task4_test_6000.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/data/dataset/zhaozhuofan/UCIT/datasets}"
RESULT_DIR="${RESULT_DIR:?Set RESULT_DIR}"
SELECTION_NAME="${SELECTION_NAME:?Set SELECTION_NAME}"
EXPERT_IDS="${EXPERT_IDS:-}"
GATES="${GATES:-}"
NORMALIZATION="${NORMALIZATION:-none}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

for required in "$MODEL_PATH" "$VISION_TOWER" "$MODEL_PATH/mm_projector.bin" \
  "$CHECKPOINT_DIR/compose_experts.json" "$CHECKPOINT_DIR/compose_experts.bin" \
  "$QUESTION_FILE" "$IMAGE_FOLDER"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done
if [[ -d "$RESULT_DIR" ]] && [[ -n "$(find "$RESULT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing non-empty result directory: $RESULT_DIR" >&2
  exit 1
fi
IFS=',' read -r -a gpus <<< "$GPU_LIST"
chunks="${#gpus[@]}"
mkdir -p "$RESULT_DIR"
extra=()
[[ -z "$MAX_SAMPLES" ]] || extra=(--max-samples "$MAX_SAMPLES")
pids=()
for ((index=0; index<chunks; index++)); do
  CUDA_VISIBLE_DEVICES="${gpus[$index]}" python -m compose.oracle.score_selection \
    --model-path "$MODEL_PATH" --checkpoint-dir "$CHECKPOINT_DIR" \
    --projector-path "$MODEL_PATH/mm_projector.bin" --vision-tower "$VISION_TOWER" \
    --question-file "$QUESTION_FILE" --image-folder "$IMAGE_FOLDER" \
    --output-file "$RESULT_DIR/${chunks}_${index}.jsonl" \
    --summary-file "$RESULT_DIR/${chunks}_${index}_summary.json" \
    --selection-name "$SELECTION_NAME" --expert-ids "$EXPERT_IDS" --gates "$GATES" \
    --normalization "$NORMALIZATION" --batch-size "$BATCH_SIZE" \
    --num-chunks "$chunks" --chunk-idx "$index" --device cuda:0 "${extra[@]}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
: > "$RESULT_DIR/scores.jsonl"
for ((index=0; index<chunks; index++)); do cat "$RESULT_DIR/${chunks}_${index}.jsonl" >> "$RESULT_DIR/scores.jsonl"; done
python -m compose.oracle.aggregate_scores \
  --input-file "$RESULT_DIR/scores.jsonl" --output-file "$RESULT_DIR/metrics.json"
