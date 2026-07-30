#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/ckpt/zhaozhuofan/models/llava-v1.5-7b}"
VISION_TOWER="${VISION_TOWER:-/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/ckpt/zhaozhuofan/compose/UCIT/Task1_Task4_experts01_rank8_seed42}"
QUESTION_FILE="${QUESTION_FILE:-/data/ckpt/zhaozhuofan/compose/inputs/task1_task4_test_6000.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/data/dataset/zhaozhuofan/UCIT/datasets}"
RESULT_DIR="${RESULT_DIR:-/data/ckpt/zhaozhuofan/compose/eval/task1_task4_selections_rank8_seed42}"

for required in "$MODEL_PATH" "$VISION_TOWER" "$MODEL_PATH/mm_projector.bin" \
  "$CHECKPOINT_DIR/compose_experts.json" "$CHECKPOINT_DIR/compose_experts.bin" \
  "$QUESTION_FILE" "$IMAGE_FOLDER"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done
if [[ -d "$RESULT_DIR" ]] && [[ -n "$(find "$RESULT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing non-empty result directory: $RESULT_DIR" >&2
  exit 1
fi
mkdir -p "$RESULT_DIR"
names=(base single0 single1 pair_l2)
ids=("" "0" "1" "0,1")
gates=("" "1" "1" "1,1")
normalizations=(none none none l2)
pids=()
for index in 0 1 2 3; do
  name="${names[$index]}"
  mkdir -p "$RESULT_DIR/$name"
  CUDA_VISIBLE_DEVICES="$index" python -m compose.eval.eval_task \
    --adapter-kind compose --model-path "$MODEL_PATH" --checkpoint-dir "$CHECKPOINT_DIR" \
    --projector-path "$MODEL_PATH/mm_projector.bin" --vision-tower "$VISION_TOWER" \
    --question-file "$QUESTION_FILE" --image-folder "$IMAGE_FOLDER" \
    --answers-file "$RESULT_DIR/$name/predictions.jsonl" \
    --run-summary-file "$RESULT_DIR/$name/run.json" \
    --expert-ids "${ids[$index]}" --gates "${gates[$index]}" \
    --normalization "${normalizations[$index]}" --device cuda:0 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
for name in "${names[@]}"; do
  python -m compose.eval.metrics --annotation-file "$QUESTION_FILE" \
    --predictions-file "$RESULT_DIR/$name/predictions.jsonl" \
    --output-file "$RESULT_DIR/$name/metrics.json"
done
