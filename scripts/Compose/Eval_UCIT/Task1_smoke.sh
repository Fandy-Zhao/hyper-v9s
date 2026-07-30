#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/ckpt/zhaozhuofan/models/llava-v1.5-7b}"
VISION_TOWER="${VISION_TOWER:-/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336}"
QUESTION_FILE="${QUESTION_FILE:-/data/dataset/zhaozhuofan/UCIT/instructions/ImageNet-R/test_3000.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/data/dataset/zhaozhuofan/UCIT/datasets}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a Compose or PEFT checkpoint}"
ADAPTER_KIND="${ADAPTER_KIND:-compose}"
RESULT_DIR="${RESULT_DIR:-/data/ckpt/zhaozhuofan/compose/eval/smoke/task1_${ADAPTER_KIND}_0730}"
GPU_ID="${GPU_ID:-0}"

if [[ -d "$RESULT_DIR" ]] && [[ -n "$(find "$RESULT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing non-empty result directory: $RESULT_DIR" >&2
  exit 1
fi
mkdir -p "$RESULT_DIR"
python -c 'import json,sys; data=json.load(open(sys.argv[1])); json.dump(data[:4], open(sys.argv[2], "w"))' "$QUESTION_FILE" "$RESULT_DIR/annotations.json"
CUDA_VISIBLE_DEVICES="$GPU_ID" python -m compose.eval.eval_task \
  --adapter-kind "$ADAPTER_KIND" \
  --model-path "$MODEL_PATH" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --projector-path "$MODEL_PATH/mm_projector.bin" \
  --vision-tower "$VISION_TOWER" \
  --question-file "$RESULT_DIR/annotations.json" \
  --image-folder "$IMAGE_FOLDER" \
  --answers-file "$RESULT_DIR/predictions.jsonl" \
  --run-summary-file "$RESULT_DIR/run.json" \
  --expert-id 0 \
  --gate 1 \
  --normalization none \
  --max-samples 4 \
  --device cuda:0
python -m compose.eval.metrics \
  --annotation-file "$RESULT_DIR/annotations.json" \
  --predictions-file "$RESULT_DIR/predictions.jsonl" \
  --output-file "$RESULT_DIR/metrics.json"
