#!/bin/bash
# Stage 01 evaluation: generate answers for one dataset with one model
# checkpoint, then score with the official evaluator.
#
# Usage: run_eval.sh <DATASET> <STAGE> <MODELPATH> <RESULT_ROOT> [GPU_LIST]
#   DATASET     one of: ImageNet-R ArxivQA VizWiz IconQA CLEVR Flickr30k
#   STAGE       e.g. hyper-task1, mini2-task1 ...
#   MODELPATH   checkpoint dir (the TaskN output dir)
#   RESULT_ROOT output root, mirrors runs/results/UCIT/... layout
#   GPU_LIST    default 0,1,2,3 (chunks = #gpus)
set -euo pipefail

# pycocoevalcap METEOR/SPICE require java; hyper env provides it
export PATH="/home/zhaozhuofan/miniconda3/envs/hyper/bin:$PATH"

DATASET=$1
STAGE=$2
MODELPATH=$3
RESULT_ROOT=$4
GPU_LIST="${5:-0,1,2,3}"

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
IFS=',' read -ra GPULIST <<< "$GPU_LIST"
CHUNKS=${#GPULIST[@]}

EVAL_ROUTING_ARG=()
if [ -n "${EVAL_MODALITY_ROUTING_MODE:-}" ]; then
    EVAL_ROUTING_ARG=(--eval-modality-routing-mode "$EVAL_MODALITY_ROUTING_MODE")
fi

RESULT_DIR="$RESULT_ROOT/$DATASET"
mkdir -p "$RESULT_DIR/$STAGE"

case "$DATASET" in
    ImageNet-R) QUESTION_FILE=/data/dataset/zhaozhuofan/UCIT/instructions/ImageNet-R/test_3000.json; SCORER=eval_deepseek_r1 ;;
    ArxivQA)    QUESTION_FILE=/data/dataset/zhaozhuofan/UCIT/instructions/ArxivQA/test_3000.json;     SCORER=eval_deepseek_r1 ;;
    IconQA)     QUESTION_FILE=/data/dataset/zhaozhuofan/UCIT/instructions/IconQA/test_3000.json;      SCORER=eval_deepseek_r1 ;;
    CLEVR)      QUESTION_FILE=/data/dataset/zhaozhuofan/UCIT/instructions/CLEVR/test_3000.json;       SCORER=eval_deepseek_r1 ;;
    VizWiz)     QUESTION_FILE=/data/dataset/zhaozhuofan/UCIT/instructions/VizWiz/test_3000.json;      SCORER=eval_caption ;;
    Flickr30k)  QUESTION_FILE=/data/dataset/zhaozhuofan/UCIT/instructions/Flickr30k/test_3000.json;   SCORER=eval_caption ;;
    *) echo "unknown dataset: $DATASET"; exit 2 ;;
esac

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} /home/zhaozhuofan/miniconda3/envs/hyper/bin/python -m llava.eval.model_answer \
        --model-path "$MODELPATH" \
        --model-base /data/ckpt/zhaozhuofan/models/llava-v1.5-7b \
        --question-file "$QUESTION_FILE" \
        --image-folder /data/dataset/zhaozhuofan/UCIT/datasets \
        --text-tower /data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336 \
        --answers-file "$RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        ${EVAL_ROUTING_ARG[@]} \
        --conv-mode vicuna_v1 &
done
wait

output_file="$RESULT_DIR/$STAGE/merge.jsonl"
> "$output_file"
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "$RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done

if [ "$SCORER" = "eval_caption" ]; then
    ANNOTATION_FILE="$(dirname "$QUESTION_FILE")/val_coco_type_3000.json"
    /home/zhaozhuofan/miniconda3/envs/hyper/bin/python -m llava.eval.eval_caption \
        --annotation-file "$ANNOTATION_FILE" \
        --result-file "$output_file" \
        --output-dir "$RESULT_DIR/$STAGE"
else
    /home/zhaozhuofan/miniconda3/envs/hyper/bin/python -m llava.eval.eval_deepseek_r1 \
        --annotation-file "$QUESTION_FILE" \
        --result-file "$output_file" \
        --output-dir "$RESULT_DIR/$STAGE"
fi

echo "DONE $DATASET $STAGE -> $RESULT_DIR/$STAGE"
