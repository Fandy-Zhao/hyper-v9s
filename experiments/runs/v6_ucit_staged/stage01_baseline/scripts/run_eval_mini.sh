#!/bin/bash
# Stage 01 mini2 evaluation: one dataset against one model checkpoint using
# the mini2 test indices (128 samples) and the official evaluator.
#
# Usage: run_eval_mini.sh <DATASET> <STAGE> <MODELPATH> <RESULT_ROOT> [GPU]
#   DATASET  ImageNet-R | ArxivQA (mini2 tasks)
set -euo pipefail
DATASET=$1
STAGE=$2
MODELPATH=$3
RESULT_ROOT=$4
GPU="${5:-3}"
BASE=/home/zhaozhuofan/Hyper-LlaVA/experiments/runs/v6_ucit_staged/stage01_baseline

RESULT_DIR="$RESULT_ROOT/$DATASET"
mkdir -p "$RESULT_DIR/$STAGE"

export CUDA_VISIBLE_DEVICES="$GPU"
/home/zhaozhuofan/miniconda3/envs/hyper/bin/python -m llava.eval.model_answer \
    --model-path "$MODELPATH" \
    --model-base /data/ckpt/zhaozhuofan/models/llava-v1.5-7b \
    --question-file "$BASE/data_indices/$DATASET/test.json" \
    --image-folder /data/dataset/zhaozhuofan/UCIT/datasets \
    --text-tower /data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336 \
    --answers-file "$RESULT_DIR/$STAGE/merge.jsonl" \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode vicuna_v1

/home/zhaozhuofan/miniconda3/envs/hyper/bin/python -m llava.eval.eval_deepseek_r1 \
    --annotation-file "$BASE/data_indices/$DATASET/test.json" \
    --result-file "$RESULT_DIR/$STAGE/merge.jsonl" \
    --output-dir "$RESULT_DIR/$STAGE"

echo "DONE $DATASET $STAGE"
