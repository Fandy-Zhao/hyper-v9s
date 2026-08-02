#!/bin/bash
# Stage 01 mini2 pipeline: train ImageNet-R -> evaluate -> train ArxivQA ->
# evaluate both. Config/GPU/port parameterized; same mini2 data for both
# baseline configs. Effective global batch is asserted by make_launch.py.
#
# Usage: run_mini2.sh <original_batch|gb24_matched> <GPU_LIST> <PORT_BASE> <OUT_TAG> <EVAL_GPU>
set -euo pipefail

CONFIG=$1
GPU_LIST=$2
PORT_BASE=$3
TAG=$4
EVAL_GPU=${5:-3}

BASE=/home/zhaozhuofan/Hyper-LlaVA/experiments/runs/v6_ucit_staged/stage01_baseline
PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
CK=/data/ckpt/zhaozhuofan/v6_ucit_staged/stage01_baseline/checkpoints
EVAL_ROOT="$BASE/evaluations/mini2_$TAG"
LOG_DIR="$BASE/logs"
mkdir -p "$EVAL_ROOT"

# Task 1: ImageNet-R
T1_OUT="$CK/mini2_${TAG}_task1_llava_lora_ours"
$PY "$BASE/scripts/make_launch.py" --task 1 --config "$CONFIG" --gpus "$GPU_LIST" \
    --tag "mini2_${TAG}_task1" --data-path "$BASE/data_indices/ImageNet-R/train.json" \
    --output-dir "$T1_OUT" --port "$((PORT_BASE + 1))" >/dev/null
bash "$BASE/generated/run_mini2_${TAG}_task1.sh" > "$LOG_DIR/mini2_${TAG}_task1.log" 2>&1
echo "task1 exit: $?"

# Task 1 eval: ImageNet-R (mini2 test)
bash "$BASE/scripts/run_eval_mini.sh" ImageNet-R "task1" "$T1_OUT" "$EVAL_ROOT" "$EVAL_GPU"

# Task 2: ArxivQA (prev task = task1 output)
T2_OUT="$CK/mini2_${TAG}_task2_llava_lora_ours"
$PY "$BASE/scripts/make_launch.py" --task 2 --config "$CONFIG" --gpus "$GPU_LIST" \
    --tag "mini2_${TAG}_task2" --data-path "$BASE/data_indices/ArxivQA/train.json" \
    --output-dir "$T2_OUT" --port "$((PORT_BASE + 2))" --prev-task-path "$T1_OUT" >/dev/null
bash "$BASE/generated/run_mini2_${TAG}_task2.sh" > "$LOG_DIR/mini2_${TAG}_task2.log" 2>&1
echo "task2 exit: $?"

# Task 2 eval: both seen tasks
bash "$BASE/scripts/run_eval_mini.sh" ImageNet-R "task2" "$T2_OUT" "$EVAL_ROOT" "$EVAL_GPU"
bash "$BASE/scripts/run_eval_mini.sh" ArxivQA "task2" "$T2_OUT" "$EVAL_ROOT" "$EVAL_GPU"

echo "MINI2_${TAG}_DONE"
