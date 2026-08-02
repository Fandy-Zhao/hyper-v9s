#!/bin/bash
set -uo pipefail

cd /home/zhaozhuofan/Hyper-LlaVA
ROOT=experiments/runs/v6_ucit_staged/stage03_composition
CK=/data/ckpt/zhaozhuofan/v6_ucit_staged/stage03_composition/checkpoints
EVAL_ROOT=$ROOT/mini2/evaluations
EXIT_DIR=$ROOT/mini2/exits
mkdir -p "$EXIT_DIR" "$EVAL_ROOT"

snapshot() {
  local name=$1
  {
    date -Is
    nvidia-smi
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
  } >"$ROOT/logs/gpu_${name}.txt" 2>&1
}

snapshot before_mini2_task1
bash "$ROOT/configs/run_mini2_gb24_task1.sh" >"$ROOT/logs/mini2_task1.log" 2>&1
STATUS=$?
printf '%s\n' "$STATUS" >"$EXIT_DIR/task1_train.exit"
snapshot after_mini2_task1
if [ "$STATUS" -ne 0 ]; then exit "$STATUS"; fi

snapshot before_mini2_task1_eval
export PYTHONPATH=/home/zhaozhuofan/Hyper-LlaVA/Hyper:/home/zhaozhuofan/Hyper-LlaVA
bash experiments/runs/v6_ucit_staged/stage01_baseline/scripts/run_eval_mini.sh \
  ImageNet-R task1 "$CK/mini2_gb24_task1_llava_lora_ours" "$EVAL_ROOT" 4 \
  >"$ROOT/logs/mini2_task1_eval_imagenet.log" 2>&1
STATUS=$?
printf '%s\n' "$STATUS" >"$EXIT_DIR/task1_eval_imagenet.exit"
snapshot after_mini2_task1_eval
if [ "$STATUS" -ne 0 ]; then exit "$STATUS"; fi

snapshot before_mini2_task2
bash "$ROOT/configs/run_mini2_gb24_task2.sh" >"$ROOT/logs/mini2_task2.log" 2>&1
STATUS=$?
printf '%s\n' "$STATUS" >"$EXIT_DIR/task2_train.exit"
snapshot after_mini2_task2
if [ "$STATUS" -ne 0 ]; then exit "$STATUS"; fi

snapshot before_mini2_task2_eval_imagenet
bash experiments/runs/v6_ucit_staged/stage01_baseline/scripts/run_eval_mini.sh \
  ImageNet-R task2 "$CK/mini2_gb24_task2_llava_lora_ours" "$EVAL_ROOT" 4 \
  >"$ROOT/logs/mini2_task2_eval_imagenet.log" 2>&1
STATUS=$?
printf '%s\n' "$STATUS" >"$EXIT_DIR/task2_eval_imagenet.exit"
snapshot after_mini2_task2_eval_imagenet
if [ "$STATUS" -ne 0 ]; then exit "$STATUS"; fi

snapshot before_mini2_task2_eval_arxivqa
bash experiments/runs/v6_ucit_staged/stage01_baseline/scripts/run_eval_mini.sh \
  ArxivQA task2 "$CK/mini2_gb24_task2_llava_lora_ours" "$EVAL_ROOT" 4 \
  >"$ROOT/logs/mini2_task2_eval_arxivqa.log" 2>&1
STATUS=$?
printf '%s\n' "$STATUS" >"$EXIT_DIR/task2_eval_arxivqa.exit"
snapshot after_mini2_task2_eval_arxivqa
exit "$STATUS"
