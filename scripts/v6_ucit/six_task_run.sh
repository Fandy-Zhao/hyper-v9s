#!/bin/bash
# V6 UCIT 六任务正式运行编排（seed 参数化）
# 依据：configs/v6_ucit_formal_locked.yaml（formal locked config v1）
# 使用方式：
#   bash scripts/v6_ucit/six_task_run.sh <seed> [gpus]
#   seed 默认 42；gpus 默认 4,5,6,7
set -Eeuo pipefail

SEED="${1:-42}"
GPUS="${2:-4,5,6,7}"
PYTHON=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
REPO=/home/zhaozhuofan/Hyper-LlaVA
CONFIG=$REPO/configs/v6_ucit_formal_locked.yaml
RUN_ROOT=$REPO/experiments/runs/v6_ucit_engineering/formal/seed_$SEED
MASTER_PORT=29681
LOGS=$RUN_ROOT

echo "== seed $SEED 六任务正式运行 (GPU $GPUS) =="
mkdir -p "$RUN_ROOT"

run_stage() {
  local stage_name="$1"; shift
  local stage_done="$RUN_ROOT/$stage_name.done"
  if [ -f "$stage_done" ]; then
    echo "[skip] $stage_name already done"
    return 0
  fi
  echo "== $stage_name =="
  "$@"
  touch "$stage_done"
}

# ---- 任务 1 (ImageNet-R): 已验收冷启动 runner -------------------------------
run_stage task0_done env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PYTHON -m compose.experiments.v6_task1_dry_run \
  --output-root "$RUN_ROOT/task0" \
  --gpus "$GPUS" --master-port $((MASTER_PORT + 0)) \
  --config "$CONFIG"

# ---- 任务 2 (ArxivQA): 已验收 task2 runner ---------------------------------
run_stage task1_done env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PYTHON -m compose.experiments.v6_task2_dry_run \
  --output-root "$RUN_ROOT/task1" \
  --task1-root "$RUN_ROOT/task0" \
  --gpus "$GPUS" --master-port $((MASTER_PORT + 1)) \
  --config "$CONFIG"

# ---- 任务 3..6 (VizWiz/IconQA/CLEVR/Flickr30k): 通用 runner ----------------
run_stage task2_done env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PYTHON -m compose.experiments.v6_task_run \
  --output-root "$RUN_ROOT/task2" \
  --prev-task-root "$RUN_ROOT/task1" \
  --task-id 2 --gpus "$GPUS" --master-port $((MASTER_PORT + 2)) \
  --config "$CONFIG"

run_stage task3_done env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PYTHON -m compose.experiments.v6_task_run \
  --output-root "$RUN_ROOT/task3" \
  --prev-task-root "$RUN_ROOT/task2" \
  --task-id 3 --gpus "$GPUS" --master-port $((MASTER_PORT + 3)) \
  --config "$CONFIG"

run_stage task4_done env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PYTHON -m compose.experiments.v6_task_run \
  --output-root "$RUN_ROOT/task4" \
  --prev-task-root "$RUN_ROOT/task3" \
  --task-id 4 --gpus "$GPUS" --master-port $((MASTER_PORT + 4)) \
  --config "$CONFIG"

run_stage task5_done env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PYTHON -m compose.experiments.v6_task_run \
  --output-root "$RUN_ROOT/task5" \
  --prev-task-root "$RUN_ROOT/task4" \
  --task-id 5 --gpus "$GPUS" --master-port $((MASTER_PORT + 5)) \
  --config "$CONFIG"

echo "== seed $SEED 六任务全部 COMPLETED =="
