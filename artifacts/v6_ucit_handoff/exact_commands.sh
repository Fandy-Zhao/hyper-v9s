#!/bin/bash
# V6 UCIT 六任务正式运行命令（下一批使用；本批次未运行六任务）
# 依据：configs/v6_ucit_formal_locked.yaml（= artifacts/v6_ucit_handoff/locked_config.yaml）
set -Eeuo pipefail

PYTHON=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
REPO=/home/zhaozhuofan/Hyper-LlaVA
RUN_ROOT=$REPO/experiments/runs/v6_ucit_engineering
MASTER_PORT=29681
# GPU 4-7 空闲；若被占用用 0,1,4,5（nvidia-smi 确认）
GPUS=4,5,6,7

echo "== 1. 环境激活 =="
# conda activate hyper   (或直接使用 $PYTHON)

echo "== 2. 状态检查（运行前） =="
git -C $REPO status --short | head
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
$PYTHON -m pytest $REPO/tests/compose -q 2>&1 | tail -1

echo "== 3. seed-42 六任务启动（按正式序列逐任务） =="
# Task 1: ImageNet-R 冷启动（正式配置：全量 23998 训练样本）
# 注意：formal_locked 中 training 规模为 full；dry-run 用 2000 样本。
# 以下为 dry-run 规模命令模板；正式规模仅改 cold_start_train_samples。
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PYTHON -m compose.experiments.v6_task1_dry_run \
  --output-root $RUN_ROOT/formal/task0 \
  --gpus $GPUS --master-port $MASTER_PORT \
  --config $REPO/configs/v6_ucit_formal_locked.yaml \
  > $RUN_ROOT/formal/task0_runner.log 2>&1 &

# Task 2..6 在 Task1 COMPLETED 后串行启动（任务序列见 task_sequence.json）
# 每任务输出根：formal/task{N-1}，--task1-root 指向上一任务根。

echo "== 4. 状态检查（运行中） =="
tail -3 $RUN_ROOT/formal/task0_runner.log
ls $RUN_ROOT/formal/task0/stages/ 2>/dev/null

echo "== 5. 安全停止 =="
pkill -f v6_task1_dry_run || true   # 阶段化幂等：重启自动从断点继续

echo "== 6. 恢复 =="
# 幂等：直接重跑同一命令（stages/*.done 存在则跳过已完成阶段）

echo "== 7. snapshot 校验 =="
$PYTHON - <<'PYEOF'
from compose.experiments.v6_snapshot import V6Snapshot, analyze_resume
import sys
snap = V6Snapshot.load(sys.argv[1] if len(sys.argv) > 1 else "$RUN_ROOT/formal/task0/snapshots/task0")
print("snapshot OK:", snap.to_dict())
print("resume:", analyze_resume(snap.directory))
PYEOF

echo "== 8. 评估（原 Hyper eval 入口，全 3000 测试样本） =="
$PYTHON -m compose.eval.eval_task --adapter-kind compose \
  --model-path /data/ckpt/zhaozhuofan/models/llava-v1.5-7b \
  --checkpoint-dir $RUN_ROOT/formal/task0/committed/expert_0010 \
  --projector-path /data/ckpt/zhaozhuofan/models/llava-v1.5-7b/mm_projector.bin \
  --vision-tower /data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336 \
  --question-file /data/dataset/zhaozhuofan/UCIT/instructions/ImageNet-R/test_3000.json \
  --image-folder /data/dataset/zhaozhuofan/UCIT/datasets \
  --answers-file $RUN_ROOT/formal/task0/eval/full_predictions.jsonl \
  --run-summary-file $RUN_ROOT/formal/task0/eval/full_summary.json \
  --device cuda:4 --max-samples 3000 \
  --expert-ids 10 --gates 1.0

echo "== 9. 持续指标（六任务后） =="
$PYTHON -m scripts.Hyper.Eval_UCIT.summarize_continual_metrics \
  --result-root $RUN_ROOT/formal --num-tasks 6 \
  --output-file $RUN_ROOT/formal/continual_metrics.json
