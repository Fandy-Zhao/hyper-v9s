#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhaozhuofan/Hyper-LlaVA
STAGE="$ROOT/experiments/runs/v6_ucit_staged/stage07_residual_buffer"
DATA=/data/ckpt/zhaozhuofan/v6_ucit_staged
PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
QUERY_KEY="$DATA/stage05_query_keys/checkpoints/continual_anchor.pt"
ROUTER="$DATA/stage06_set_router/checkpoints/continual_anchor.pt"
HEAD="$DATA/stage07_residual_buffer/checkpoints/sufficiency.pt"
mkdir -p "$STAGE/logs" "$STAGE/metrics" "$STAGE/manifests" \
         "$DATA/stage07_residual_buffer/features" "$DATA/stage07_residual_buffer/checkpoints" \
         "$DATA/stage07_residual_buffer/buffers"
cd "$ROOT"
export PYTHONPATH="$ROOT"
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  nvidia-smi
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
} > "$STAGE/manifests/gpu_preflight.txt"

CUDA_VISIBLE_DEVICES=4 "$PY" -m compose.cli.export_router_features \
  --features "$DATA/stage05_query_keys/features/bundles/train.pt" \
  --query-key-checkpoint "$QUERY_KEY" --router-checkpoint "$ROUTER" \
  --output "$DATA/stage07_residual_buffer/features/train.pt" --device cuda:0 \
  > "$STAGE/logs/export_train.log" 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=5 "$PY" -m compose.cli.export_router_features \
  --features "$DATA/stage05_query_keys/features/bundles/validation.pt" \
  --query-key-checkpoint "$QUERY_KEY" --router-checkpoint "$ROUTER" \
  --output "$DATA/stage07_residual_buffer/features/validation.pt" --device cuda:0 \
  > "$STAGE/logs/export_validation.log" 2>&1 & p2=$!
status=0; wait "$p1" || status=1; wait "$p2" || status=1; [[ "$status" == 0 ]]

for diagnostic in smoke:0 mini2:1 mini3:2; do
  name=${diagnostic%%:*}; max_task=${diagnostic##*:}
  CUDA_VISIBLE_DEVICES=4 "$PY" -m compose.cli.eval_sufficiency \
    --train "$DATA/stage07_residual_buffer/features/train.pt" \
    --validation "$DATA/stage07_residual_buffer/features/validation.pt" \
    --output "$DATA/stage07_residual_buffer/checkpoints/${name}.pt" \
    --metrics-output "$STAGE/metrics/${name}.json" --max-task-id "$max_task" \
    --epochs 40 --device cuda:0 > "$STAGE/logs/${name}.log" 2>&1
done

CUDA_VISIBLE_DEVICES=4 "$PY" -m compose.cli.eval_sufficiency \
  --train "$DATA/stage07_residual_buffer/features/train.pt" \
  --validation "$DATA/stage07_residual_buffer/features/validation.pt" \
  --output "$HEAD" --metrics-output "$STAGE/metrics/sufficiency_validation.json" \
  --epochs 40 --device cuda:0 > "$STAGE/logs/train_eval_sufficiency.log" 2>&1

CUDA_VISIBLE_DEVICES=4 "$PY" -m compose.cli.build_residual_buffer \
  --features "$DATA/stage07_residual_buffer/features/train.pt" \
  --sufficiency-checkpoint "$HEAD" \
  --output-dir "$DATA/stage07_residual_buffer/buffers" --device cuda:0 \
  > "$STAGE/logs/build_buffers.log" 2>&1
cp "$DATA/stage07_residual_buffer/buffers/metrics.json" "$STAGE/metrics/residual_buffers.json"
printf '0\n' > "$STAGE/logs/stage07.exit"
