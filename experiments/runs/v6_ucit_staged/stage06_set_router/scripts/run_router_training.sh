#!/usr/bin/env bash
set -euo pipefail

cd /home/zhaozhuofan/Hyper-LlaVA
export PYTHONPATH=/home/zhaozhuofan/Hyper-LlaVA
ROOT=experiments/runs/v6_ucit_staged/stage06_set_router
DATA=/data/ckpt/zhaozhuofan/v6_ucit_staged
PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
mkdir -p "$ROOT/logs" "$ROOT/metrics" "$ROOT/manifests" "$DATA/stage06_set_router/checkpoints"
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  nvidia-smi
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
} > "$ROOT/manifests/gpu_preflight_router_training.txt"

CUDA_VISIBLE_DEVICES=4 "$PY" -m compose.cli.train_set_router \
  --features "$DATA/stage05_query_keys/features/bundles/train.pt" \
  --query-key-checkpoint "$DATA/stage05_query_keys/checkpoints/continual_anchor.pt" \
  --output "$DATA/stage06_set_router/checkpoints/continual_anchor.pt" --mode continual_anchor --epochs 40 --seed 42 --device cuda:0 \
  > "$ROOT/logs/train_continual_anchor.log" 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=5 "$PY" -m compose.cli.train_set_router \
  --features "$DATA/stage05_query_keys/features/bundles/train.pt" \
  --query-key-checkpoint "$DATA/stage05_query_keys/checkpoints/offline_diagnostic.pt" \
  --output "$DATA/stage06_set_router/checkpoints/offline_diagnostic.pt" --mode offline_diagnostic --epochs 40 --seed 42 --device cuda:0 \
  > "$ROOT/logs/train_offline_diagnostic.log" 2>&1 & p2=$!
status=0; wait "$p1" || status=1; wait "$p2" || status=1; [[ "$status" == 0 ]]

CUDA_VISIBLE_DEVICES=4 "$PY" -m compose.cli.eval_set_router \
  --features "$DATA/stage05_query_keys/features/bundles/validation.pt" \
  --query-key-checkpoint "$DATA/stage05_query_keys/checkpoints/continual_anchor.pt" \
  --router-checkpoint "$DATA/stage06_set_router/checkpoints/continual_anchor.pt" \
  --output "$ROOT/metrics/continual_anchor_validation.json" --features-output "$DATA/stage06_set_router/features/validation_router.pt" --device cuda:0 \
  > "$ROOT/logs/eval_continual_anchor.log" 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=5 "$PY" -m compose.cli.eval_set_router \
  --features "$DATA/stage05_query_keys/features/bundles/validation.pt" \
  --query-key-checkpoint "$DATA/stage05_query_keys/checkpoints/offline_diagnostic.pt" \
  --router-checkpoint "$DATA/stage06_set_router/checkpoints/offline_diagnostic.pt" \
  --output "$ROOT/metrics/offline_diagnostic_validation.json" --device cuda:0 \
  > "$ROOT/logs/eval_offline_diagnostic.log" 2>&1 & p2=$!
status=0; wait "$p1" || status=1; wait "$p2" || status=1; [[ "$status" == 0 ]]
printf '0\n' > "$ROOT/logs/router_training.exit"
