#!/usr/bin/env bash
set -euo pipefail

cd /home/zhaozhuofan/Hyper-LlaVA
export PYTHONPATH=/home/zhaozhuofan/Hyper-LlaVA
ROOT=experiments/runs/v6_ucit_staged/stage05_query_keys
DATA=/data/ckpt/zhaozhuofan/v6_ucit_staged/stage05_query_keys
PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
mkdir -p "$ROOT/logs" "$ROOT/metrics" "$ROOT/validation" "$DATA/checkpoints"

"$PY" "$ROOT/scripts/validate_oracle_expansion.py" | tee "$ROOT/logs/validate_oracle_expansion.log"
bash "$ROOT/scripts/run_feature_extraction.sh" full > "$ROOT/logs/feature_extraction_full_supervisor.log" 2>&1
"$PY" "$ROOT/scripts/aggregate_feature_bundles.py" | tee "$ROOT/logs/aggregate_feature_bundles.log"

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  nvidia-smi
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
} > "$ROOT/manifests/gpu_preflight_query_key_training.txt"

CUDA_VISIBLE_DEVICES=4 "$PY" -m compose.cli.train_expert_keys \
  --features "$DATA/features/bundles/train.pt" --output "$DATA/checkpoints/continual_anchor.pt" \
  --mode continual_anchor --epochs 40 --seed 42 --device cuda:0 > "$ROOT/logs/train_continual_anchor.log" 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=5 "$PY" -m compose.cli.train_expert_keys \
  --features "$DATA/features/bundles/train.pt" --output "$DATA/checkpoints/offline_diagnostic.pt" \
  --mode offline_diagnostic --epochs 40 --seed 42 --device cuda:0 > "$ROOT/logs/train_offline_diagnostic.log" 2>&1 & p2=$!
status=0; wait "$p1" || status=1; wait "$p2" || status=1; [[ "$status" == 0 ]]

CUDA_VISIBLE_DEVICES=4 "$PY" -m compose.cli.eval_expert_retrieval \
  --features "$DATA/features/bundles/validation.pt" --checkpoint "$DATA/checkpoints/continual_anchor.pt" \
  --output "$ROOT/metrics/continual_anchor.json" --device cuda:0 > "$ROOT/logs/eval_continual_anchor.log" 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=5 "$PY" -m compose.cli.eval_expert_retrieval \
  --features "$DATA/features/bundles/validation.pt" --checkpoint "$DATA/checkpoints/offline_diagnostic.pt" \
  --output "$ROOT/metrics/offline_diagnostic.json" --device cuda:0 > "$ROOT/logs/eval_offline_diagnostic.log" 2>&1 & p2=$!
status=0; wait "$p1" || status=1; wait "$p2" || status=1; [[ "$status" == 0 ]]

"$PY" -m compileall -q compose
"$PY" -m pytest -q tests/compose > "$ROOT/logs/tests_compose_full_final.log" 2>&1
bash -n "$ROOT/scripts/run_oracle_expansion.sh"
bash -n "$ROOT/scripts/run_feature_extraction.sh"
bash -n "$ROOT/scripts/run_stage05_after_oracle.sh"
git diff --check
"$PY" "$ROOT/scripts/finalize_stage05.py" | tee "$ROOT/logs/finalize_stage05.log"
printf '0\n' > "$ROOT/logs/stage05_after_oracle.exit"
