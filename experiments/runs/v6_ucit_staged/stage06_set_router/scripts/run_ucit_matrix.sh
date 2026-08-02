#!/bin/bash
# Evaluate Mini-2, then Mini-3, then the remaining full 6x6 lower triangle.
set -euo pipefail

ROOT=/home/zhaozhuofan/Hyper-LlaVA
STAGE_ROOT="$ROOT/experiments/runs/v6_ucit_staged/stage06_set_router"
ART=/data/ckpt/zhaozhuofan/v6_ucit_staged/stage06_set_router
ROUTE_DIR=${ROUTE_DIR:-$ART/test_routes_anchor_capped}
CK=/data/ckpt/zhaozhuofan/v6_ucit_staged/stage01_baseline/checkpoints
OUT="$STAGE_ROOT/evaluations/continual_anchor"
TASKS=(ImageNet-R ArxivQA VizWiz IconQA CLEVR Flickr30k)
mkdir -p "$OUT" "$STAGE_ROOT/logs"

ROUTE_DIR="$ROUTE_DIR" bash "$STAGE_ROOT/scripts/run_test_routes.sh"
for limit in 2 3 6; do
    for stage in $(seq 1 "$limit"); do
        model="$CK/full_gb24_task${stage}_llava_lora_ours"
        for seen in $(seq 1 "$stage"); do
            dataset=${TASKS[$((seen-1))]}
            result="$OUT/$dataset/routed-task$stage/Result.text"
            if [ -s "$result" ]; then
                continue
            fi
            routes="$ROUTE_DIR/task${stage}_${dataset}.json"
            bash "$STAGE_ROOT/scripts/run_routed_eval.sh" "$dataset" "$stage" "$model" "$routes" "$OUT" 4,5,6,7
        done
    done
    echo "MINI_${limit}_COMPLETE"
done
PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
"$PY" "$STAGE_ROOT/scripts/summarize_routed_ucit.py" \
    --eval-root "$OUT" --route-root "$ROUTE_DIR" \
    --baseline "$ROOT/experiments/runs/v6_ucit_staged/stage01_baseline/metrics/gb24_matched_seed42.json" \
    --output "$STAGE_ROOT/metrics/full_ucit_continual_anchor.json"
echo FULL_ROUTED_UCIT_COMPLETE
