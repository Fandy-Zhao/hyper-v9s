#!/bin/bash
# Build the complete lower-triangular UCIT route matrix with at most four GPUs.
set -euo pipefail

PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
ROOT=/home/zhaozhuofan/Hyper-LlaVA
DATA=/data/dataset/zhaozhuofan/UCIT
ART=/data/ckpt/zhaozhuofan/v6_ucit_staged/stage06_set_router
QUERY_KEY=/data/ckpt/zhaozhuofan/v6_ucit_staged/stage05_query_keys/checkpoints/continual_anchor.pt
ROUTER="$ART/checkpoints/continual_anchor.pt"
TASKS=(ImageNet-R ArxivQA VizWiz IconQA CLEVR Flickr30k)
GPUS=(4 5 6 7)
ROUTE_DIR=${ROUTE_DIR:-$ART/test_routes}

mkdir -p "$ROUTE_DIR" "$ART/test_features" "$ART/logs"
cd "$ROOT"
source /home/zhaozhuofan/miniconda3/etc/profile.d/conda.sh
conda activate hyper

running=0
for stage in $(seq 1 6); do
    visible=$(seq -s, 0 $((stage-1)))
    for seen in $(seq 1 "$stage"); do
        dataset=${TASKS[$((seen-1))]}
        output="$ROUTE_DIR/task${stage}_${dataset}.json"
        if [ -s "$output" ]; then
            continue
        fi
        gpu=${GPUS[$running]}
        CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m compose.cli.route_test_questions \
            --questions "$DATA/instructions/$dataset/test_3000.json" \
            --images "$DATA/datasets" \
            --query-key-checkpoint "$QUERY_KEY" \
            --router-checkpoint "$ROUTER" \
            --visible-experts "$visible" \
            --feature-cache "$ART/test_features/$dataset.pt" \
            --output "$output" --batch-size 32 --device cuda:0 \
            > "$ART/logs/route_task${stage}_${dataset}.log" 2>&1 &
        running=$((running+1))
        if [ "$running" -eq 4 ]; then
            wait
            running=0
        fi
    done
done
wait
echo ROUTE_MATRIX_DONE
