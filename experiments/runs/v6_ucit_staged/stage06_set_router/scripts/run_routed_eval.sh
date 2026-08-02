#!/bin/bash
# Generate and officially score one answer-free routed UCIT cell on four GPUs.
set -euo pipefail

DATASET=$1
STAGE=$2
MODEL=$3
ROUTES=$4
OUTPUT_ROOT=$5
GPU_LIST=${6:-4,5,6,7}
PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
DATA=/data/dataset/zhaozhuofan/UCIT
export PATH="/home/zhaozhuofan/miniconda3/envs/hyper/bin:$PATH"

case "$DATASET" in
    ImageNet-R) SCORER=eval_deepseek_r1; INTRO_STAGE=1 ;;
    ArxivQA) SCORER=eval_deepseek_r1; INTRO_STAGE=2 ;;
    VizWiz) SCORER=eval_caption; INTRO_STAGE=3 ;;
    IconQA) SCORER=eval_deepseek_r1; INTRO_STAGE=4 ;;
    CLEVR) SCORER=eval_deepseek_r1; INTRO_STAGE=5 ;;
    Flickr30k) SCORER=eval_caption; INTRO_STAGE=6 ;;
    *) echo "unknown dataset: $DATASET" >&2; exit 2 ;;
esac
QUESTIONS="$DATA/instructions/$DATASET/test_3000.json"
OUT="$OUTPUT_ROOT/$DATASET/routed-task$STAGE"
mkdir -p "$OUT"
IFS=',' read -ra GPUS <<< "$GPU_LIST"
CHUNKS=${#GPUS[@]}
if [ "$CHUNKS" -gt 4 ]; then
    echo "at most four GPUs are permitted" >&2
    exit 2
fi
REUSE_ARGS=()
# Registered experts are immutable.  A dataset has a frozen Stage-01 answer
# artifact for expert e exactly when it had already been introduced by e+1.
for expert in $(seq 0 $((STAGE-1))); do
    expert_stage=$((expert+1))
    if [ "$expert_stage" -ge "$INTRO_STAGE" ]; then
        reuse=/home/zhaozhuofan/Hyper-LlaVA/experiments/runs/v6_ucit_staged/stage01_baseline/evaluations/full_gb24/$DATASET/hyper-task$expert_stage/merge.jsonl
        if [ -s "$reuse" ]; then
            REUSE_ARGS+=(--reuse-expert-answer "$expert=$reuse")
        fi
    fi
done

if "$PY" -m compose.cli.assemble_frozen_single_answers \
    --questions "$QUESTIONS" --routes "$ROUTES" "${REUSE_ARGS[@]/--reuse-expert-answer/--expert-answer}" \
    --output "$OUT/merge.jsonl" --metrics "$OUT/reuse_metrics.json"; then
    REUSED_COMPLETE=1
else
    status=$?
    if [ "$status" -ne 3 ]; then
        exit "$status"
    fi
    REUSED_COMPLETE=0
fi

if [ "$REUSED_COMPLETE" -eq 0 ]; then
for index in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[$index]} "$PY" -m compose.cli.model_answer_routed \
        --model-path "$MODEL" \
        --model-base /data/ckpt/zhaozhuofan/models/llava-v1.5-7b \
        --text-tower /data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336 \
        --question-file "$QUESTIONS" --image-folder "$DATA/datasets" \
        --route-decisions "$ROUTES" --answers-file "$OUT/${CHUNKS}_${index}.jsonl" \
        --metrics-file "$OUT/${CHUNKS}_${index}_metrics.json" \
        "${REUSE_ARGS[@]}" \
        --num-chunks "$CHUNKS" --chunk-idx "$index" --conv-mode vicuna_v1 \
        > "$OUT/${CHUNKS}_${index}.log" 2>&1 &
done
wait

merge="$OUT/merge.jsonl"
: > "$merge"
for index in $(seq 0 $((CHUNKS-1))); do
    cat "$OUT/${CHUNKS}_${index}.jsonl" >> "$merge"
done
else
merge="$OUT/merge.jsonl"
fi
if [ "$(wc -l < "$merge")" -ne 3000 ]; then
    echo "routed answer count is not 3000" >&2
    exit 1
fi
if [ "$SCORER" = eval_caption ]; then
    "$PY" -m llava.eval.eval_caption \
        --annotation-file "$DATA/instructions/$DATASET/val_coco_type_3000.json" \
        --result-file "$merge" --output-dir "$OUT"
else
    "$PY" -m llava.eval.eval_deepseek_r1 \
        --annotation-file "$QUESTIONS" --result-file "$merge" --output-dir "$OUT"
fi
echo "ROUTED_CELL_DONE $DATASET task$STAGE"
