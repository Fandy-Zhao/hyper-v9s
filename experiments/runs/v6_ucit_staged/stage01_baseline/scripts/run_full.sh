#!/bin/bash
# Stage 01 full UCIT seed42 pipeline: 6 tasks in official order, each task
# trained then all seen tasks evaluated with the official evaluator
# (test_3000, chunked over the eval GPU list).
#
# Usage: run_full.sh <original_batch|gb24_matched> <GPU_LIST> <PORT_BASE> <TAG> <EVAL_GPU_LIST> [START_TASK]
#   START_TASK: resume from this task (1-based); prior task checkpoints must
#               exist on disk; prior evals are re-run for the start task stage.
set -euo pipefail

CONFIG=$1
GPU_LIST=$2
PORT_BASE=$3
TAG=$4
EVAL_GPU_LIST="${5:-0,1,2,3}"
START_TASK="${6:-1}"

BASE=/home/zhaozhuofan/Hyper-LlaVA/experiments/runs/v6_ucit_staged/stage01_baseline
PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
CK=/data/ckpt/zhaozhuofan/v6_ucit_staged/stage01_baseline/checkpoints
EVAL_ROOT="$BASE/evaluations/full_$TAG"
LOG_DIR="$BASE/logs"
mkdir -p "$EVAL_ROOT"

# Official UCIT order (locked in Stage 00)
TASKS=(ImageNet-R ArxivQA VizWiz IconQA CLEVR Flickr30k)
DATA_FILES=(ImageNet-R/train.json ArxivQA/train_4w.json VizWiz/train.json IconQA/train.json CLEVR/train_4w.json Flickr30k/train_brief_4w.json)

PREV=""
if [ "$START_TASK" -gt 1 ]; then
    PREV="$CK/full_${TAG}_task$((START_TASK-1))_llava_lora_ours"
    if [ ! -d "$PREV" ]; then
        echo "ERROR: resume requires prior checkpoint $PREV" >&2
        exit 1
    fi
    # backfill missing evals of prior stages
    for t in $(seq 1 $((START_TASK-1))); do
        T_OUT="$CK/full_${TAG}_task${t}_llava_lora_ours"
        for s in $(seq 1 "$t"); do
            R_TEXT="$EVAL_ROOT/${TASKS[$((s-1))]}/hyper-task${t}/Result.text"
            if [ ! -f "$R_TEXT" ]; then
                echo "[$(date +%H:%M)] backfill eval ${TASKS[$((s-1))]} hyper-task${t}"
                bash "$BASE/scripts/run_eval.sh" "${TASKS[$((s-1))]}" "hyper-task${t}" "$T_OUT" "$EVAL_ROOT" "$EVAL_GPU_LIST"
            fi
        done
    done
fi
for t in $(seq "$START_TASK" 6); do
    T_OUT="$CK/full_${TAG}_task${t}_llava_lora_ours"
    LAUNCH_ARGS=(--task "$t" --config "$CONFIG" --gpus "$GPU_LIST" \
        --tag "full_${TAG}_task${t}" \
        --data-path "/data/dataset/zhaozhuofan/UCIT/instructions/${DATA_FILES[$((t-1))]}" \
        --output-dir "$T_OUT" --port "$((PORT_BASE + t))")
    if [ -n "$PREV" ]; then
        LAUNCH_ARGS+=(--prev-task-path "$PREV")
    fi
    SKIP_TRAIN=0
    if [ -f "$T_OUT/adapter_model.bin" ] && [ -z "$(ls -d "$T_OUT"/checkpoint-* 2>/dev/null || true)" ]; then
        SKIP_TRAIN=1
        echo "[$(date +%H:%M)] full_${TAG} task $t checkpoint complete, skipping training"
    fi
    if [ "$SKIP_TRAIN" = 0 ]; then
        $PY "$BASE/scripts/make_launch.py" "${LAUNCH_ARGS[@]}" >/dev/null
        echo "[$(date +%H:%M)] starting full_${TAG} task $t (${TASKS[$((t-1))]})"
        bash "$BASE/generated/run_full_${TAG}_task${t}.sh" > "$LOG_DIR/full_${TAG}_task${t}.log" 2>&1
        echo "[$(date +%H:%M)] full_${TAG} task $t exit: $? -> $T_OUT"
    fi

    # evaluate all seen tasks with the official evaluator
    for s in $(seq 1 "$t"); do
        bash "$BASE/scripts/run_eval.sh" "${TASKS[$((s-1))]}" "hyper-task${t}" "$T_OUT" "$EVAL_ROOT" "$EVAL_GPU_LIST"
        echo "[$(date +%H:%M)] eval ${TASKS[$((s-1))]} hyper-task${t} done"
    done
    PREV="$T_OUT"
done

echo "FULL_${TAG}_DONE"
