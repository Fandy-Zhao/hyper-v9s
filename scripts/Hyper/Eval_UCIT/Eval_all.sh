#!/bin/bash

set -e

CKPT_ROOT="${1:-/data/ckpt/zhaozhuofan/hyper_llava/Hyper/UCIT/06_18}"
RESULT_ROOT="${2:-/home/zhaozhuofan/Hyper-LlaVA/runs/results/UCIT/06_18}"

run_task_eval() {
    local task_id="$1"
    local stage="hyper-task${task_id}"
    local model_path="${CKPT_ROOT}/Task${task_id}"

    if [ ! -d "$model_path" ]; then
        echo "Missing checkpoint directory: $model_path" >&2
        exit 1
    fi

    echo "Evaluating ${stage}: ${model_path}"
    echo "Writing results under: ${RESULT_ROOT}"
    bash scripts/Hyper/Eval_UCIT/eval_imagenet.sh "$stage" "$model_path" "$RESULT_ROOT"
    bash scripts/Hyper/Eval_UCIT/eval_arxivqa.sh "$stage" "$model_path" "$RESULT_ROOT"
    bash scripts/Hyper/Eval_UCIT/eval_vizwiz.sh "$stage" "$model_path" "$RESULT_ROOT"
    bash scripts/Hyper/Eval_UCIT/eval_iconqa.sh "$stage" "$model_path" "$RESULT_ROOT"
    bash scripts/Hyper/Eval_UCIT/eval_clevr.sh "$stage" "$model_path" "$RESULT_ROOT"
    bash scripts/Hyper/Eval_UCIT/eval_flickr30k.sh "$stage" "$model_path" "$RESULT_ROOT"
}

run_task_eval 6

# Examples for evaluating earlier stages with the same checkpoint root:
# run_task_eval 5
# run_task_eval 4
# run_task_eval 3
# run_task_eval 2
# run_task_eval 1


