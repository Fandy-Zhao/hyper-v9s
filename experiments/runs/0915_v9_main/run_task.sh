#!/usr/bin/env bash
# One formal V9-S task, start to finish (spec §29 execution order).
#
#   run_task.sh <task-index 0..5>
#
# Each task gets its own --root.  That is forced by the query cache, not a
# preference: the cache path is features/train.json with no task in the name and
# an existing file is never re-encoded (spec §4), so a shared root would hand
# task 1 the ImageNet-R queries written by task 0.  Per-task roots also keep
# each task's metrics, audit and the pool handoff next to the task that produced
# them.  The chain still shares one --previous-checkpoint per step, which
# _previous_state_path resolves to the predecessor's committed state.
#
# Re-running a task that died resumes it: train_compose passes
# resume_from_checkpoint=True whenever output_dir already holds checkpoint-*
# entries and the mode is resumable, and the V9 trainer restores router, stage,
# rms and optimizer state from the checkpoint's v9_state.pt (which is why the
# config is fingerprinted there -- a changed config refuses to resume).
#
# A task that already committed its pool is a no-op, so the chain is safe to
# restart from the top at any point.
set -euo pipefail

PY=/root/miniconda3/envs/hyper/bin/python
REPO=/root/autodl-tmp/Hyper-LlaVA
UCIT=/root/autodl-tmp/data/dataset/zhaozhuofan/UCIT
MODELS=/root/autodl-tmp/data/ckpt/zhaozhuofan/models
BASE=$REPO/experiments/runs/0915_v9_main
GPU=0
WORLD=2

TASK="${1:?usage: run_task.sh <task-index 0..5>}"
NAMES=(ImageNet-R ArxivQA VizWiz IconQA CLEVR Flickr30k)
NAME="${NAMES[$TASK]:?task index must be 0..5}"

RUN=$BASE/task$TASK
LOG=$BASE/logs/task$TASK.log
TRAIN_FILE=$UCIT/v7_train/$NAME/train.json
VAL_FILE=$UCIT/v7_validation/$NAME/validation.json
COMMITTED=$RUN/state/key_pool_task$TASK.pt
STATISTICS=$RUN/training/task$TASK/v9_task_statistics.json

mkdir -p "$RUN" "$BASE/logs"

if [ -f "$COMMITTED" ] && [ -f "$STATISTICS" ]; then
  echo "[chain] task $TASK ($NAME) already committed -- skipping"
  exit 0
fi

# The full-split query caches are the run's largest artefact (~19.7 GB across the
# six tasks, measured at 93.4 KB/sample).  Running out of disk mid-task would
# cost the whole task, so refuse to start one that cannot finish.
FREE_GB=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt 6 ]; then
  echo "[chain] refusing to start task $TASK: only ${FREE_GB}G free" >&2
  exit 3
fi

PREV_ARGS=()
if [ "$TASK" -gt 0 ]; then
  PREV=$BASE/task$((TASK - 1))/training/task$((TASK - 1))
  if [ ! -f "$PREV/compose_experts.bin" ]; then
    echo "[chain] task $TASK needs the committed task $((TASK - 1)) at $PREV" >&2
    exit 4
  fi
  PREV_ARGS=(--previous-checkpoint "$PREV")
fi

echo "[chain] task $TASK = $NAME  root=$RUN  train=$(python -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$TRAIN_FILE")  free=${FREE_GB}G"

cd "$REPO"
CUDA_VISIBLE_DEVICES=$GPU "$PY" -m compose.experiments.v9_task_run \
  --config configs/v9s_main.yaml \
  --root "$RUN" \
  --task-index "$TASK" \
  --train-file "$TRAIN_FILE" \
  --val-file "$VAL_FILE" \
  "${PREV_ARGS[@]}" \
  --model-path "$MODELS/llava-v1.5-7b" \
  --vision-tower "$MODELS/clip-vit-large-patch14-336" \
  --projector-path "$MODELS/llava-v1.5-7b/mm_projector.bin" \
  --image-folder "$UCIT/datasets" \
  --query-encoder "$MODELS/clip-vit-large-patch14-336" \
  --device cuda:0 \
  --training-world-size "$WORLD" \
  --training-launcher local-ranks \
  --training-per-device-batch-size 2 \
  --training-gradient-accumulation-steps 8 \
  --num-train-epochs 1 \
  --learning-rate 2e-04 \
  --v9-key-learning-rate 3e-04 \
  --logging-steps 5 \
  --save-strategy steps \
  --save-steps 250 \
  --save-total-limit 2 \
  --dataloader-num-workers 2 \
  --require-full-coverage \
  --calibrate \
  2>&1 | tee "$LOG"

echo "[chain] task $TASK ($NAME) committed:"
"$PY" - "$RUN" "$TASK" <<'PY'
import json, sys
run, task = sys.argv[1], sys.argv[2]
for name in (f"data/audit_task{task}.json", f"training/task{task}/v9_contribution_calibration.json",
             f"training/task{task}/v9_full_data_coverage.json",
             f"training/task{task}/v9_candidate_validation_gain.json"):
    try:
        print(f"  {name}: {json.dumps(json.load(open(f'{run}/{name}')), ensure_ascii=False)[:400]}")
    except FileNotFoundError:
        print(f"  {name}: MISSING")
PY
