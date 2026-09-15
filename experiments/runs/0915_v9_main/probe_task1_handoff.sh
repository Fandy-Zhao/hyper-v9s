#!/usr/bin/env bash
# The task-1 handoff probe -- the one path the task-0 preflight structurally
# cannot reach, checked once before it is first executed 5.8 h into the formal
# chain.
#
# The preflight ran task 0, where the pool starts empty: no historical expert,
# no committed pool to load, no task key to audit, and a routing row that is just
# the M candidates.  Everything the continual setting adds is therefore
# unexecuted code until task 1 starts:
#
#   * load the committed task-0 pool + its candidate LoRA as *frozen history*
#   * merge the historical keys into the routing row (Top-C recall over history)
#   * run the periodic wide retrieval over the grown pool
#   * create this task's temporary New Key, trainable while the old key is frozen
#   * train with historical experts participating in the gate and in the
#     deployment rule, under the §38 contract that their LoRA takes no answer
#     gradient and their key takes no gradient at all
#   * audit the task key and commit *new* candidates on top of a non-empty pool
#
# It is a compressed run in the identical sense as the preflight: only the split
# size differs (1600 ArxivQA train samples -> 50 optimizer steps instead of
# 6601).  Real data, real 2-rank DDP, real calibration, real audit, real commit;
# the recipe comes from configs/v9s_main.yaml, the config the formal chain runs.
#
# This probe's own outputs are throwaway -- it consumes the *preflight's* task 0,
# so its task-1 pool is not the formal one and the formal chain must still start
# from task 0.  Its product is the answer to one question: does the handoff path
# execute, and does it produce sane numbers while doing it.
set -euo pipefail

PY=/root/miniconda3/envs/hyper/bin/python
REPO=/root/autodl-tmp/Hyper-LlaVA
UCIT=/root/autodl-tmp/data/dataset/zhaozhuofan/UCIT
MODELS=/root/autodl-tmp/data/ckpt/zhaozhuofan/models
RUN=$REPO/experiments/runs/0915_v9_main/probe_task1_handoff
PREV=$REPO/experiments/runs/0915_v9_preflight/my_run_v9s/training/task0
GPU=0

TRAIN_SAMPLES=1600   # 1600 / (2 ranks x 2 per device x 8 accumulation) = 50 steps
VAL_SAMPLES=64
WORLD=2

cd "$REPO"
mkdir -p "$RUN/data"

# ---- smoke splits, from the *task-1* dataset (ArxivQA) --------------------
if [ ! -f "$RUN/data/train_smoke.json" ]; then
  "$PY" - "$UCIT" "$RUN" "$TRAIN_SAMPLES" "$VAL_SAMPLES" <<'PY'
import json, sys
ucit, run, train_n, val_n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
for src, dst, count in (
    (f"{ucit}/v7_train/ArxivQA/train.json", f"{run}/data/train_smoke.json", train_n),
    (f"{ucit}/v7_validation/ArxivQA/validation.json", f"{run}/data/val_smoke.json", val_n),
):
    records = json.load(open(src))
    json.dump(records[:count], open(dst, "w"))
    print(f"{src} -> {dst}: {len(records[:count])} records")
PY
fi

# ---- the whole pipeline for task 1, against the preflight's committed task 0
# No --stop-after: the audit and the commit are two of the paths under test.
CUDA_VISIBLE_DEVICES=$GPU "$PY" -m compose.experiments.v9_task_run \
  --config configs/v9s_main.yaml \
  --root "$RUN" \
  --task-index 1 \
  --train-file "$RUN/data/train_smoke.json" \
  --val-file "$RUN/data/val_smoke.json" \
  --previous-checkpoint "$PREV" \
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
  --save-steps 25 \
  --save-total-limit 2 \
  --dataloader-num-workers 2 \
  --calibrate \
  2>&1 | tee "$REPO/experiments/runs/0915_v9_main/probe_task1_handoff.log"
