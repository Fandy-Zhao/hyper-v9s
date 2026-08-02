#!/usr/bin/env bash
set -u
cd /home/zhaozhuofan/Hyper-LlaVA
export PYTHONPATH=/home/zhaozhuofan/Hyper-LlaVA
ROOT=experiments/runs/v6_ucit_staged/stage03_composition/controlled_regression
SCRIPT=experiments/runs/v6_ucit_staged/stage03_composition/scripts/controlled_direct_sum_regression.py
CKPT=/data/ckpt/zhaozhuofan/compose/format_controlled_composition_v1/assembled/seed42
DATA=experiments/data/controlled_format_v1_training/instructions
IMAGES=experiments/data/controlled_format_v1
PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
mkdir -p "$ROOT"

CUDA_VISIBLE_DEVICES=4 "$PY" "$SCRIPT" --checkpoint "$CKPT/a_independent_b" --questions "$DATA/A_plus_B/test_eval.json" --images "$IMAGES" --expert-ids 0,1 --output "$ROOT/a_independent_b.json" > "$ROOT/a_independent_b.log" 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=5 "$PY" "$SCRIPT" --checkpoint /data/ckpt/zhaozhuofan/v6_ucit_staged/stage03_composition/controlled/a_residual_b --questions "$DATA/A_plus_B/test_eval.json" --images "$IMAGES" --expert-ids 0,1 --output "$ROOT/a_residual_b.json" > "$ROOT/a_residual_b.log" 2>&1 & p2=$!
CUDA_VISIBLE_DEVICES=6 "$PY" "$SCRIPT" --checkpoint "$CKPT/independent_b_c" --questions "$DATA/B_plus_C/test_eval.json" --images "$IMAGES" --expert-ids 1,2 --output "$ROOT/independent_b_c.json" > "$ROOT/independent_b_c.log" 2>&1 & p3=$!
CUDA_VISIBLE_DEVICES=7 "$PY" "$SCRIPT" --checkpoint "$CKPT/residual_b_c" --questions "$DATA/B_plus_C/test_eval.json" --images "$IMAGES" --expert-ids 1,2 --output "$ROOT/residual_b_c.json" > "$ROOT/residual_b_c.log" 2>&1 & p4=$!

status=0
for pid in "$p1" "$p2" "$p3" "$p4"; do
  wait "$pid" || status=1
done
exit "$status"
