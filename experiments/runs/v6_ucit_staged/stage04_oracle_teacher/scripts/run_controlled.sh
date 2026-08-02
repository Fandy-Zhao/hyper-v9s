#!/usr/bin/env bash
set -euo pipefail
cd /home/zhaozhuofan/Hyper-LlaVA
export PYTHONPATH=/home/zhaozhuofan/Hyper-LlaVA
ROOT=experiments/runs/v6_ucit_staged/stage04_oracle_teacher
CACHE=/data/ckpt/zhaozhuofan/v6_ucit_staged/stage04_oracle_teacher/cache/controlled
RMS=/data/ckpt/zhaozhuofan/v6_ucit_staged/stage04_oracle_teacher/rms/controlled
PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
SCRIPT="$ROOT/scripts/oracle_generate.py"
IMAGES=experiments/data/controlled_format_v1
mkdir -p "$CACHE" "$RMS" "$ROOT/controlled_regression"

run_pair() {
  local gpu="$1" name="$2" checkpoint="$3" questions="$4" visible="$5" task_id="$6"
  mkdir -p "$CACHE/$name" "$ROOT/controlled_regression/$name"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT" \
    --checkpoint-kind compose --checkpoint "$checkpoint" --questions "$questions" \
    --images "$IMAGES" --task-id "$task_id" --task-name "$name" --split train_calibration \
    --temporal-scope post_task_diagnostic --visible-experts "$visible" \
    --config "$ROOT/configs/oracle_direct.yaml" --config-hash 10f2cdce7726e9f5cecdcebbd056555cd98393415e9665a5a80a4ce1f6b99fe5 \
    --cache-file "$CACHE/$name/direct.json" --summary-file "$ROOT/controlled_regression/$name/direct_summary.json" \
    --batch-size 2 --stage03-single-token-regression \
    > "$ROOT/controlled_regression/$name/direct.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT" \
    --checkpoint-kind compose --checkpoint "$checkpoint" --questions "$questions" \
    --calibration-questions "$questions" --images "$IMAGES" --task-id "$task_id" --task-name "$name" \
    --split train_calibration --temporal-scope post_task_diagnostic --visible-experts "$visible" \
    --config "$ROOT/configs/oracle_rms.yaml" --config-hash 12fa7623df4d392c748a61d8bf980c9610f1dcde25276eaae7b47701a24e0a8d \
    --rms-statistics "$RMS/$name.json" --cache-file "$CACHE/$name/rms.json" \
    --summary-file "$ROOT/controlled_regression/$name/rms_summary.json" --batch-size 2 \
    --stage03-single-token-regression > "$ROOT/controlled_regression/$name/rms.log" 2>&1
}

run_pair 4 a_independent_b /data/ckpt/zhaozhuofan/compose/format_controlled_composition_v1/assembled/seed42/a_independent_b /data/dataset/zhaozhuofan/v6_ucit_stage04/controlled/A_plus_B.json 0,1 1
run_pair 4 a_residual_b /data/ckpt/zhaozhuofan/v6_ucit_staged/stage03_composition/controlled/a_residual_b /data/dataset/zhaozhuofan/v6_ucit_stage04/controlled/A_plus_B.json 0,1 1
run_pair 4 independent_b_c /data/ckpt/zhaozhuofan/compose/format_controlled_composition_v1/assembled/seed42/independent_b_c /data/dataset/zhaozhuofan/v6_ucit_stage04/controlled/B_plus_C.json 1,2 2
run_pair 4 residual_b_c /data/ckpt/zhaozhuofan/compose/format_controlled_composition_v1/assembled/seed42/residual_b_c /data/dataset/zhaozhuofan/v6_ucit_stage04/controlled/B_plus_C.json 1,2 2
