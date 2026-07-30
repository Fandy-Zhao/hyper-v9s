#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/zhaozhuofan/Hyper-LlaVA"
MODEL="/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
VISION="/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
DATA="/data/dataset/zhaozhuofan/Hyper-LlaVA/controlled_functional_v1/instructions/A_only/test.json"
OUT="${ROOT}/experiments/runs/0730_residual_expert_feasibility/stage_f2/diagnostics"
PYTHON="/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"
mkdir -p "${OUT}"
cd "${ROOT}"
for seed in 42 43 44; do
  source="/data/ckpt/zhaozhuofan/compose/residual_feasibility_v1/seed${seed}/expert_a"
  residual="/data/ckpt/zhaozhuofan/compose/residual_feasibility_v1/seed${seed}/residual_b"
  seed_out="${OUT}/seed${seed}"
  mkdir -p "${seed_out}"
  "${PYTHON}" -m compose.experts.verify_isolation \
    --before-dir "${source}" --after-dir "${residual}" --expert-ids 0 \
    --output-file "${seed_out}/tensor_isolation.json" > "${seed_out}/tensor_isolation.log" 2>&1
  "${PYTHON}" -m compose.experts.analyze_geometry \
    --first-checkpoint "${residual}" --first-expert-id 0 \
    --second-checkpoint "${residual}" --second-expert-id 1 \
    --output-file "${seed_out}/old_residual_geometry.json" > "${seed_out}/geometry.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m compose.experts.verify_behavior \
    --model-path "${MODEL}" --before-checkpoint "${source}" --after-checkpoint "${residual}" \
    --expert-id 0 --projector-path "${MODEL}/mm_projector.bin" --vision-tower "${VISION}" \
    --question-file "${DATA}" --image-folder /data/dataset/zhaozhuofan/Hyper-LlaVA \
    --output-file "${seed_out}/fixed_logits.json" --sample-count 8 --device cuda:0 \
    > "${seed_out}/fixed_logits.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m compose.experts.activation_identity \
    --model-path "${MODEL}" --checkpoint-dir "${residual}" --expert-id 1 \
    --projector-path "${MODEL}/mm_projector.bin" --vision-tower "${VISION}" \
    --data-root /data/dataset/zhaozhuofan/Hyper-LlaVA/controlled_functional_v1 \
    --image-folder /data/dataset/zhaozhuofan/Hyper-LlaVA \
    --output-file "${seed_out}/activation_identity.json" --samples-per-dataset 16 \
    --device cuda:0 > "${seed_out}/activation_identity.log" 2>&1
  sha256sum "${seed_out}/tensor_isolation.json" "${seed_out}/old_residual_geometry.json" \
    "${seed_out}/fixed_logits.json" "${seed_out}/activation_identity.json" \
    > "${seed_out}/SHA256SUMS"
done
date -u +%FT%TZ > "${OUT}/complete_utc.txt"
