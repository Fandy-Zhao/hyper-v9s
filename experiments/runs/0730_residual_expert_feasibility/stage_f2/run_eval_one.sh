#!/usr/bin/env bash
set -euo pipefail
[[ "$#" -eq 4 ]] || { echo "Usage: $0 MODEL_NAME SEED DATASET GPU" >&2; exit 2; }
model_name="$1"
seed="$2"
dataset="$3"
gpu="$4"
case "${seed}" in 42|43|44) ;; *) echo "Unsupported seed: ${seed}" >&2; exit 2 ;; esac
case "${dataset}" in A_only|B_only|C_only|A_plus_B|B_plus_C) ;; *) echo "Unsupported dataset: ${dataset}" >&2; exit 2 ;; esac

ROOT="/home/zhaozhuofan/Hyper-LlaVA"
MODEL_PATH="/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
VISION_TOWER="/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
TRAIN_ROOT="/data/ckpt/zhaozhuofan/compose/residual_feasibility_v1/seed${seed}"
ASSEMBLED_ROOT="/data/ckpt/zhaozhuofan/compose/residual_feasibility_v1/assembled/seed${seed}"
DATA_ROOT="/data/dataset/zhaozhuofan/Hyper-LlaVA/controlled_functional_v1"
RESULT_ROOT="${ROOT}/experiments/runs/0730_residual_expert_feasibility/stage_f2/evaluation/seed${seed}/${dataset}/${model_name}"
checkpoint=""
expert_ids=""
case "${model_name}" in
  base) checkpoint="${TRAIN_ROOT}/expert_a"; expert_ids="" ;;
  expert_a) checkpoint="${TRAIN_ROOT}/expert_a"; expert_ids=0 ;;
  independent_b) checkpoint="${TRAIN_ROOT}/independent_b"; expert_ids=1 ;;
  residual_b) checkpoint="${TRAIN_ROOT}/residual_b"; expert_ids=1 ;;
  a_independent_b) checkpoint="${ASSEMBLED_ROOT}/a_independent_b"; expert_ids=0,1 ;;
  a_residual_b) checkpoint="${TRAIN_ROOT}/residual_b"; expert_ids=0,1 ;;
  expert_c) checkpoint="${TRAIN_ROOT}/expert_c"; expert_ids=2 ;;
  independent_b_c) checkpoint="${ASSEMBLED_ROOT}/independent_b_c"; expert_ids=1,2 ;;
  residual_b_c) checkpoint="${ASSEMBLED_ROOT}/residual_b_c"; expert_ids=1,2 ;;
  rank16_ab) checkpoint="${TRAIN_ROOT}/rank16_ab"; expert_ids=0 ;;
  task_ab) checkpoint="${TRAIN_ROOT}/task_ab"; expert_ids=0 ;;
  upper_bc) checkpoint="${TRAIN_ROOT}/upper_bc"; expert_ids=0 ;;
  *) echo "Unknown model: ${model_name}" >&2; exit 2 ;;
esac
for required in "${checkpoint}/compose_experts.bin" "${checkpoint}/compose_experts.json" \
  "${DATA_ROOT}/instructions/${dataset}/test.json"; do
  [[ -s "${required}" ]] || { echo "Missing required input: ${required}" >&2; exit 1; }
done
if [[ -e "${RESULT_ROOT}/summary.json" || -e "${RESULT_ROOT}/per_sample.jsonl" ]]; then
  echo "Refusing existing evaluation: ${RESULT_ROOT}" >&2
  exit 1
fi
mkdir -p "${RESULT_ROOT}"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader,nounits -l 2 \
  > "${RESULT_ROOT}/gpu.csv" &
monitor_pid="$!"
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT
args=(
  --model-path "${MODEL_PATH}" --checkpoint-dir "${checkpoint}"
  --projector-path "${MODEL_PATH}/mm_projector.bin" --vision-tower "${VISION_TOWER}"
  --question-file "${DATA_ROOT}/instructions/${dataset}/test.json"
  --image-folder "/data/dataset/zhaozhuofan/Hyper-LlaVA"
  --output-dir "${RESULT_ROOT}" --selection-name "${model_name}"
  --expert-ids "${expert_ids}" --gates "$(sed 's/[^,]*/1/g' <<< "${expert_ids}")"
  --normalization none --batch-size 8 --max-new-tokens 16
  --experiment-seed "${seed}" --device cuda:0
)
{
  printf 'CUDA_VISIBLE_DEVICES=%q /home/zhaozhuofan/miniconda3/envs/hyper/bin/python -m compose.eval.eval_controlled' "${gpu}"
  printf ' %q' "${args[@]}"
  printf '\n'
} > "${RESULT_ROOT}/command.txt"
cd "${ROOT}"
set +e
CUDA_VISIBLE_DEVICES="${gpu}" /usr/bin/time -v /home/zhaozhuofan/miniconda3/envs/hyper/bin/python \
  -m compose.eval.eval_controlled "${args[@]}" > "${RESULT_ROOT}/eval.log" 2>&1
status="$?"
set -e
printf 'exit_code=%s\nfinished_utc=%s\n' "${status}" "$(date -u +%FT%TZ)" > "${RESULT_ROOT}/status.txt"
if [[ "${status}" -eq 0 ]]; then
  sha256sum "${RESULT_ROOT}/per_sample.jsonl" "${RESULT_ROOT}/summary.json" \
    "${checkpoint}/compose_experts.bin" "${checkpoint}/compose_experts.json" \
    > "${RESULT_ROOT}/SHA256SUMS"
fi
exit "${status}"
