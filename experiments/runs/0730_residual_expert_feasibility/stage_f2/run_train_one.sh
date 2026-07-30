#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 MODEL_NAME SEED GPU_IDS MASTER_PORT" >&2
  exit 2
fi
model_name="$1"
seed="$2"
gpu_ids="$3"
master_port="$4"
run_variant="${RUN_VARIANT:-formal}"
max_steps="${MAX_STEPS:-}"
case "${seed}" in 42|43|44) ;; *) echo "Unsupported seed: ${seed}" >&2; exit 2 ;; esac

ROOT="/home/zhaozhuofan/Hyper-LlaVA"
DEEPSPEED="/home/zhaozhuofan/miniconda3/envs/hyper/bin/deepspeed"
MODEL_PATH="/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
VISION_TOWER="/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
DATA_ROOT="/data/dataset/zhaozhuofan/Hyper-LLaVA/controlled_functional_v1"
IMAGE_FOLDER="/data/dataset/zhaozhuofan/Hyper-LLaVA"
if [[ "${run_variant}" == "formal" ]]; then
  CHECKPOINT_ROOT="/data/ckpt/zhaozhuofan/compose/residual_feasibility_v1/seed${seed}"
  RUN_ROOT="${ROOT}/experiments/runs/0730_residual_expert_feasibility/stage_f2/training/seed${seed}/${model_name}"
else
  CHECKPOINT_ROOT="/data/ckpt/zhaozhuofan/compose/residual_feasibility_v1/${run_variant}/seed${seed}"
  RUN_ROOT="${ROOT}/experiments/runs/0730_residual_expert_feasibility/stage_f2/${run_variant}/seed${seed}/${model_name}"
fi
OUTPUT_DIR="${CHECKPOINT_ROOT}/${model_name}"
mkdir -p "${RUN_ROOT}" "${CHECKPOINT_ROOT}"

rank=8
alpha=16
expected_parameters=19988480
active_ids=0
trainable_ids=0
expert_name="${model_name}"
origin=""
tags="controlled"
checkpoint_args=()
case "${model_name}" in
  expert_a)
    data_name=A_only; active_ids=0; trainable_ids=0; origin=Controlled/A_only; tags=controlled,shape-recognition ;;
  independent_b)
    data_name=B_only; active_ids=1; trainable_ids=1; origin=Controlled/B_only; tags=controlled,counting,independent ;;
  expert_c)
    data_name=C_only; active_ids=2; trainable_ids=2; origin=Controlled/C_only; tags=controlled,spatial ;;
  residual_b)
    data_name=A_plus_B; active_ids=0,1; trainable_ids=1; origin=Controlled/A_plus_B; tags=controlled,counting,residual
    checkpoint_args=(--compose_checkpoint "${CHECKPOINT_ROOT}/expert_a" --compose_existing_expert_origins 0=Controlled/A_only) ;;
  rank16_ab)
    data_name=A_plus_B; rank=16; alpha=32; expected_parameters=39976960; active_ids=0; trainable_ids=0; origin=Controlled/A_plus_B; tags=controlled,capacity-control ;;
  task_ab)
    data_name=A_plus_B; active_ids=0; trainable_ids=0; origin=Controlled/A_plus_B; tags=controlled,task-expert ;;
  upper_bc)
    data_name=B_plus_C; active_ids=0; trainable_ids=0; origin=Controlled/B_plus_C; tags=controlled,supervised-upper-bound ;;
  *) echo "Unknown model name: ${model_name}" >&2; exit 2 ;;
esac
DATA_PATH="${DATA_ROOT}/instructions/${data_name}/train.json"

IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
gpu_count="${#gpu_array[@]}"
if (( gpu_count == 0 || 8 % gpu_count != 0 )); then
  echo "GPU count must divide 8 to preserve global batch 64" >&2
  exit 2
fi
gradient_accumulation=$((8 / gpu_count))
for required in "${MODEL_PATH}" "${VISION_TOWER}" "${MODEL_PATH}/mm_projector.bin" "${DATA_PATH}" "${IMAGE_FOLDER}"; do
  [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 1; }
done
if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing non-empty output directory: ${OUTPUT_DIR}" >&2
  exit 1
fi
record_count="$(/home/zhaozhuofan/miniconda3/envs/hyper/bin/python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "${DATA_PATH}")"
[[ "${record_count}" == "1600" ]] || { echo "Unexpected record count: ${record_count}" >&2; exit 1; }

training_length_args=(--num_train_epochs 1 --save_strategy epoch)
if [[ -n "${max_steps}" ]]; then
  training_length_args=(--max_steps "${max_steps}" --save_strategy no)
fi

args=(
  --include "localhost:${gpu_ids}" --master_port "${master_port}"
  --module compose.train.train_compose
  --deepspeed ./scripts/zero2.json
  --model_name_or_path "${MODEL_PATH}"
  --pretrain_mm_mlp_adapter "${MODEL_PATH}/mm_projector.bin"
  --version v1 --data_path "${DATA_PATH}" --image_folder "${IMAGE_FOLDER}"
  --vision_tower "${VISION_TOWER}"
  --compose_rank "${rank}" --compose_alpha "${alpha}" --compose_dropout 0
  --compose_expert_ids "${active_ids}" --compose_trainable_expert_ids "${trainable_ids}"
  --compose_expert_name "${expert_name}" --compose_origin_task_id "${origin}"
  --compose_expert_tags "${tags}" --compose_gates "$(sed 's/[^,]*/1/g' <<< "${active_ids}")"
  --compose_gate_normalization none --expected_adapter_parameters "${expected_parameters}"
  "${checkpoint_args[@]}"
  --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2
  --mm_use_im_start_end False --mm_use_im_patch_token False --image_aspect_ratio pad
  --group_by_modality_length True --bf16 True --output_dir "${OUTPUT_DIR}"
  "${training_length_args[@]}" --per_device_train_batch_size 8 --per_device_eval_batch_size 16
  --gradient_accumulation_steps "${gradient_accumulation}" --evaluation_strategy no
  --learning_rate 2e-4 --weight_decay 0 --warmup_ratio 0.03
  --lr_scheduler_type cosine --logging_steps 1 --tf32 True --model_max_length 2048
  --gradient_checkpointing True --dataloader_num_workers 4 --report_to none --seed "${seed}"
)
{
  printf '%q' "${DEEPSPEED}"
  printf ' %q' "${args[@]}"
  printf '\n'
} > "${RUN_ROOT}/command.txt"
/home/zhaozhuofan/miniconda3/envs/hyper/bin/python - "${RUN_ROOT}/config.json" <<PY
import json, subprocess, sys
payload = {
    "model_name": "${model_name}", "seed": ${seed}, "gpu_ids": "${gpu_ids}", "run_variant": "${run_variant}",
    "data_path": "${DATA_PATH}", "records": ${record_count}, "output_dir": "${OUTPUT_DIR}",
    "rank": ${rank}, "alpha": ${alpha}, "active_expert_ids": "${active_ids}",
    "trainable_expert_ids": "${trainable_ids}", "expected_trainable_parameters": ${expected_parameters},
    "global_batch_size": 64, "max_steps": "${max_steps}", "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")
PY
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader,nounits -l 2 > "${RUN_ROOT}/gpu.csv" &
monitor_pid="$!"
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT
cd "${ROOT}"
set +e
/usr/bin/time -v "${DEEPSPEED}" "${args[@]}" > "${RUN_ROOT}/train.log" 2>&1
status="$?"
set -e
printf 'exit_code=%s\nfinished_utc=%s\ncheckpoint=%s\n' "${status}" "$(date -u +%FT%TZ)" "${OUTPUT_DIR}" > "${RUN_ROOT}/status.txt"
if [[ "${status}" -eq 0 ]]; then
  sha256sum "${OUTPUT_DIR}/compose_experts.bin" "${OUTPUT_DIR}/compose_experts.json" > "${RUN_ROOT}/checkpoint_SHA256SUMS"
fi
exit "${status}"
