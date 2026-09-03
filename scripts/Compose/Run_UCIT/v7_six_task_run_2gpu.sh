#!/usr/bin/env bash
# Formal V7: sequential Task0..Task5; every S3 stage uses GPU0+GPU1 DDP.
set -euo pipefail

REPO="${REPO:-/home/zhaozhuofan/Hyper-LlaVA}"
PY="${PY:-/home/zhaozhuofan/miniconda3/envs/hyper/bin/python}"
FORMAL_CONFIG="${FORMAL_CONFIG:-${REPO}/configs/v7_ucit_formal.yaml}"
METHOD_CONFIG="${METHOD_CONFIG:-${REPO}/configs/v7_global_coevolution.yaml}"
RUN_ROOT="${RUN_ROOT:-/data/ckpt/zhaozhuofan/Hyper-LlaVA-runs/v7_ucit_formal_2gpu_b2a16_w4_seed42_20260903}"
TRAINING_GPUS="${TRAINING_GPUS:-0,1}"
TRAINING_BATCH_SIZE="${TRAINING_BATCH_SIZE:-2}"
TRAINING_GRAD_ACCUM="${TRAINING_GRAD_ACCUM:-16}"
TRAINING_WORKERS="${TRAINING_WORKERS:-4}"

export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PATH="$(dirname "${PY}"):${PATH}"
cd "${REPO}"

"${PY}" - "${TRAINING_GPUS}" <<'PY'
import subprocess, sys
requested = [int(value) for value in sys.argv[1].split(",")]
if requested != [0, 1]:
    raise SystemExit("formal two-GPU launcher is pinned to GPU0,1")
rows = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
    text=True,
)
state = {}
for row in rows.splitlines():
    index, memory, utilization = [int(value.strip()) for value in row.split(",")]
    state[index] = (memory, utilization)
busy = {index: state[index] for index in requested if state[index][0] >= 1024 or state[index][1] != 0}
if busy:
    raise SystemExit("GPU0-1 availability gate failed: {}".format(busy))
print("GPU0-1 availability gate passed: {}".format({i: state[i] for i in requested}))
PY

mapfile -t rows < <("${PY}" - "${FORMAL_CONFIG}" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
d = cfg["data"]
print("META\t" + "\t".join([cfg["method_config"], d["model_path"], d["vision_tower"], d["projector_path"], d["image_folder"]]))
for task in cfg["tasks"]:
    print("TASK\t" + "\t".join(str(task[key]) for key in (
        "task_index", "name", "train_file", "validation_file", "test_file",
        "validation_metric", "validation_annotation_file"
    )))
PY
)

IFS=$'\t' read -r kind method_config model_path vision_tower projector_path image_folder <<<"${rows[0]}"
for required in "${method_config}" "${model_path}" "${vision_tower}" "${projector_path}" "${image_folder}"; do
  [[ -e "${required}" ]] || { echo "Missing required formal path: ${required}" >&2; exit 1; }
done

previous=""
for row in "${rows[@]:1}"; do
  IFS=$'\t' read -r kind task_index task_name train_file val_file test_file validation_metric validation_annotation <<<"${row}"
  task_root="${RUN_ROOT}/task${task_index}"
  command=("${PY}" -m compose.experiments.v7_task_run
    --config "${method_config}" --root "${task_root}" --task-index "${task_index}" --task-name "${task_name}"
    --train-file "${train_file}" --val-file "${val_file}" --test-file "${test_file}"
    --validation-metric "${validation_metric}" --validation-annotation-file "${validation_annotation}"
    --model-path "${model_path}" --vision-tower "${vision_tower}" --projector-path "${projector_path}"
    --image-folder "${image_folder}" --device cuda:0 --training-world-size 2 --training-gpus "${TRAINING_GPUS}"
    --training-per-device-batch-size "${TRAINING_BATCH_SIZE}" --training-gradient-accumulation-steps "${TRAINING_GRAD_ACCUM}"
    --training-dataloader-num-workers "${TRAINING_WORKERS}" --distributed-backend nccl --skip-eval)
  [[ -z "${previous}" ]] || command+=(--previous-checkpoint "${previous}")
  [[ ! -d "${task_root}" || -z "$(find "${task_root}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || command+=(--resume)
  "${command[@]}"
  previous="${task_root}/committed"
done

"${PY}" -m compose.eval.v7_formal_ucit_eval --root "${RUN_ROOT}" --formal-config "${FORMAL_CONFIG}" \
  --method-config "${METHOD_CONFIG}" --gpus "${TRAINING_GPUS}" --python "${PY}"
"${PY}" -m compose.eval.formal_ucit_summary --root "${RUN_ROOT}"
