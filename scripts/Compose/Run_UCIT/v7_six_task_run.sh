#!/usr/bin/env bash
# Formal Hyper-LLaVA V7 six-task lifecycle. This does not replace the V6.x
# six_task_run.sh and never derives validation data from official test files.
set -euo pipefail

REPO="${REPO:-/home/zhaozhuofan/Hyper-LlaVA}"
PY="${PY:-/home/zhaozhuofan/miniconda3/envs/hyper/bin/python}"
FORMAL_CONFIG="${FORMAL_CONFIG:-${REPO}/configs/v7_ucit_formal.yaml}"
RUN_ROOT="${RUN_ROOT:-/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_ucit_formal_seed42}"
GPUS="${GPUS:-0,1,2,3}"

export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
# The hyper env installs the `java` runtime used by pycocoevalcap's PTB
# tokenizer (VizWiz/Flickr30k caption scoring). Make this launcher
# self-sufficient even from a shell whose PATH has no conda env.
export PATH="$(dirname "${PY}"):${PATH}"

cd "${REPO}"
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
IFS=',' read -r -a gpu_ids <<<"${GPUS}"
for required in "${method_config}" "${model_path}" "${vision_tower}" "${projector_path}" "${image_folder}"; do
  [[ -e "${required}" ]] || { echo "Missing required formal path: ${required}" >&2; exit 1; }
done

previous=""
for row in "${rows[@]:1}"; do
  IFS=$'\t' read -r kind task_index task_name train_file val_file test_file validation_metric validation_annotation <<<"${row}"
  for required in "${train_file}" "${val_file}" "${test_file}" "${validation_annotation}"; do
    [[ -f "${required}" ]] || { echo "Missing declared Task${task_index} split/annotation: ${required}" >&2; exit 1; }
  done
  task_root="${RUN_ROOT}/task${task_index}"
  gpu="${gpu_ids[$((task_index % ${#gpu_ids[@]}))]}"
  command=(
    "${PY}" -m compose.experiments.v7_task_run
    --config "${method_config}" --root "${task_root}" --task-index "${task_index}"
    --task-name "${task_name}" --train-file "${train_file}" --val-file "${val_file}"
    --test-file "${test_file}" --validation-metric "${validation_metric}"
    --validation-annotation-file "${validation_annotation}"
    --model-path "${model_path}" --vision-tower "${vision_tower}"
    --projector-path "${projector_path}" --image-folder "${image_folder}"
    --device "cuda:${gpu}"
  )
  [[ -z "${previous}" ]] || command+=(--previous-checkpoint "${previous}")
  [[ ! -d "${task_root}" || -z "$(find "${task_root}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || command+=(--resume)
  "${command[@]}"
  previous="${task_root}/committed"
done
