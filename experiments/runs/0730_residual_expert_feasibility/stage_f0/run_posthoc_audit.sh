#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zhaozhuofan/Hyper-LlaVA"
PYTHON="/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"
RUN_DIR="${ROOT}/experiments/runs/0730_residual_expert_feasibility/stage_f0"
EVAL_ROOT="/data/ckpt/zhaozhuofan/compose/eval"
ORACLE_ROOT="/data/ckpt/zhaozhuofan/compose/oracle/task1_task4_experts01_rank8_seed42"
SCORE_ROOT="/data/ckpt/zhaozhuofan/compose/scores"

mkdir -p "${RUN_DIR}"
cd "${ROOT}"

COMMAND_FILE="${RUN_DIR}/command.txt"
LOG_FILE="${RUN_DIR}/posthoc_audit.log"
STATUS_FILE="${RUN_DIR}/posthoc_audit.status"
META_FILE="${RUN_DIR}/posthoc_audit.meta.json"

python_args=(
  -m compose.oracle.posthoc_audit
  --oracle-cache "${ORACLE_ROOT}/oracle_cache.jsonl"
  --rank16-scores "${SCORE_ROOT}/task1_task4_rank16_seed42/scores.jsonl"
  --direct-sum-scores "${SCORE_ROOT}/task1_task4_pair_direct_sum_rank8_seed42/scores.jsonl"
  --annotation-file /data/ckpt/zhaozhuofan/compose/inputs/task1_task4_test_6000.json
  --base-predictions "${EVAL_ROOT}/task1_task4_selections_rank8_seed42/base/predictions.jsonl"
  --single0-predictions "${EVAL_ROOT}/task1_task4_selections_rank8_seed42/single0/predictions.jsonl"
  --single1-predictions "${EVAL_ROOT}/task1_task4_selections_rank8_seed42/single1/predictions.jsonl"
  --pair-l2-predictions "${EVAL_ROOT}/task1_task4_selections_rank8_seed42/pair_l2/predictions.jsonl"
  --pair-direct-sum-predictions "${EVAL_ROOT}/task1_task4_pair_direct_sum_rank8_seed42/predictions.jsonl"
  --rank16-predictions "${EVAL_ROOT}/task1_task4_rank16_seed42/predictions.jsonl"
  --output-json "${RUN_DIR}/posthoc_oracle.json"
  --output-csv "${RUN_DIR}/per_sample_audit.csv"
  --report-file "${ROOT}/docs/reports/0730_stage_f0_posthoc_oracle.md"
)

{
  printf '%q' "${PYTHON}"
  printf ' %q' "${python_args[@]}"
  printf '\n'
} > "${COMMAND_FILE}"
"${PYTHON}" - "${META_FILE}" <<'PY'
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

path = sys.argv[1]
payload = {
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "python": sys.executable,
    "python_version": platform.python_version(),
    "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
    "seed": None,
    "gpu_used": False,
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

set +e
"${PYTHON}" "${python_args[@]}" > "${LOG_FILE}" 2>&1
status="$?"
set -e
printf 'exit_code=%s\nfinished_utc=%s\n' "${status}" "$(date -u +%FT%TZ)" > "${STATUS_FILE}"
if [[ "${status}" -eq 0 ]]; then
  sha256sum "${RUN_DIR}/posthoc_oracle.json" "${RUN_DIR}/per_sample_audit.csv" \
    "${ROOT}/docs/reports/0730_stage_f0_posthoc_oracle.md" > "${RUN_DIR}/SHA256SUMS"
fi
exit "${status}"
