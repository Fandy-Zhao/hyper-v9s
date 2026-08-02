#!/usr/bin/env bash
set -euo pipefail
cd /home/zhaozhuofan/Hyper-LlaVA
ROOT=experiments/runs/v6_ucit_staged/stage04_oracle_teacher
PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
exec > >(tee "$ROOT/logs/l0_final.log") 2>&1

echo "[L0] Python compile"
"$PY" -m compileall -q compose/teacher tests/compose/test_oracle_teacher_stage04.py "$ROOT/scripts"

echo "[L0] Shell syntax"
for script in "$ROOT"/scripts/*.sh; do bash -n "$script"; done

echo "[L0] Hyper import isolation"
if grep -R -nE '(^|[[:space:]])(from|import)[[:space:]]+compose' Hyper --include='*.py'; then
  echo "Hyper must not import compose" >&2
  exit 1
fi

echo "[L0] tests/compose full discovery"
"$PY" -m unittest discover -s tests/compose -p 'test_*.py'

echo "[L0] git diff --check"
git diff --check
echo "[L0] PASSED"
