#!/usr/bin/env bash
# Resume the FULL-DATA UCIT six-task formal feasibility pilot (seed 42)
# on physical GPUs 4,5,6,7 (spec §26). Tasks 0-2 are complete (single-GPU
# run, experts 0-5 committed); this script resumes at task 3 and runs
# tasks 3-5 plus the formal per-stage evals and the summary.
#
# Canonical 4-GPU invocation (spec §26):
#
#   torchrun --standalone --nproc_per_node=4 \
#     -m compose.experiments.task_run \
#     --config configs/compose_ucit.yaml \
#     --root experiments/runs/compose_ucit_formal_seed42 \
#     --first-task 0 --last-task 5 --seed 42
#
# task_run is the Rank-0 orchestrator: it runs task-level strictly
# sequentially and launches torchrun --nproc_per_node=4 subprocesses for
# the distributed stages (S1 features, S2 teacher NLL, S6 LoRA training,
# S9 RMS, S11 inference) — this script's --gpus 4,5,6,7 selects that
# 4-GPU execution plan, so wrapping task_run itself in torchrun would
# double-launch. The seed is taken from configs/compose_ucit.yaml
# (data.seed: 42).
#
# Prerequisite: the one-shot smoke must have passed
# (scripts/Compose/smoke_4gpu.sh).
#
set -euo pipefail

PY=/home/zhaozhuofan/miniconda3/envs/hyper/bin/python
REPO=/home/zhaozhuofan/Hyper-LlaVA
cd "$REPO"

GPUS="4,5,6,7"
CONFIG="$REPO/configs/compose_ucit.yaml"
ROOT="$REPO/experiments/runs/compose_ucit_formal_seed42"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

for g in 4 5 6 7; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  [ "$used" -lt 1000 ] || { echo "GPU $g busy (${used} MiB used)" >&2; exit 1; }
done

GIT_COMMIT=$(git rev-parse HEAD)

echo "resuming formal seed42 run: tasks 3..5 on GPUs $GPUS (commit $GIT_COMMIT)"

# ---- tasks 3..5 (resumes from the last completed stage marker) ----------
$PY -m compose.experiments.task_run \
  --config "$CONFIG" --root "$ROOT" \
  --first-task 3 --last-task 5 --gpus "$GPUS" \
  | tee "$LOG_DIR/resume_tasks3_5.log"

# ---- formal evals after each new stage (same 4 GPUs) ---------------------
for stage in 3 4 5; do
  $PY -m compose.eval.formal_ucit_eval \
    --root "$ROOT/task$stage" --stage-task "$stage" \
    --config "$CONFIG" --gpus "$GPUS" \
    | tee "$LOG_DIR/formal_eval_stage$stage.log"
done

# ---- final summary (continual matrix, MFN/MAA/MFT/BWT vs Hyper-LLaVA) ----
$PY -m compose.eval.formal_ucit_summary --root "$ROOT" \
  | tee "$LOG_DIR/formal_summary.log"

echo "formal seed42 run complete (commit $GIT_COMMIT)"
