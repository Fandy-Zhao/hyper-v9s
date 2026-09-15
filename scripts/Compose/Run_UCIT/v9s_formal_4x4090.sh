#!/usr/bin/env bash
# The formal V9-S launcher: four GPUs, one DDP world, six tasks, unattended.
#
#   ./v9s_formal_4x4090.sh check        run every pre-start check, print the contract, start nothing
#   ./v9s_formal_4x4090.sh sanity       one capped 4-GPU pass over tasks 0-1 into SANITY_ROOT
#   ./v9s_formal_4x4090.sh start        clean the sanity run, then launch the formal chain in tmux
#   ./v9s_formal_4x4090.sh foreground   the same chain, in this terminal
#   ./v9s_formal_4x4090.sh status       read the markers and say where the run is
#
# The shape this file exists to guarantee:
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3  +  torchrun --standalone --nproc_per_node=4
#   PER_DEVICE_BATCH=4  x  GRAD_ACCUM=2  x  WORLD_SIZE=4  =  GLOBAL_BATCH=32
#   QUERY_SOURCE=precomputed     QUERY_ENCODER_CALLS=0
#
# Every one of those numbers is checked rather than printed: the global batch is
# recomputed from its factors and the script exits non-zero if the product is not
# 32, so a mistyped override cannot silently change the optimisation problem the
# formal numbers describe.
#
# ``start`` is detached on purpose.  The run has to survive the shell that
# launched it -- an SSH session that ends must not take a 40-hour six-task chain
# with it -- so the chain is started in a tmux session and the launcher returns
# as soon as the session exists.  Nothing about the run's progress is decided
# here afterwards: the chain runs the completion gates itself.
set -euo pipefail

REPO="${REPO:-/root/autodl-tmp/Hyper-LlaVA}"
PY="${PY:-/root/miniconda3/envs/hyper/bin/python}"
UCIT="${UCIT:-/root/autodl-tmp/data/dataset/zhaozhuofan/UCIT}"
MODELS="${MODELS:-/root/autodl-tmp/data/ckpt/zhaozhuofan/models}"
CONFIG="${CONFIG:-$REPO/configs/v9s_main.yaml}"

GPUS="${GPUS:-0,1,2,3}"
EVAL_GPUS="${EVAL_GPUS:-$GPUS}"
WORLD_SIZE="${WORLD_SIZE:-4}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-32}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"
SAVE_STEPS="${SAVE_STEPS:-250}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
TASKS="${TASKS:-0,1,2,3,4,5}"
MIN_FREE_GB="${MIN_FREE_GB:-8}"
MIN_GPU_MEMORY_GIB="${MIN_GPU_MEMORY_GIB:-30}"

RUN_ROOT="${RUN_ROOT:-$REPO/experiments/runs/v9s_formal}"
SANITY_ROOT="${SANITY_ROOT:-$REPO/experiments/runs/v9s_formal_sanity}"
TMUX_SESSION="${TMUX_SESSION:-v9s_formal}"
SANITY_STEPS="${SANITY_STEPS:-12}"

# The fixed V7 query cache.  The manifest names a ``cache_root`` from the machine
# the cache was built on, so the runtime root is passed explicitly and the
# manifest keeps its provenance.  Neither file is ever re-encoded during a formal
# run: the queries are read, not derived.
QUERY_CACHE_MANIFEST="${QUERY_CACHE_MANIFEST:-/root/autodl-tmp/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_fixed_query_cache_gpu01_20260903/query_cache_manifest.json}"
# The manifest's own ``cache_root`` names /data/..., and /data is a bind mount
# of this same directory (the two share a device and an inode, checked) -- but
# it resolves intermittently on this host, so the run is pointed at the path
# underneath rather than at the mount that comes and goes over it.
QUERY_CACHE_ROOT="${QUERY_CACHE_ROOT:-/root/autodl-tmp/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_fixed_query_cache_gpu01_20260903/query_cache}"
# The two dataset roots are not the same directory: the declared v7 splits live
# under ``$UCIT/v7_train/<task>/train.json``, the evaluation instructions and
# their COCO annotations under ``$UCIT/instructions/<task>/``.  Eval cells come
# from the second, training data from the first.
SPLIT_ROOT="${SPLIT_ROOT:-$UCIT}"
INSTRUCTIONS_ROOT="${INSTRUCTIONS_ROOT:-$UCIT/instructions}"

log()  { printf '[v9s-formal] %s\n' "$*"; }
fail() { printf '[v9s-formal] FAIL: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# pre-start checks
# ---------------------------------------------------------------------------
check_gpus() {
  command -v nvidia-smi >/dev/null || fail "nvidia-smi is not on PATH"
  nvidia-smi >/dev/null || fail "nvidia-smi exited non-zero; the driver is not usable"
  IFS=',' read -r -a REQUESTED <<< "$GPUS"
  [ "${#REQUESTED[@]}" -eq "$WORLD_SIZE" ] || \
    fail "GPUS=$GPUS names ${#REQUESTED[@]} devices but WORLD_SIZE=$WORLD_SIZE"

  local visible
  visible=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  [ "$visible" -ge "$WORLD_SIZE" ] || \
    fail "only $visible GPUs are visible, the run needs $WORLD_SIZE"

  # Each selected device has to be big enough and idle enough.  Both are read
  # per device: a check that only counted GPUs would pass on a machine where the
  # device the run was pointed at is the one another job is holding.
  local index total used
  for index in "${REQUESTED[@]}"; do
    read -r total used < <(
      nvidia-smi --id="$index" --query-gpu=memory.total,memory.used \
        --format=csv,noheader,nounits | tr -d ','
    )
    local total_gib=$(( total / 1024 ))
    [ "$total_gib" -ge "$MIN_GPU_MEMORY_GIB" ] || \
      fail "GPU $index has ${total_gib}GiB, expected at least ${MIN_GPU_MEMORY_GIB}GiB"
    # 1 GiB of slack: a driver or a desktop session holds a little, a training
    # job holds tens of GiB.  Refusing on any nonzero use would make the check
    # impossible to satisfy on a machine that has ever drawn a frame.
    [ "$used" -lt 1024 ] || \
      fail "GPU $index already has ${used}MiB in use; another job is running on it"
    log "GPU $index: ${total_gib}GiB total, ${used}MiB used"
  done

  local compute_apps
  compute_apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader || true)
  [ -z "$compute_apps" ] || fail "compute processes are already attached:
$compute_apps"
}

check_runtime() {
  # The chain has to outlive the shell that starts it.  This is checked up
  # front because the way it fails otherwise is not an error -- the run simply
  # dies with the SSH session, hours in, with nothing to resume from.
  command -v tmux >/dev/null || fail "tmux is not installed; 'start' has nothing to detach into"
  "$PY" - <<'PY' || fail "the Python runtime cannot see CUDA/NCCL"
import sys, torch
checks = {
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "nccl": torch.distributed.is_nccl_available(),
}
print("[v9s-formal] " + " ".join(f"{k}={v}" for k, v in checks.items()), file=sys.stderr)
ok = checks["cuda_available"] and checks["nccl"] and checks["device_count"] >= 1
sys.exit(0 if ok else 1)
PY
}

check_contract() {
  local effective=$(( WORLD_SIZE * PER_DEVICE_BATCH * GRAD_ACCUM ))
  [ "$effective" -eq "$TARGET_GLOBAL_BATCH" ] || fail \
    "effective batch WORLD_SIZE x PER_DEVICE_BATCH x GRAD_ACCUM = $effective != $TARGET_GLOBAL_BATCH"
  [ -n "$QUERY_CACHE_MANIFEST" ] || fail "no --query-cache-manifest: a formal run reads precomputed queries"
  [ -f "$QUERY_CACHE_MANIFEST" ] || fail "query cache manifest not found: $QUERY_CACHE_MANIFEST"
  [ -d "$QUERY_CACHE_ROOT" ] || fail "query cache root not found: $QUERY_CACHE_ROOT"
  [ -f "$CONFIG" ] || fail "v9 config not found: $CONFIG"
  local task
  for task in ${TASKS//,/ }; do
    [ -f "$QUERY_CACHE_ROOT/task$task/train/queries.pt" ] || \
      fail "no train queries for task $task under $QUERY_CACHE_ROOT"
    [ -f "$QUERY_CACHE_ROOT/task$task/test/queries.pt" ] || \
      fail "no test queries for task $task under $QUERY_CACHE_ROOT"
  done
  local free_gb
  free_gb=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
  [ "$free_gb" -ge "$MIN_FREE_GB" ] || \
    fail "only ${free_gb}G free on $REPO; a task that runs out of disk loses the whole task"
}

check_clean_tree() {
  cd "$REPO"
  git rev-parse --git-dir >/dev/null 2>&1 || fail "$REPO is not a git repository"
  if ! git diff --quiet || ! git diff --cached --quiet; then
    git status --short >&2
    fail "the working tree or index has uncommitted changes; commit them first -- the run records its own SHA and a tree that can change under it makes that record a guess"
  fi
  GIT_SHA=$(git rev-parse HEAD)
  log "git SHA $GIT_SHA"
}

# ---------------------------------------------------------------------------
# the contract the run is bound to
# ---------------------------------------------------------------------------
print_contract() {
  cat <<EOF
[v9s-formal] ================= launch contract =================
[v9s-formal]   WORLD_SIZE=$WORLD_SIZE
[v9s-formal]   PER_DEVICE_BATCH=$PER_DEVICE_BATCH
[v9s-formal]   GRAD_ACCUM=$GRAD_ACCUM
[v9s-formal]   GLOBAL_BATCH=$(( WORLD_SIZE * PER_DEVICE_BATCH * GRAD_ACCUM ))
[v9s-formal]   QUERY_SOURCE=precomputed
[v9s-formal]   QUERY_ENCODER_CALLS=0
[v9s-formal]   CUDA_VISIBLE_DEVICES=$GPUS
[v9s-formal]   launcher=torchrun --standalone --nproc_per_node=$WORLD_SIZE
[v9s-formal]   config=$CONFIG
[v9s-formal]   query_manifest=$QUERY_CACHE_MANIFEST
[v9s-formal]   query_root=$QUERY_CACHE_ROOT
[v9s-formal]   run_root=$RUN_ROOT
[v9s-formal] =======================================================
EOF
}

# The DataLoader question, answered from the profiler rather than from taste.
# The rule the efficiency brief sets is that below 1% of the step spent waiting
# on the loader, the loader is not the bottleneck and a prefetch_factor or
# worker-count change would be unmeasurable.  This prints the measurement when
# there is one to print and does not change anything either way.
print_data_wait_verdict() {
  local source="${1:-}"
  [ -n "$source" ] && [ -d "$source" ] || return 0
  "$PY" -m compose.v9.data_wait "$source" 2>/dev/null | sed 's/^/[v9s-formal] /' || true
}

# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
cmd_check() {
  check_gpus
  check_runtime
  check_contract
  check_clean_tree
  print_contract
  log "all pre-start checks passed"
}

chain_arguments() {
  local run_root="$1"; shift
  local extra=("$@")
  printf '%s\n' \
    --repo "$REPO" --python "$PY" --run-root "$run_root" --config "$CONFIG" \
    --instructions-root "$INSTRUCTIONS_ROOT" --split-root "$SPLIT_ROOT" \
    --model-path "$MODELS/llava-v1.5-7b" \
    --vision-tower "$MODELS/clip-vit-large-patch14-336" \
    --projector-path "$MODELS/llava-v1.5-7b/mm_projector.bin" \
    --image-folder "$UCIT/datasets" --query-encoder "$MODELS/clip-vit-large-patch14-336" \
    --query-cache-manifest "$QUERY_CACHE_MANIFEST" --query-cache-root "$QUERY_CACHE_ROOT" \
    --gpus "$GPUS" --eval-gpus "$EVAL_GPUS" --world-size "$WORLD_SIZE" \
    --per-device-batch "$PER_DEVICE_BATCH" --grad-accum "$GRAD_ACCUM" \
    --dataloader-num-workers "$DATALOADER_WORKERS" \
    --logging-steps "$LOGGING_STEPS" --save-steps "$SAVE_STEPS" \
    "${extra[@]}"
}

cmd_sanity() {
  check_gpus
  check_runtime
  check_contract
  check_clean_tree
  print_contract
  [ -n "$SANITY_ROOT" ] || fail "SANITY_ROOT is unset"
  case "$SANITY_ROOT" in "$RUN_ROOT"*) fail "SANITY_ROOT must not be inside RUN_ROOT ($RUN_ROOT)";; esac
  rm -rf "$SANITY_ROOT"
  mkdir -p "$SANITY_ROOT"
  log "sanity run -> $SANITY_ROOT (capped at $SANITY_STEPS optimizer steps, tasks 0-1)"
  # A capped run cannot satisfy the full-coverage gate, so it is told what it is
  # and keeps its own markers; its product is the 4-GPU evidence, not a matrix.
  # The capped run has to reach a checkpoint inside its cap, or the reload and
  # resume paths it exists to exercise are never touched.  ``save_steps`` is
  # appended after ``chain_arguments`` has already supplied the formal value, so
  # argparse takes this one; the formal run is unaffected.
  mapfile -t ARGS < <(chain_arguments "$SANITY_ROOT" \
    --sanity --max-steps "$SANITY_STEPS" --to-task 1 --eval-through 0 \
    --save-steps 4 --logging-steps 2 --from-task 0 --profile-training)
  cd "$REPO"
  "$PY" -m compose.experiments.v9_chain "${ARGS[@]}" 2>&1 | tee "$SANITY_ROOT/sanity.log"
  local status=${PIPESTATUS[0]}
  [ "$status" -eq 0 ] || fail "the 4-GPU sanity run failed (exit $status); see $SANITY_ROOT/sanity.log"
  log "sanity COMPLETE-OR-REPORTED; evidence in $SANITY_ROOT"
}

cmd_start() {
  cmd_check
  # Spec §25: the sanity run is cleaned before the formal one starts.  Its
  # checkpoints come from a capped run against a different root and must never be
  # reachable from the formal chain, which would treat them as work already done.
  if [ -d "$SANITY_ROOT" ]; then
    log "removing the sanity run at $SANITY_ROOT (its checkpoints are not formal work)"
    rm -rf "$SANITY_ROOT"
  fi
  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    fail "tmux session '$TMUX_SESSION' already exists; attach to it, or kill it first"
  fi
  mkdir -p "$RUN_ROOT"
  printf '%s\n' "$GIT_SHA" > "$RUN_ROOT/formal_git_sha.txt"

  mapfile -t ARGS < <(chain_arguments "$RUN_ROOT" \
    --from-task 0 --to-task 5 --profile-training)
  local command
  printf -v command '%q ' "$PY" -m compose.experiments.v9_chain "${ARGS[@]}"
  # The chain appends to one log per stage; a run restarted by hand must not
  # overwrite the record of what the first attempt did.
  tmux new-session -d -s "$TMUX_SESSION" \
    "cd $(printf '%q' "$REPO") && $command 2>&1 | tee -a $(printf '%q' "$RUN_ROOT/chain.log")"
  sleep 3
  tmux has-session -t "$TMUX_SESSION" || fail "the tmux session died immediately; check $RUN_ROOT/chain.log"
  log "formal chain running in tmux '$TMUX_SESSION'"
  log "  attach:  tmux attach -t $TMUX_SESSION"
  log "  log:     tail -f $RUN_ROOT/chain.log"
  log "  status:  $0 status"
}

cmd_foreground() {
  cmd_check
  mapfile -t ARGS < <(chain_arguments "$RUN_ROOT" --from-task 0 --to-task 5 --profile-training)
  cd "$REPO"
  exec "$PY" -m compose.experiments.v9_chain "${ARGS[@]}"
}

cmd_status() {
  local root="$RUN_ROOT" marker
  for marker in FORMAL_COMPLETE FORMAL_FAILED FORMAL_RUNNING; do
    if [ -f "$root/$marker" ]; then
      printf '[v9s-formal] %s  (%s)\n' "$marker" "$(cat "$root/$marker")"
    fi
  done
  [ -d "$root/task0" ] || { log "no task directories yet under $root"; return 0; }
  "$PY" - "$root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for task in range(6):
    marker = root / "task{}".format(task) / "data" / "task_complete.json"
    if marker.is_file():
        payload = json.loads(marker.read_text())
        print("[v9s-formal] task {}: complete at {} (eval {})".format(
            task, payload.get("timestamp"), "yes" if payload.get("evaluation") else "no"))
    else:
        print("[v9s-formal] task {}: not complete".format(task))
PY
}

[ $# -ge 1 ] || { sed -n '2,12p' "$0"; exit 2; }
case "$1" in
  check)      cmd_check ;;
  sanity)     cmd_sanity ;;
  start)      cmd_start ;;
  foreground) cmd_foreground ;;
  status)     cmd_status ;;
  *)          fail "unknown command '$1' (check|sanity|start|foreground|status)" ;;
esac
