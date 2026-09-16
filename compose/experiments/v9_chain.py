"""The unattended V9-S six-task chain: train -> evaluate -> hand off -> next.

This module exists so that the formal run does not need anyone watching it.  It
is the difference between a *script that starts a task* and a *pipeline that
finishes a sequence*: every decision the run would otherwise need a human to
make -- is this task done, did training finish, did the handoff land, is the
row complete, did it OOM, may I take the next step -- is taken here, written
down, and enforced.

The order per task is fixed by the spec and is not configurable:

    train -> in-process audits -> task-end commit -> A[t][t]
          -> verify the handoff Task{t+1} will load -> next task

The evaluation that admits the next task is the task's *own* test set -- the
diagonal cell ``A[t][t]``, measured with the pool that task just committed.
The other cells of the row are cross-task measurements (the model after task
``t`` on an earlier task's test set) and no later task depends on any of them,
so building them here would spend the next task's training time on numbers that
can be taken just as well once the last task is trained.  They are filled by
:func:`final_sweep` at the end, which never regenerates a cell that already
exists.

What it refuses to do is as important as what it does.  It will not advance on
a task whose training, commit or diagonal cell is missing, it will not resume
across a batch-shape change (that would re-enter an optimizer state built for
different gradients), it will not start a second training against a task that
already has one running, and it will not print "sequence complete" unless every
gate in :mod:`compose.v9.closure` passes for every task over the full 21-cell
matrix.  A failure leaves a ``FORMAL_FAILED`` marker and a
``failure_report.json`` and exits non-zero rather than looking finished.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from compose.v9.closure import TASK_NAMES, go_no_go, task_completion
from compose.v9.heartbeat import beat

MARKER_RUNNING = "FORMAL_RUNNING"
MARKER_COMPLETE = "FORMAL_COMPLETE"
MARKER_FAILED = "FORMAL_FAILED"

#: Launches of one task's training before the chain stops trying: the first
#: attempt plus two retries, which is what the recovery policy permits.  The OOM
#: fallback changes the batch shape and is a separate, once-only decision, so it
#: does not spend this budget.
MAX_TRAINING_ATTEMPTS = 3

#: Substrings that mean the process died because the batch did not fit.  The
#: fallback below is only ever taken for one of these: any other non-zero exit
#: is a bug to be reported, not a memory problem to be tuned around.
OOM_MARKERS = (
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
    "OutOfMemoryError",
    "CUDA error: out of memory",
)


class V9ChainError(RuntimeError):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> Optional[str]:
    """The file's digest, or ``None`` when there is no file to digest.

    ``None`` rather than a digest of nothing: a cell measured against a pool
    that no longer exists must not read as a cell measured against a pool that
    matches.
    """
    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class task_lock:
    """``RUN_ROOT/locks/taskN.lock``, held while one task is being run.

    An ``flock`` rather than a PID file, because a PID file cannot distinguish
    "the holder is running" from "the holder was killed and the file was not
    cleaned up" -- and the cost of that mistake here is a second training run
    writing into the same directory as the first.  The kernel drops an flock
    when the holder dies, so the question the lock answers is the question that
    matters.  It is *advisory*: it stops a second copy of this chain, it does
    not stop a process that never asks.
    """

    def __init__(self, run_root: Path, task: int) -> None:
        self.task = int(task)
        self.path = Path(run_root) / "locks" / "task{}.lock".format(self.task)
        self.handle = None

    def __enter__(self) -> "task_lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            self.handle = None
            raise V9ChainError(
                "another process holds {}; two supervisors must not run the "
                "same task at once".format(self.path)
            )
        return self

    def __exit__(self, *exception: Any) -> None:
        if self.handle is not None:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            finally:
                self.handle.close()
                self.handle = None


def _marker(root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(_now() + "\n", encoding="utf-8")


def _clear_markers(root: Path) -> None:
    """Remove every run marker, sanity and formal, before a run writes its own.

    Both families are cleared so a directory can never hold ``SANITY_COMPLETE``
    beside ``FORMAL_FAILED`` -- the reader of a run root would otherwise have to
    know which marker to believe.
    """
    for prefix in ("FORMAL", "SANITY"):
        for name in ("RUNNING", "COMPLETE", "FAILED"):
            target = root / "{}_{}".format(prefix, name)
            if target.exists():
                target.unlink()


def _tail(path: Path, lines: int = 200) -> List[str]:
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return content[-lines:]


def _log_looks_like_oom(text: str) -> bool:
    return any(marker in text for marker in OOM_MARKERS)


def _git_sha(repo: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty(repo: Path) -> bool:
    """True if the working tree or index has uncommitted changes.

    The formal run is bound to a commit: if the checkout can change under it,
    the run's own record of what produced it is a guess.  The launcher refuses
    to start on a dirty tree for that reason, and this repeats the check so the
    chain cannot be driven directly without it.
    """
    try:
        for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
            result = subprocess.run(command, cwd=str(repo), stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            if result.returncode != 0:
                return True
    except OSError:
        return True
    return False


# ----------------------------------------------------------------------
# performance warnings (spec §30)
# ----------------------------------------------------------------------
def performance_report(task_root: Path, first_steps: int = 20) -> Dict[str, Any]:
    """Summarise the first ``first_steps`` optimizer steps and flag anomalies.

    Warnings only.  The spec is explicit that nothing here may stop the run:
    every quantity below is a *performance* signal, and a slow step is not a
    wrong one.  Correctness failures stop the run; throughput failures are
    recorded for the report.
    """
    metrics_dir = Path(task_root) / "metrics"
    rows: List[Dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("task*_train_steps*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda row: int(row.get("step", 0)))
    window = rows[: int(first_steps)] if first_steps > 0 else rows
    warnings: List[str] = []
    summary: Dict[str, Any] = {"steps_observed": len(rows), "window": len(window)}
    if not window:
        summary["warnings"] = ["no step metrics were written"]
        summary["warning"] = True
        return summary

    def total(key: str) -> float:
        return sum(float(row.get(key) or 0.0) for row in window)

    step_seconds = total("training_step_sec")
    wait_seconds = total("inter_step_wait_sec")
    summary["mean_step_sec"] = round(step_seconds / len(window), 4)
    summary["inter_step_wait_fraction"] = round(
        wait_seconds / max(step_seconds + wait_seconds, 1e-9), 6
    )
    peak = max((float(row.get("peak_memory_bytes") or 0.0) for row in window), default=0.0)
    summary["peak_allocated_GiB"] = round(peak / (1024 ** 3), 4)
    ratio = [float(row.get("backbone_forwards_per_micro_step") or 0.0) for row in window]
    summary["backbone_forwards_per_micro_step_max"] = round(max(ratio, default=0.0), 6)
    hook_ratio = [
        float(row.get("model_forwards_per_micro_step") or 0.0)
        for row in window
        if row.get("model_forwards_per_micro_step") is not None
    ]
    if hook_ratio:
        summary["model_forwards_per_micro_step_max"] = round(max(hook_ratio), 6)
    wide = [float(row.get("wide_model_forwards_per_micro_step") or 0.0) for row in window]
    if any(value > 0.0 for value in wide):
        summary["wide_model_forwards_per_micro_step_max"] = round(max(wide), 6)

    if summary["inter_step_wait_fraction"] > 0.30:
        warnings.append(
            "data wait {:.1%} of the step exceeds the 30% threshold".format(
                summary["inter_step_wait_fraction"]
            )
        )
    if summary["backbone_forwards_per_micro_step_max"] > 1.01:
        warnings.append(
            "backbone_forwards_per_micro_step {} > 1: something is traversing the "
            "backbone more than once per micro-batch".format(
                summary["backbone_forwards_per_micro_step_max"]
            )
        )
    if hook_ratio and max(hook_ratio) > 1.01:
        warnings.append(
            "the model's own forward hook counted {} traversals per micro-step".format(
                max(hook_ratio)
            )
        )
    if any(value > 1.01 for value in wide if value > 0.0):
        warnings.append(
            "wide retrieval increased backbone traversals: wide="
            "{} narrow={}".format(
                summary.get("wide_model_forwards_per_micro_step_max"),
                summary.get("model_forwards_per_micro_step_max"),
            )
        )
    gpu_util = summary.get("gpu_utilization_percent")
    if isinstance(gpu_util, (int, float)) and gpu_util < 20.0:
        warnings.append("GPU utilization {:.1f}% is below 20%".format(float(gpu_util)))
    summary["warnings"] = warnings
    summary["warning"] = bool(warnings)
    return summary


# ----------------------------------------------------------------------
# one task
# ----------------------------------------------------------------------
def training_command(
    args,
    task: int,
    per_device_batch: int,
    grad_accum: int,
    resume: bool,
    profile_training: bool,
    require_full_coverage: bool = True,
) -> List[str]:
    task_root = Path(args.run_root) / "task{}".format(task)
    name = TASK_NAMES[task]
    instructions = Path(args.instructions_root)
    # The declared splits and the evaluation instructions sit in different trees
    # of the same dataset: ``v7_train/<task>/train.json`` under one root,
    # ``<task>/test_3000.json`` under another.  They are resolved separately
    # rather than assumed equal -- getting this wrong trains on the instructions
    # directory and fails with a missing file rather than a wrong one, but only
    # because the names differ, which is luck rather than a check.
    split_root = Path(args.split_root) if args.split_root else instructions
    if not split_root.is_dir():
        split_root = instructions
    train_file = split_root / "v7_train" / name / "train.json"
    val_file = split_root / "v7_validation" / name / "validation.json"
    command = [
        args.python, "-m", "compose.experiments.v9_task_run",
        "--config", args.config,
        "--root", str(task_root),
        "--task-index", str(task),
        "--train-file", str(train_file),
        "--val-file", str(val_file),
        "--model-path", args.model_path,
        "--vision-tower", args.vision_tower,
        "--projector-path", args.projector_path,
        "--image-folder", args.image_folder,
        "--query-encoder", args.query_encoder,
        "--device", "cuda:0",
        "--training-world-size", str(args.world_size),
        "--training-launcher", "torchrun",
        "--training-per-device-batch-size", str(per_device_batch),
        "--training-gradient-accumulation-steps", str(grad_accum),
        "--num-train-epochs", "1",
        "--learning-rate", str(args.learning_rate),
        "--v9-key-learning-rate", str(args.key_learning_rate),
        "--logging-steps", str(args.logging_steps),
        "--save-strategy", "steps",
        "--save-steps", str(args.save_steps),
        "--save-total-limit", "2",
        "--dataloader-num-workers", str(args.dataloader_num_workers),
        "--calibrate",
    ]
    if require_full_coverage:
        command += ["--require-full-coverage"]
    if int(getattr(args, "max_steps", 0)) > 0:
        # Only the sanity run caps this.  The cap is an optimizer-step count,
        # not a smaller split: the task still streams its whole train file, so
        # the code path under test is the formal one and only the stopping
        # point differs.
        command += ["--max-steps", str(int(args.max_steps))]
    if args.query_cache_manifest:
        command += ["--query-cache-manifest", args.query_cache_manifest]
        if args.query_cache_root:
            command += ["--query-cache-root", args.query_cache_root]
    if task > 0:
        previous = Path(args.run_root) / "task{}".format(task - 1) / "training" / "task{}".format(task - 1)
        if not (previous / "compose_experts.bin").is_file():
            raise V9ChainError(
                "task {} needs the committed task {} checkpoint at {}".format(task, task - 1, previous)
            )
        command += ["--previous-checkpoint", str(previous)]
    if resume:
        command += ["--resume"]
    if profile_training:
        command += ["--profile-training"]
    return command


def run_training(args, command: List[str], task: int, env: Dict[str, str]) -> Dict[str, Any]:
    log_path = Path(args.run_root) / "logs" / "task{}_train.log".format(task)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n===== {} task {} =====\n{}\n".format(_now(), task, " ".join(command)))
        log.flush()
        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return {
        "returncode": int(completed.returncode),
        "seconds": round(time.time() - started, 3),
        "log": str(log_path),
        "oom": _log_looks_like_oom(text),
    }


def full_data_contract(args, task: int) -> Dict[str, Any]:
    """The tail predicate, asserted where the task ends rather than where the next begins.

    ``task_completion`` already reads these numbers, but it is consulted when the
    *next* task starts -- so without this the chain would train task t+1 for
    hours and only then discover that task t dropped its last accumulation
    window.  The step count is recomputed from the artefact's own declared split
    and global batch, so agreement here means agreement with the contract and not
    with the run's arithmetic: both HF and the old V9 scheduler used the same
    floor division, and comparing them to each other is exactly the check that
    would have missed this.
    """
    training = Path(args.run_root) / "task{}".format(task) / "training" / "task{}".format(task)
    path = training / "v9_full_data_coverage.json"
    blank = {
        "declared_unique_samples": 0, "padded_epoch_samples": 0,
        "padding_duplicate_count": 0, "expected_optimizer_steps": 0,
        "actual_optimizer_steps": 0, "forward_unique_coverage": 0.0,
        "optimizer_unique_coverage": 0.0, "ok": False,
    }
    if not path.is_file():
        blank["failed"] = ["no coverage artefact at {}".format(path)]
        return blank
    coverage = _read_json(path)
    declared = int(coverage.get("declared_unique_samples", 0) or coverage.get("num_train_samples", 0))
    global_batch = int(coverage.get("global_batch", 0))
    actual = int(coverage.get("optimizer_steps", 0))
    forward = float(coverage.get("train_sample_coverage", 0.0))
    optimizer = float(coverage.get("optimizer_coverage", 0.0))
    required = int(math.ceil(declared / global_batch)) if declared and global_batch else 0
    failed = []
    if required < 1:
        failed.append("declared {} / global batch {} does not define a step".format(declared, global_batch))
    if actual != required:
        failed.append("actual_optimizer_steps {} != ceil({} / {}) = {}".format(actual, declared, global_batch, required))
    if int(coverage.get("expected_optimizer_steps", 0)) != required:
        failed.append("artefact expected_optimizer_steps {} != {}".format(coverage.get("expected_optimizer_steps"), required))
    if forward < 1.0:
        failed.append("forward_unique_coverage {:.6f}".format(forward))
    if optimizer < 1.0:
        failed.append("optimizer_unique_coverage {:.6f}".format(optimizer))
    if int(coverage.get("unclosed_window_sample_count", 0)) != 0:
        failed.append("unclosed_window_sample_count {}".format(coverage.get("unclosed_window_sample_count")))
    return {
        "declared_unique_samples": declared,
        "padded_epoch_samples": int(coverage.get("padded_epoch_samples", 0)),
        "padding_duplicate_count": int(coverage.get("padding_duplicate_count", 0)),
        "expected_optimizer_steps": required,
        "actual_optimizer_steps": actual,
        "forward_unique_coverage": forward,
        "optimizer_unique_coverage": optimizer,
        "ok": not failed,
        "failed": failed,
    }


def handoff_checks(args, task: int) -> Dict[str, Any]:
    """Everything Task{t+1} will rely on, asserted before it is started.

    A task that trained and committed but cannot be handed off would otherwise
    surface the failure hours later, inside the *next* task's startup, where the
    cause is no longer visible.  These are the same artefacts the next task
    loads, checked by the task that produced them.
    """
    task_root = Path(args.run_root) / "task{}".format(task)
    training = task_root / "training" / "task{}".format(task)
    checks: Dict[str, Any] = {}
    checks["trained_pool"] = (training / "v9_key_pool.pt").is_file()
    checks["committed_state"] = (task_root / "state" / "key_pool_task{}.pt".format(task)).is_file()
    checks["experts_checkpoint"] = (training / "compose_experts.bin").is_file()
    checks["audit"] = (task_root / "data" / "audit_task{}.json".format(task)).is_file()
    checks["calibration"] = (training / "v9_contribution_calibration.json").is_file()
    if task > 0:
        # What the *next* task's loader will read: the committed pool one level
        # above the checkpoint directory, and the frozen experts beside it.
        previous_ok = checks["committed_state"] and checks["experts_checkpoint"]
        checks["previous_state_resolvable"] = previous_ok
    checks["all_ok"] = all(
        value for key, value in checks.items() if key != "all_ok"
    )
    return checks


def evaluate_task(args, stage: int, cells: Sequence[int]) -> Dict[str, Any]:
    """Measure stage ``stage`` on the test sets named by ``cells``.

    ``stage`` is the model state -- the checkpoint and the pool committed at the
    end of that task -- and ``cells`` are the tasks whose test sets it is
    measured on.  ``A[stage][stage]`` is the task's own test set; every smaller
    index is a cross-task cell.  Both are produced by this one function, with
    the same committed pool and the same fixed queries, so a cross-task cell
    measured in the sweep is the same measurement it would have been had it
    been taken beside the diagonal.
    """
    cells = sorted(int(cell) for cell in cells)
    eval_root = Path(args.run_root) / "evaluation_matrix"
    task_root = Path(args.run_root) / "task{}".format(stage)
    cells_path = eval_root / "cells_t{}_{}.json".format(stage, "_".join(str(c) for c in cells))
    key_state = task_root / "state" / "key_pool_task{}.pt".format(stage)
    build = [
        args.python, "-m", "compose.v9.formal_eval",
        "--build-cells", "--root", str(eval_root), "--stage", str(stage),
        "--tasks", ",".join(str(cell) for cell in cells),
        "--key-state", str(key_state), "--cells-json", str(cells_path),
        "--instructions-root", args.instructions_root,
    ]
    if args.query_cache_manifest:
        build += ["--query-cache-manifest", args.query_cache_manifest]
        if args.query_cache_root:
            build += ["--query-cache-root", args.query_cache_root]
    else:
        build += ["--legacy-query-cache-root", args.instructions_root]
    subprocess.run(build, check=True, cwd=args.repo)

    run = [
        args.python, "-m", "compose.v9.formal_eval",
        "--root", str(eval_root), "--stage", str(stage),
        "--cells-json", str(cells_path), "--key-state", str(key_state),
        "--checkpoint-dir", str(task_root / "training" / "task{}".format(stage)),
        "--model-path", args.model_path, "--projector-path", args.projector_path,
        "--vision-tower", args.vision_tower, "--image-folder", args.image_folder,
        "--gpus", args.eval_gpus, "--python", args.python,
    ]
    log_path = Path(args.run_root) / "logs" / "task{}_eval.log".format(stage)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n===== {} eval stage {} cells {} =====\n{}\n".format(
            _now(), stage, cells, " ".join(run)))
        log.flush()
        completed = subprocess.run(run, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise V9ChainError(
            "evaluation of A[{}][{}] failed (exit {}); see {}".format(
                stage, cells, completed.returncode, log_path
            )
        )
    matrix = eval_root / "evaluation" / "continual_matrix.json"
    return {"matrix": str(matrix), "log": str(log_path), "stage": int(stage), "cells": cells}


def matrix_cells(args, stage: int) -> Dict[int, Dict[str, Any]]:
    """The scored cells of row ``stage``, as the matrix currently holds them."""
    matrix_path = Path(args.run_root) / "evaluation_matrix" / "evaluation" / "continual_matrix.json"
    if not matrix_path.is_file():
        return {}
    matrix = _read_json(matrix_path)
    row = (matrix.get("rows") or {}).get(str(int(stage)), {})
    return {
        int(cell): metric
        for cell, metric in row.items()
        if isinstance(metric, dict) and metric.get("value") is not None
    }


def clean_task_state(args, task: int) -> None:
    """Remove a task's partial state so the retry starts from a clean pool.

    Only ever called for the OOM fallback, and only with the task's own
    directory: the batch shape is part of what the optimizer state was built
    for, so a retry at a different micro-batch must not resume onto it.
    """
    task_root = Path(args.run_root) / "task{}".format(task)
    for name in ("training", "state", "metrics", "features", "data"):
        target = task_root / name
        if target.exists():
            shutil.rmtree(target)


def training_in_flight(args, task: int) -> List[int]:
    """PIDs already training this task, so a second copy is never started.

    A supervisor that is restarted mid-task -- because it died, or because it
    was replaced -- must attach to the training that is already running rather
    than begin another one against the same output directory.  Two trainers
    writing one checkpoint directory corrupt it in a way no later gate can
    detect, so the question is asked *before* a command is built, and it is
    asked of the process table rather than of a file the dead supervisor left
    behind.
    """
    training_dir = str(Path(args.run_root) / "task{}".format(task) / "training" / "task{}".format(task))
    pids: List[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if training_dir in command and ("torch.distributed.run" in command or "train_compose" in command):
            pids.append(int(entry.name))
    return sorted(pids)


def _recorded_batch(task_root: Path) -> Dict[str, Any]:
    """The batch a task actually ran at, from its own commit record."""
    path = task_root / "data" / "task_complete.json"
    if not path.is_file():
        return {}
    try:
        payload = _read_json(path)
    except (OSError, ValueError):
        return {}
    return {
        key: payload[key]
        for key in ("per_device_batch", "grad_accum")
        if isinstance(payload.get(key), int)
    }


def self_eval_record(args, task: int, metric: Dict[str, Any]) -> Dict[str, Any]:
    """What ``A[t][t]`` was measured from, in a form a later pass can re-check.

    The score on its own is not evidence: a number in the matrix cannot say
    which pool produced it, so a diagonal cell whose pool was later replaced
    would look exactly like one whose pool was not.  The hashes here are what
    lets the sweep *reuse* the cell instead of regenerating it -- reuse is only
    safe if the artefact it was measured from is still the artefact on disk.
    """
    task_root = Path(args.run_root) / "task{}".format(task)
    training = task_root / "training" / "task{}".format(task)
    selection = (Path(args.run_root) / "evaluation_matrix" / "evaluation"
                 / "selections" / "t{}".format(task) / "task{}".format(task) / "selections.json")
    if not selection.is_file():
        raise V9ChainError(
            "A[{}][{}] is scored but {} does not exist, so the query geometry it "
            "was measured with cannot be established; a cell whose provenance is "
            "unknown must not be reused as a formal number".format(task, task, selection)
        )
    manifest = _read_json(selection)
    query_source = manifest.get("query_source") or {}
    answer_file = metric.get("prediction_file")
    return {
        "task_index": int(task),
        "metric": metric.get("metric"),
        "score": metric.get("value"),
        "checkpoint": str(training),
        "committed_key_state": str(task_root / "state" / "key_pool_task{}.pt".format(task)),
        "committed_key_state_sha256": _sha256(task_root / "state" / "key_pool_task{}.pt".format(task)),
        "expert_pool_file": str(training / "compose_experts.bin"),
        "expert_pool_hash": _sha256(training / "compose_experts.bin"),
        "query_hash": query_source.get("query_value_hash"),
        "query_source": query_source,
        "answer_file": answer_file,
        "answer_file_sha256": _sha256(answer_file) if answer_file else None,
        "result_text_sha256": metric.get("result_text_sha256"),
        "scorer": metric.get("scorer"),
        "dataset": metric.get("dataset"),
        # The formal inference contract, recorded per cell rather than asserted
        # once for the run: these are the claims that make the number a
        # committed-pool measurement instead of a leak.
        "top_k": manifest.get("top_k"),
        "cardinality_scale": "none",
        "answer_features_used": manifest.get("answer_features_used"),
        "task_id_used": manifest.get("task_id_used"),
        "training_retrieval_cache_used": manifest.get("training_retrieval_cache_used"),
        "query_encoder_calls": manifest.get("query_encoder_calls"),
        "timestamp": _now(),
    }


def self_eval_intact(args, task: int, record: Dict[str, Any]) -> Dict[str, Any]:
    """Does the recorded cell still describe the artefacts on disk?"""
    task_root = Path(args.run_root) / "task{}".format(task)
    training = task_root / "training" / "task{}".format(task)
    checks = {
        "expert_pool": record.get("expert_pool_hash") == _sha256(training / "compose_experts.bin"),
        "committed_key_state": record.get("committed_key_state_sha256")
        == _sha256(task_root / "state" / "key_pool_task{}.pt".format(task)),
    }
    return {"ok": all(checks.values()), "checks": checks}


def self_eval(args, task: int) -> Dict[str, Any]:
    """``A[t][t]`` -- the one cell the next task is allowed to wait for."""
    task_root = Path(args.run_root) / "task{}".format(task)
    record_path = task_root / "evaluation" / "self_eval.json"
    scored = matrix_cells(args, task)
    cell = scored.get(int(task))
    if cell is not None and record_path.is_file() and not args.force:
        record = _read_json(record_path)
        intact = self_eval_intact(args, task, record)
        if intact["ok"]:
            return {"task": task, "action": "diagonal-already-scored", "record": record}
        raise V9ChainError(
            "A[{}][{}] was measured from a pool that is no longer the committed "
            "one ({}); the cell cannot be reused and must not be silently "
            "regenerated either, because whatever replaced the pool is a "
            "correctness problem of its own".format(task, task, intact["checks"])
        )
    if cell is not None and not args.force:
        # A cell that exists without its provenance record -- the per-task pass
        # that produced it predates this record.  Do not regenerate it: the
        # answers cost GPU hours and the cell is already in the matrix.  Write
        # down what it was measured from instead.
        record = self_eval_record(args, task, cell)
        _write_json(record_path, record)
        return {"task": task, "action": "diagonal-recorded-from-existing", "record": record}

    beat(args.run_root, "self_eval", task)
    outcome = evaluate_task(args, task, [task])
    cell = matrix_cells(args, task).get(int(task))
    if cell is None:
        raise V9ChainError(
            "A[{}][{}] is missing after evaluation; task {} cannot be handed "
            "off without its own-test-set number".format(task, task, task)
        )
    record = self_eval_record(args, task, cell)
    _write_json(record_path, record)
    return {"task": task, "action": "self-eval-produced", "record": record, "evaluation": outcome}


def train_phase(args, task: int, require_diagonal: bool) -> Dict[str, Any]:
    """The gate that admits task ``task + 1``.

    It is ``task_completion`` with the row narrowed to this task's own cell, and
    it is written to disk as its own artefact so that "task t is finished" is
    answerable without re-deriving it from six requirements across three
    directories.  The cross-task cells are deliberately not part of it: they
    are deferred to the sweep, and a gate that waited for them would be back to
    holding the next task's training behind the whole row.
    """
    task_root = Path(args.run_root) / "task{}".format(task)
    completion = task_completion(
        task_root, task,
        require_full_coverage=not args.sanity,
        require_eval=require_diagonal,
        eval_root=Path(args.run_root) / "evaluation_matrix",
        eval_cells=[task] if require_diagonal else None,
    )
    if not completion["complete"]:
        raise V9ChainError(
            "task {} is not ready to hand off: {}".format(task, completion["failed"])
        )
    payload = {
        "task": task,
        "task_name": TASK_NAMES[task],
        "timestamp": _now(),
        "completion": completion,
        "diagonal_required": bool(require_diagonal),
        "git_sha": _git_sha(Path(args.repo)),
    }
    if require_diagonal:
        record_path = task_root / "evaluation" / "self_eval.json"
        if not record_path.is_file():
            raise V9ChainError(
                "task {} has no {}: the diagonal cell exists but nothing records "
                "what it was measured from".format(task, record_path)
            )
        record = _read_json(record_path)
        intact = self_eval_intact(args, task, record)
        if not intact["ok"]:
            raise V9ChainError(
                "task {}'s self-evaluation no longer matches its committed "
                "artefacts: {}".format(task, intact["checks"])
            )
        payload["self_eval"] = record
    _write_json(task_root / "data" / "train_phase_complete.json", payload)
    return payload


def final_sweep(args) -> Dict[str, Any]:
    """Fill the cross-task cells, once, after the last task has trained.

    A cell that is already scored is never regenerated.  For the diagonal that
    is not merely an optimisation -- it is the point of the scheduling change:
    ``A[t][t]`` was measured beside the training that produced it, and a sweep
    that re-measured it would be spending GPU hours to re-derive a number the
    run already has, with the extra risk that the two disagree.
    """
    reused: List[Dict[str, Any]] = []
    filled: List[Dict[str, Any]] = []
    for stage in range(int(args.from_task), int(args.to_task) + 1):
        task_root = Path(args.run_root) / "task{}".format(stage)
        if not (task_root / "state" / "key_pool_task{}.pt".format(stage)).is_file():
            raise V9ChainError(
                "the sweep reached stage {} but the task has not committed a "
                "pool; the sweep may only run after every task has finished".format(stage)
            )
        scored = matrix_cells(args, stage)
        record_path = task_root / "evaluation" / "self_eval.json"
        if int(stage) in scored:
            if not record_path.is_file():
                raise V9ChainError(
                    "A[{}][{}] is in the matrix but {} is missing, so the cell "
                    "cannot be certified as reused".format(stage, stage, record_path)
                )
            record = _read_json(record_path)
            intact = self_eval_intact(args, stage, record)
            if not intact["ok"]:
                raise V9ChainError(
                    "refusing to build row {} on top of a diagonal cell measured "
                    "from artefacts that have since changed: {}".format(stage, intact["checks"])
                )
            reused.append({"stage": stage, "cell": int(stage), "score": record.get("score")})
        missing = [cell for cell in range(int(stage) + 1) if cell not in scored]
        if not missing:
            continue
        beat(args.run_root, "final_sweep", stage, cells=missing)
        outcome = evaluate_task(args, stage, missing)
        after = matrix_cells(args, stage)
        still = [cell for cell in missing if cell not in after]
        if still:
            raise V9ChainError(
                "the sweep asked for A[{}][{}] and they are still missing after a "
                "successful evaluation run".format(stage, still)
            )
        filled.append({"stage": stage, "cells": missing, "log": outcome["log"]})
    return {"reused_diagonal": reused, "filled": filled}


def _attempts_path(task_root: Path) -> Path:
    # Deliberately outside ``data/``: the OOM fallback wipes that directory to
    # restart the task from a clean state, and an attempt counter that the
    # recovery itself erases cannot bound anything.
    return task_root / "attempts.json"


def _record_attempt(task_root: Path, reason: str) -> int:
    path = _attempts_path(task_root)
    history: List[Dict[str, Any]] = []
    if path.is_file():
        try:
            history = list(_read_json(path).get("attempts") or [])
        except (OSError, ValueError, AttributeError):
            history = []
    history.append({"attempt": len(history) + 1, "reason": reason, "timestamp": _now()})
    _write_json(path, {"attempts": history, "count": len(history)})
    return len(history)


def await_training(args, task: int, poll_seconds: float = 30.0) -> Dict[str, Any]:
    """Wait for a training of ``task`` that another supervisor started.

    Returns when the task's artefacts are complete or when the training is gone.
    A supervisor replaced mid-task must not begin a second training against the
    same output directory -- two trainers sharing a checkpoint directory corrupt
    it in a way no later gate can detect -- so it waits instead.  The wait ends
    early if the processes disappear, because at that point there is nothing to
    attach to and the caller has to decide what the death means.
    """
    started = time.time()
    while True:
        completion = task_completion(
            Path(args.run_root) / "task{}".format(task), task,
            require_full_coverage=not args.sanity, require_eval=False,
            eval_root=Path(args.run_root) / "evaluation_matrix",
        )
        if completion["complete"]:
            return {"task": task, "waited": round(time.time() - started, 1), "complete": True}
        pids = training_in_flight(args, task)
        if not pids:
            _record_attempt(Path(args.run_root) / "task{}".format(task), "died-while-attached")
            return {"task": task, "waited": round(time.time() - started, 1), "complete": False}
        beat(args.run_root, "attached", task, pids=pids)
        time.sleep(poll_seconds)


def other_supervisors(args) -> List[int]:
    """Other ``v9_chain`` processes driving this same run root."""
    root = str(Path(args.run_root))
    pids: List[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if "compose.experiments.v9_chain" in command and root in command:
            pids.append(int(entry.name))
    return sorted(pids)


def run_task(args, task: int, per_device_batch: int, grad_accum: int) -> Dict[str, Any]:
    task_root = Path(args.run_root) / "task{}".format(task)
    task_root.mkdir(parents=True, exist_ok=True)
    full_coverage = not args.sanity
    eval_root = Path(args.run_root) / "evaluation_matrix"
    with task_lock(Path(args.run_root), task):
        already = task_completion(
            task_root, task, require_full_coverage=full_coverage, require_eval=False,
            eval_root=eval_root,
        )
        if already["complete"] and not args.force:
            recorded = _recorded_batch(task_root)
            return {
                "task": task, "action": "training-already-complete", "completion": already,
                "per_device_batch": recorded.get("per_device_batch", per_device_batch),
                "grad_accum": recorded.get("grad_accum", grad_accum),
            }
        waiting = training_in_flight(args, task)
        if waiting and not args.force:
            print(
                "[v9s] task {} is already being trained by pids {}; attaching to it "
                "rather than starting a second training".format(task, waiting),
                flush=True,
            )
            beat(args.run_root, "attached", task, pids=waiting)
            await_training(args, task)
            handoff = handoff_checks(args, task)
            contract = full_data_contract(args, task)
            if not handoff["all_ok"]:
                raise V9ChainError(
                    "task {} finished under another supervisor but its handoff "
                    "artefacts are incomplete: {}".format(
                        task, sorted(k for k, v in handoff.items() if v is False and k != "all_ok")
                    )
                )
            if not args.sanity and not contract["ok"]:
                raise V9ChainError(
                    "task {} finished under another supervisor but did not cover "
                    "its declared split: {}".format(task, contract["failed"])
                )
            _write_json(task_root / "data" / "task_complete.json", {
                "task": task, "task_name": TASK_NAMES[task], "timestamp": _now(),
                "per_device_batch": per_device_batch, "grad_accum": grad_accum,
                "world_size": args.world_size,
                "global_batch": args.world_size * per_device_batch * grad_accum,
                "sanity": bool(args.sanity), "max_steps": int(args.max_steps),
                "supervised_by": "attached", "attached_pids": waiting,
                "full_data_contract": contract, "handoff": handoff,
            })
            return {
                "task": task, "action": "trained-under-previous-supervisor",
                "handoff": handoff, "full_data_contract": contract,
                "per_device_batch": per_device_batch, "grad_accum": grad_accum,
            }

        attempts = _record_attempt(task_root, "launch")
        if attempts > MAX_TRAINING_ATTEMPTS:
            raise V9ChainError(
                "task {} has been launched {} times and has not produced a "
                "handoff; the permitted retries are exhausted, so this is a "
                "failure to diagnose rather than one to retry".format(task, attempts)
            )
        env = dict(os.environ)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
        training_dir = task_root / "training" / "task{}".format(task)
        oom_fallback_used = False
        while True:
            # A crash resumes from the checkpoint the task already wrote; the
            # batch shape, the data order and the query geometry are all
            # unchanged, which is what makes the resume the same training rather
            # than a second one.  The count is bounded, and each launch is
            # recorded before it happens, so a crash loop cannot outrun it.
            resume = training_dir.is_dir() and any(training_dir.glob("checkpoint-*"))
            command = training_command(args, task, per_device_batch, grad_accum, resume,
                                       args.profile_training, full_coverage)
            beat(args.run_root, "training", task, attempt=attempts, resume=bool(resume))
            outcome = run_training(args, command, task, env)
            if outcome["returncode"] == 0:
                break
            if outcome["oom"]:
                if oom_fallback_used:
                    raise V9ChainError(
                        "task {} ran out of memory again at micro-batch {} with "
                        "accumulation {} (global batch {}); that was the one "
                        "permitted fallback, so the chain stops here rather than "
                        "trying a third size -- see {}. Reported, not tuned "
                        "around.".format(
                            task, per_device_batch, grad_accum,
                            args.world_size * per_device_batch * grad_accum,
                            outcome["log"],
                        )
                    )
                # One fallback, recorded, from a clean task state.  The fallback
                # is the last attempt: micro 4 / accum 2 OOMs -> restart this
                # task clean at micro 2 / accum 4 (global batch still 32) -> if
                # *that* also runs out of memory the chain stops.  It is not "if
                # micro 4 fails again" -- micro 4 is gone by then -- so no third
                # size and no continued tuning.
                fallback_batch = max(1, per_device_batch // 2)
                fallback_accum = max(1, (args.world_size * per_device_batch * grad_accum)
                                     // (args.world_size * fallback_batch))
                _write_json(Path(args.run_root) / "data" / "oom_fallback_task{}.json".format(task), {
                    "task": task, "timestamp": _now(),
                    "status": "OOM_INVALIDATED",
                    "from": {"per_device_batch": per_device_batch, "grad_accum": grad_accum},
                    "to": {"per_device_batch": fallback_batch, "grad_accum": fallback_accum},
                    "reason": "CUDA out of memory",
                    "global_batch_preserved": args.world_size * fallback_batch * fallback_accum,
                    "note": "the task restarts from a clean state; the interrupted "
                            "attempt's optimizer state was built for a different "
                            "micro-batch and is not resumed onto",
                })
                clean_task_state(args, task)
                _record_attempt(task_root, "oom-fallback")
                per_device_batch, grad_accum = fallback_batch, fallback_accum
                oom_fallback_used = True
                continue
            if attempts >= MAX_TRAINING_ATTEMPTS:
                raise V9ChainError(
                    "task {} training crashed {} times (last exit {}); the "
                    "permitted retries are exhausted -- see {}".format(
                        task, attempts, outcome["returncode"], outcome["log"]
                    )
                )
            attempts = _record_attempt(task_root, "retry-after-crash")
            print(
                "[v9s] task {} exited {}; retrying from the latest checkpoint "
                "(launch {} of {})".format(task, outcome["returncode"], attempts,
                                           MAX_TRAINING_ATTEMPTS),
                flush=True,
            )

        handoff = handoff_checks(args, task)
        if not handoff["all_ok"]:
            raise V9ChainError(
                "task {} trained but the handoff artefacts are incomplete: {}".format(
                    task, sorted(k for k, v in handoff.items() if v is False and k != "all_ok")
                )
            )
        # The full-data tail contract, checked here so a task that dropped its last
        # accumulation window stops the chain instead of being handed to the next
        # one.  A sanity run caps its step count and cannot meet it, so it is asked
        # only to record the numbers -- the same reason its completion gate relaxes
        # the coverage requirement.
        contract = full_data_contract(args, task)
        if not args.sanity and not contract["ok"]:
            raise V9ChainError(
                "task {} trained but did not cover its declared split: {}".format(
                    task, contract["failed"]
                )
            )
        performance = performance_report(task_root)
        if performance.get("warning"):
            _write_json(task_root / "data" / "performance_warning.json", performance)
        _write_json(task_root / "data" / "task_complete.json", {
            "task": task, "task_name": TASK_NAMES[task], "timestamp": _now(),
            "per_device_batch": per_device_batch, "grad_accum": grad_accum,
            "world_size": args.world_size,
            "global_batch": args.world_size * per_device_batch * grad_accum,
            "sanity": bool(args.sanity), "max_steps": int(args.max_steps),
            "supervised_by": "launched", "attempts": attempts,
            "full_data_contract": contract,
            "handoff": handoff, "performance": performance,
        })
        return {
            "task": task, "action": "trained", "outcome": outcome, "handoff": handoff,
            "full_data_contract": contract, "performance": performance,
            "per_device_batch": per_device_batch, "grad_accum": grad_accum,
        }


# ----------------------------------------------------------------------
# the sequence
# ----------------------------------------------------------------------
def final_verification(args, results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Every gate over every task, before anything is allowed to say COMPLETE.

    The sanity run reaches this same function, so the check it performs is the
    check the formal run performs -- what differs is only that a sanity run is
    allowed to come out incomplete, because it deliberately stops short (a
    capped step count cannot satisfy the full-coverage gate, and eval is
    restricted so that six rows of 3000 samples are not generated twice).  The
    verdict is still written, so "the sanity passed" is never a claim made
    without the evidence beside it.
    """
    run_root = Path(args.run_root)
    eval_root = run_root / "evaluation_matrix"
    full_coverage = not args.sanity
    per_task: Dict[str, Any] = {}
    blocked: List[str] = []
    for task in range(len(TASK_NAMES)):
        task_root = run_root / "task{}".format(task)
        if not task_root.is_dir():
            blocked.append("task{}: no run directory".format(task))
            continue
        completion = task_completion(
            task_root, task, require_full_coverage=full_coverage, require_eval=True,
            eval_root=eval_root,
        )
        per_task[str(task)] = completion
        if not completion["complete"]:
            blocked.append("task{}: {}".format(task, completion["failed"]))
        # The diagonal cell of every row is reused, not regenerated, by the
        # sweep -- so the record of what it was measured from has to still
        # describe the artefacts on disk, or the run is reporting a number
        # against a pool it no longer has.
        record_path = task_root / "evaluation" / "self_eval.json"
        if not record_path.is_file():
            blocked.append("task{}: no self-evaluation record at {}".format(task, record_path))
        else:
            try:
                intact = self_eval_intact(args, task, _read_json(record_path))
            except (OSError, ValueError) as error:
                blocked.append("task{}: unreadable self-evaluation record: {}".format(task, error))
            else:
                per_task[str(task)]["self_eval_intact"] = intact
                if not intact["ok"]:
                    blocked.append(
                        "task{}: self-evaluation no longer matches its committed "
                        "artefacts: {}".format(task, intact["checks"])
                    )
    matrix_path = eval_root / "evaluation" / "continual_matrix.json"
    cells = 0
    if matrix_path.is_file():
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        for task in range(len(TASK_NAMES)):
            cells += len((matrix.get("rows") or {}).get(str(task), {}))
    expected_cells = sum(range(1, int(args.to_task) + 2))
    if cells != expected_cells:
        blocked.append(
            "matrix has {} cells over tasks 0..{}, expected {}".format(
                cells, args.to_task, expected_cells
            )
        )
    return {
        "complete": not blocked,
        "sanity": bool(args.sanity),
        "blocked_by": blocked,
        "cells": cells,
        "expected_cells": expected_cells,
        "tasks": per_task,
        "results": list(results),
    }


def failure_report(args, task: Optional[int], stage: str, error: BaseException) -> Dict[str, Any]:
    run_root = Path(args.run_root)
    log_tail: List[str] = []
    if task is not None:
        for name in ("logs/task{}_train.log".format(task), "logs/task{}_eval.log".format(task)):
            log_tail.extend(_tail(run_root / name, 200))
    checkpoint = None
    if task is not None:
        checkpoints = sorted((run_root / "task{}".format(task) / "training" / "task{}".format(task)).glob("checkpoint-*"))
        checkpoint = str(checkpoints[-1]) if checkpoints else None
    return {
        "timestamp": _now(), "task": task, "stage": stage,
        "error": "{}: {}".format(type(error).__name__, error),
        "git_sha": _git_sha(Path(args.repo)),
        "config": args.config,
        "run_root": str(run_root),
        "last_checkpoint": checkpoint,
        "resumable": checkpoint is not None,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "log_tail": log_tail[-200:],
    }


def write_summary(args, verification: Dict[str, Any]) -> None:
    run_root = Path(args.run_root)
    prefix = "sanity" if args.sanity else "formal"
    _write_json(run_root / "{}.json".format(prefix + "_summary"), verification)
    lines = [
        "# V9-S {} run".format(prefix),
        "",
        "- git SHA: `{}`".format(_git_sha(Path(args.repo))),
        "- run root: `{}`".format(run_root),
        "- world size: {}".format(args.world_size),
        "- global batch: {} ({} x {} x {})".format(
            args.world_size * args.per_device_batch * args.grad_accum,
            args.world_size, args.per_device_batch, args.grad_accum,
        ),
        "- query source: {}".format(args.query_cache_manifest or "precomputed"),
        "- query encoder calls: 0",
        "- max optimizer steps per task: {}".format(args.max_steps or "uncapped"),
        "- per task: training -> commit -> `A[t][t]`; cross-task cells filled by "
        "one sweep after the last task",
        "- lower-triangular cells: {}/{}".format(verification["cells"], verification["expected_cells"]),
        "",
        "## tasks",
        "",
    ]
    for task in range(int(args.to_task) + 1):
        row = verification["tasks"].get(str(task)) or {}
        lines.append(
            "- task {} ({}): {}".format(
                task, TASK_NAMES[task],
                "COMPLETE" if row.get("complete") else "INCOMPLETE {}".format(row.get("failed")),
            )
        )
    if verification["blocked_by"]:
        lines += ["", "## blocked by", ""] + ["- {}".format(item) for item in verification["blocked_by"]]
    (run_root / "{}.md".format(prefix + "_summary")).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def chain(args) -> int:
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    prefix = "SANITY" if args.sanity else "FORMAL"
    running, complete, failed = (
        "{}_{}".format(prefix, name)
        for name in ("RUNNING", "COMPLETE", "FAILED")
    )
    _clear_markers(run_root)
    _marker(run_root, running)
    (run_root / "formal_git_sha.txt").write_text(
        "{}\n".format(_git_sha(Path(args.repo)) or "unknown"), encoding="utf-8"
    )
    results: List[Dict[str, Any]] = []
    task: Optional[int] = None
    stage = "startup"
    try:
        if args.require_clean_git and _git_dirty(Path(args.repo)):
            raise V9ChainError(
                "the working tree or index has uncommitted changes; the formal "
                "run is bound to a commit and refuses to start on a tree that "
                "can change under it"
            )
        gate = go_no_go(
            args.preflight_root,
            world_size=args.world_size,
            per_device_batch=args.per_device_batch,
            grad_accum=args.grad_accum,
            query_encoder_calls=0,
        )
        _write_json(run_root / "data" / "go_no_go.json", gate)
        if gate["decision"] != "GO":
            raise V9ChainError("GO/NO-GO gate is NO-GO: {}".format(gate["blocked_by"]))

        rivals = other_supervisors(args)
        if rivals and not args.takeover:
            raise V9ChainError(
                "another v9_chain (pids {}) is driving this run root; two "
                "supervisors cannot share a run -- pass --takeover only after "
                "the other one has actually stopped".format(rivals)
            )
        if rivals:
            print(
                "[v9s] taking over from supervisor pids {}; the task lock and the "
                "attach check decide what may safely be continued".format(rivals),
                file=sys.stderr,
            )

        per_device_batch = args.per_device_batch
        grad_accum = args.grad_accum
        for task in range(args.from_task, args.to_task + 1):
            stage = "task{}".format(task)
            result = run_task(args, task, per_device_batch, grad_accum)
            # The batch the task actually ran at travels forward: if task 0
            # fell back for memory, tasks 1..5 -- which offer more experts and
            # need at least as much -- must not silently retry the larger one.
            per_device_batch = result.get("per_device_batch", per_device_batch)
            grad_accum = result.get("grad_accum", grad_accum)
            # The task's own test set, measured with the pool it just committed.
            # Only after this exists is the task finished with itself and the
            # next one allowed to start.
            if task <= int(args.eval_through):
                stage = "task{}_self_eval".format(task)
                result["self_eval"] = self_eval(args, task)
            stage = "task{}_handoff".format(task)
            result["train_phase"] = train_phase(
                args, task, require_diagonal=task <= int(args.eval_through)
            )
            results.append(result)

        stage = "final_sweep"
        sweep = final_sweep(args) if not args.sanity else {
            "skipped": "a sanity run does not fill the cross-task cells; its "
                       "evaluation is restricted on purpose"
        }
        _write_json(run_root / "data" / "final_sweep.json", sweep)
        stage = "final_verification"
        verification = final_verification(args, results)
        verification["sweep"] = sweep
        write_summary(args, verification)
        if not verification["complete"]:
            if args.sanity:
                # A capped sanity run is *expected* to fall short of the
                # full-coverage and cell-count gates; it reports them instead of
                # failing on them.  Nothing a sanity run writes is read by the
                # formal chain, so this cannot soften a formal verdict.
                print(
                    "SANITY incomplete (expected for a capped run): {}".format(
                        verification["blocked_by"]
                    ),
                    file=sys.stderr,
                )
                _marker(run_root, complete)
                (run_root / running).unlink(missing_ok=True)
                return 0
            raise V9ChainError(
                "sequence finished but the final gates are not satisfied: {}".format(
                    verification["blocked_by"]
                )
            )
        _marker(run_root, complete)
        (run_root / running).unlink(missing_ok=True)
        return 0
    except BaseException as error:  # noqa: BLE001 - the marker must be written for any failure
        report = failure_report(args, task, stage, error)
        _write_json(run_root / "{}.json".format("sanity_failure" if args.sanity else "failure_report"), report)
        _marker(run_root, failed)
        (run_root / running).unlink(missing_ok=True)
        print("{}_FAILED at {}: {}".format(prefix, stage, error), file=sys.stderr)
        return 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--instructions-root", required=True)
    parser.add_argument("--split-root", default=None,
                        help="where v7_train/ and v7_validation/ live; defaults to --instructions-root")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--projector-path", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--query-encoder", default=None)
    parser.add_argument("--query-cache-manifest", default=None)
    parser.add_argument("--query-cache-root", default=None)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--eval-gpus", default="0,1,2,3")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--per-device-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--key-learning-rate", type=float, default=3e-4)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--from-task", type=int, default=0)
    parser.add_argument("--to-task", type=int, default=len(TASK_NAMES) - 1)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="cap optimizer steps per task; the sanity run sets it, formal leaves it 0")
    parser.add_argument("--eval-through", type=int, default=None,
                        help="highest task index whose row is generated; defaults to --to-task.  "
                             "The sanity run stops generating after task 0 so the six full test "
                             "splits are not scored twice on the same day")
    parser.add_argument("--sanity", action="store_true",
                        help="a capped preflight run: own markers, own summary name, and the "
                             "full-coverage gate relaxed because a capped run cannot meet it")
    parser.add_argument("--preflight-root", default=None,
                        help="e.g. the sanity run's nested output dir; task0 alone when omitted")
    parser.add_argument("--profile-training", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="re-run a task even if its completion gate already passes")
    parser.add_argument("--takeover", action="store_true",
                        help="start even though another v9_chain is recorded against this "
                             "run root; use only once that supervisor has actually stopped")
    parser.add_argument("--require-clean-git", action="store_true", default=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.eval_through is None:
        args.eval_through = int(args.to_task)
    global_batch = args.world_size * args.per_device_batch * args.grad_accum
    if global_batch != 32:
        print(
            "refusing to start: world {} x per-device {} x accum {} = {} != 32".format(
                args.world_size, args.per_device_batch, args.grad_accum, global_batch
            ),
            file=sys.stderr,
        )
        return 2
    return chain(args)


if __name__ == "__main__":
    raise SystemExit(main())
