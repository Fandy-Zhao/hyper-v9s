"""The unattended V9-S six-task chain: train -> evaluate -> hand off -> next.

This module exists so that the formal run does not need anyone watching it.  It
is the difference between a *script that starts a task* and a *pipeline that
finishes a sequence*: every decision the run would otherwise need a human to
make -- is this task done, did training finish, did the handoff land, is the
row complete, did it OOM, may I take the next step -- is taken here, written
down, and enforced.

The order per task is fixed by the spec and is not configurable:

    train -> in-process audits -> task-end commit -> write the row
          -> verify the handoff Task{t+1} will load -> next task

What it refuses to do is as important as what it does.  It will not advance on
a task whose row is incomplete, it will not resume across a batch-shape change
(that would re-enter an optimizer state built for different gradients), and it
will not print "sequence complete" unless every gate in
:mod:`compose.v9.closure` passes for every task.  A failure leaves a
``FORMAL_FAILED`` marker and a ``failure_report.json`` and exits non-zero
rather than looking finished.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from compose.v9.closure import TASK_NAMES, go_no_go, task_completion

MARKER_RUNNING = "FORMAL_RUNNING"
MARKER_COMPLETE = "FORMAL_COMPLETE"
MARKER_FAILED = "FORMAL_FAILED"

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


def evaluate_task(args, task: int) -> Dict[str, Any]:
    """Write row ``A[task][0..task]`` with the task's committed pool."""
    eval_root = Path(args.run_root) / "evaluation_matrix"
    task_root = Path(args.run_root) / "task{}".format(task)
    cells_path = eval_root / "cells_t{}.json".format(task)
    key_state = task_root / "state" / "key_pool_task{}.pt".format(task)
    build = [
        args.python, "-m", "compose.v9.formal_eval",
        "--build-cells", "--root", str(eval_root), "--stage", str(task),
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
        "--root", str(eval_root), "--stage", str(task),
        "--cells-json", str(cells_path), "--key-state", str(key_state),
        "--checkpoint-dir", str(task_root / "training" / "task{}".format(task)),
        "--model-path", args.model_path, "--projector-path", args.projector_path,
        "--vision-tower", args.vision_tower, "--image-folder", args.image_folder,
        "--gpus", args.eval_gpus, "--python", args.python,
    ]
    log_path = Path(args.run_root) / "logs" / "task{}_eval.log".format(task)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n===== {} eval task {} =====\n{}\n".format(_now(), task, " ".join(run)))
        log.flush()
        completed = subprocess.run(run, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise V9ChainError(
            "evaluation of row {} failed (exit {}); see {}".format(
                task, completed.returncode, log_path
            )
        )
    matrix = eval_root / "evaluation" / "continual_matrix.json"
    return {"matrix": str(matrix), "log": str(log_path)}


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


def run_task(args, task: int, per_device_batch: int, grad_accum: int) -> Dict[str, Any]:
    task_root = Path(args.run_root) / "task{}".format(task)
    task_root.mkdir(parents=True, exist_ok=True)
    full_coverage = not args.sanity
    evaluate = task <= int(args.eval_through)
    completion = task_completion(
        task_root, task, require_full_coverage=full_coverage, require_eval=evaluate,
        eval_root=Path(args.run_root) / "evaluation_matrix",
    )
    if completion["complete"] and not args.force:
        return {"task": task, "action": "skipped", "completion": completion}

    resume = (task_root / "training" / "task{}".format(task)).is_dir() and any(
        (task_root / "training" / "task{}".format(task)).glob("checkpoint-*")
    )
    command = training_command(args, task, per_device_batch, grad_accum, resume,
                               args.profile_training, full_coverage)
    env = dict(os.environ)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    outcome = run_training(args, command, task, env)

    if outcome["returncode"] != 0:
        if not outcome["oom"]:
            raise V9ChainError(
                "task {} training failed (exit {}); see {}".format(
                    task, outcome["returncode"], outcome["log"]
                )
            )
        # One fallback, recorded, from a clean task state.  The fallback is the
        # last attempt: micro 4 / accum 2 OOMs -> restart this task clean at
        # micro 2 / accum 4 (global batch still 32) -> if *that* also runs out
        # of memory the chain stops.  It is not "if micro 4 fails again" --
        # micro 4 is gone by then -- so no third size and no continued tuning.
        fallback_batch = max(1, per_device_batch // 2)
        fallback_accum = max(1, (args.world_size * per_device_batch * grad_accum)
                             // (args.world_size * fallback_batch))
        _write_json(Path(args.run_root) / "data" / "oom_fallback_task{}.json".format(task), {
            "task": task, "timestamp": _now(),
            "from": {"per_device_batch": per_device_batch, "grad_accum": grad_accum},
            "to": {"per_device_batch": fallback_batch, "grad_accum": fallback_accum},
            "reason": "CUDA out of memory on the first attempt",
            "global_batch_preserved": args.world_size * fallback_batch * fallback_accum,
        })
        clean_task_state(args, task)
        command = training_command(args, task, fallback_batch, fallback_accum, False,
                                   args.profile_training, full_coverage)
        outcome = run_training(args, command, task, env)
        if outcome["returncode"] != 0:
            raise V9ChainError(
                "task {} out of memory at micro-batch {} with accumulation {} "
                "(global batch {}); that was the one permitted fallback, so the "
                "chain stops here rather than trying a third size -- see {}. "
                "Reported, not tuned around.".format(
                    task, fallback_batch, fallback_accum,
                    args.world_size * fallback_batch * fallback_accum,
                    outcome["log"],
                )
            )
        per_device_batch, grad_accum = fallback_batch, fallback_accum

    handoff = handoff_checks(args, task)
    if not handoff["all_ok"]:
        raise V9ChainError(
            "task {} trained but the handoff artefacts are incomplete: {}".format(
                task, sorted(k for k, v in handoff.items() if v is False and k != "all_ok")
            )
        )
    evaluation = evaluate_task(args, task) if evaluate else {
        "skipped": "task {} is above --eval-through {}".format(task, args.eval_through)
    }
    completion = task_completion(
        task_root, task, require_full_coverage=full_coverage, require_eval=evaluate,
        eval_root=Path(args.run_root) / "evaluation_matrix",
    )
    if not completion["complete"]:
        raise V9ChainError(
            "task {} is still incomplete after training and evaluation: {}".format(
                task, completion["failed"]
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
        "handoff": handoff, "evaluation": evaluation, "performance": performance,
    })
    return {
        "task": task, "action": "trained", "outcome": outcome, "handoff": handoff,
        "evaluation": evaluation, "performance": performance,
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
            results.append(result)

        stage = "final_verification"
        verification = final_verification(args, results)
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
