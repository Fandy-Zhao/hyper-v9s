"""An independent supervisor for a running V9-S formal chain.

This is a *separate process*, in its own tmux session, so that an AI session or
an SSH connection ending is not an event the run can notice.  It watches; it
does not participate.  The chain schedules tasks, and this process may only ever
undo a *runtime* fault -- a process that died -- never a *semantic* one, because
a supervisor that answers "this number looks wrong" by restarting something is a
supervisor that can silently turn a broken run into a run that looks finished.

Three rules define what it will not do:

**It never edits the code.**  A daemon that can ``sed`` a source file can change
what the formal SHA means while the run is still claiming that SHA.  If a real
code bug is found the watchdog's job is to *report* it and let a human or an
agent fix it, commit it, and restart the run under the new commit.

**It never kills a training.**  A high GPU utilisation with an old metrics
timestamp is a long batch, a checkpoint, or a calibration -- all of which write
no step metrics for minutes at a time.  ``SUSPECTED_STALL`` is recorded and
waited out; the verdict escalates to a diagnosis only when the process table and
the heartbeat both stop offering a reason.

**It never guesses a verdict.**  A correctness fault -- NaN, a non-finite loss,
a broken one-backbone-forward invariant, a lost optimizer coverage -- stops the
run being *resumable*.  Those are recorded as ``FATAL_CORRECTNESS`` and the
watchdog does not restart anything over them, because the checkpoint it would
resume is the one whose numbers are in doubt.

State lives in ``RUN_ROOT/monitor/``: ``watchdog_status.json`` (atomic, the
current picture) and ``watchdog_events.jsonl`` (append-only, what happened).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from compose.v9.closure import TASK_NAMES
from compose.v9.heartbeat import age as heartbeat_age
from compose.v9.heartbeat import read as heartbeat_read

#: Seconds between checks.  Fast enough to notice a death within a couple of
#: minutes, slow enough that the check itself never becomes load on a machine
#: that is training four ways at once.
CHECK_INTERVAL = 120.0

#: Metrics older than this with the trainer still alive is worth saying out
#: loud; older than the second is worth diagnosing.  Neither is a kill: a
#: checkpoint save, a calibration pass and a validation gain measurement all
#: write no step metrics and all take minutes.
STALL_SUSPECT_SECONDS = 15 * 60
STALL_DIAGNOSE_SECONDS = 20 * 60

#: Consecutive checks with the GPUs idle and no metric movement.
HARD_STALL_CYCLES = 5

#: Free-space thresholds on the filesystem holding the run root.
DISK_WARNING_GB = 50.0
DISK_PAUSE_GB = 20.0

EVENT_TYPES = (
    "PROGRESS", "CHECKPOINT", "TASK_COMPLETE", "SELF_EVAL_START", "SELF_EVAL_COMPLETE",
    "WARNING", "RECOVERY_START", "RECOVERY_COMPLETE", "FATAL",
)

#: Strings that mean the *training* failed, as opposed to a warning the stack
#: prints on a healthy run.  Reported as a warning with the line's age beside
#: it, never as a verdict: a traceback a later restart fixed is still in the
#: log, and reading it as the current state would report a failure that is not
#: happening.
FATAL_PATTERNS = (
    "CUDA out of memory", "NCCL error", "uncorrectable", "Traceback (most recent call last)",
    "RuntimeError", "AssertionError", "detected NaN", "loss is nan",
)

#: The numbers that must be finite wherever a run logs them.  ``None`` is not a
#: violation and not a pass: a metric a run does not log is not evidence, and
#: reporting it as finite would be reporting a reading that was never taken.
FINITE_FIELDS = (
    "loss_total", "loss_answer", "loss_key", "loss_sparse", "loss_budget",
    "contribution_mean", "contribution_positive_rate", "responsibility_mean",
    "responsibility_max", "responsibility_row_sum", "temperature", "gate_grad_abs_mean",
)

#: The one-backbone-forward invariant, as three independent measurements.
RATIO_FIELDS = (
    "backbone_forwards_per_micro_step",
    "model_forwards_per_micro_step",
    "wide_model_forwards_per_micro_step",
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _append_event(root: Path, kind: str, **fields: Any) -> Dict[str, Any]:
    event = {"type": kind, "timestamp": time.time(), "iso": _now()}
    event.update(fields)
    path = Path(root) / "monitor" / "watchdog_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
    return event


def _cmdline(pid: int) -> str:
    try:
        return (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def _processes(match: str, root: Path) -> List[int]:
    """PIDs running ``match`` for this run root, fork-clones collapsed.

    A DataLoader worker is forked from the rank that owns it with its parent's
    argv intact, so "a command line that mentions the module" matches hundreds
    of processes that are not the module -- counting them would report a second
    supervisor, and a hundred ranks, on every single check.

    Two rules separate the real thing from its clones.  The process must have
    been exec'd from a python interpreter, which keeps out the shell and the
    tmux server, whose own argv quote the launch command.  And its argv must
    differ from its parent's: a fork-clone inherits the parent's argv byte for
    byte, while a genuinely spawned child -- torchrun's ranks -- was exec'd with
    an argv of its own even though its parent's command line mentions the same
    module.
    """
    needle = str(root)
    candidates: Dict[int, tuple] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = [part.decode("utf-8", "replace")
                    for part in (entry / "cmdline").read_bytes().split(b"\0") if part]
            stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not argv or "python" not in os.path.basename(argv[0]):
            continue
        command = " ".join(argv)
        if match not in command or needle not in command:
            continue
        try:
            parent = int(stat.rsplit(")", 1)[1].split()[1])
        except (IndexError, ValueError):
            continue
        candidates[int(entry.name)] = (parent, argv)
    return collapse_clones(candidates)


def collapse_clones(candidates: Mapping[int, tuple]) -> List[int]:
    """Drop the processes that are copies of another candidate.

    Separated from the ``/proc`` walk because this is the rule that was wrong:
    the first version dropped any candidate whose parent was also a candidate,
    which silently discarded torchrun's four ranks -- their parent's command
    line names the same module it was launching.  Identical ``argv`` is what
    marks a clone, and it is the only thing that does.
    """
    return sorted(
        pid for pid, (parent, argv) in candidates.items()
        if candidates.get(parent, (None, None))[1] != argv
    )


def _git_sha(repo: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def _gpus() -> List[Dict[str, Any]]:
    query = "index,utilization.gpu,memory.used,temperature.gpu,power.draw"
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode:
        return []
    devices = []
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            devices.append({
                "index": int(parts[0]), "utilization": float(parts[1]),
                "memory_used_mib": float(parts[2]), "temperature_c": float(parts[3]),
                "power_w": float(parts[4]),
            })
        except ValueError:
            continue
    return devices


def _last_metric(task_root: Path) -> Optional[Dict[str, Any]]:
    """The newest step record across ranks, with the file it came from."""
    newest: Optional[Dict[str, Any]] = None
    for path in sorted((task_root / "metrics").glob("*train_steps*.jsonl")):
        try:
            stamp = path.stat().st_mtime
        except OSError:
            continue
        if newest is not None and stamp <= newest["mtime"]:
            continue
        newest = {"path": path, "mtime": stamp, "file": path.name}
    if newest is None:
        return None
    try:
        lines = [line for line in newest["path"].read_text(encoding="utf-8").splitlines() if line.strip()]
        newest["row"] = json.loads(lines[-1]) if lines else {}
        newest["rows"] = len(lines)
    except (OSError, ValueError):
        newest["row"] = {}
        newest["rows"] = 0
    return newest


def _recent_errors(root: Path) -> List[Dict[str, Any]]:
    """Errors the logs still contain, each with how long ago the log last moved."""
    found: List[Dict[str, Any]] = []
    for path in sorted((root / "logs").glob("*.log")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            age_seconds = time.time() - path.stat().st_mtime
        except OSError:
            continue
        for pattern in FATAL_PATTERNS:
            position = text.rfind(pattern)
            if position < 0:
                continue
            found.append({
                "log": path.name,
                "pattern": pattern,
                "age_seconds": round(age_seconds, 1),
                "context": text[max(0, position - 200):position + 400].strip()[-400:],
            })
    return found


def evaluate_status(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Turn one set of readings into a verdict.

    Kept separate from the reading so the taxonomy is testable without a
    training run in front of it: which faults stop the run being resumable,
    which are runtime faults worth undoing, and which merely slow it are decided
    here and nowhere else.
    """
    findings: List[Dict[str, Any]] = []
    fatal: List[str] = []
    recoverable: List[str] = []

    if state.get("non_finite"):
        fatal.append(
            "FATAL_CORRECTNESS: non-finite {}".format(sorted(set(state["non_finite"])))
        )
    if state.get("backbone_invariant_broken"):
        fatal.append(
            "FATAL_CORRECTNESS: traversals per micro-step are {} and must all be 1"
            .format(state.get("backbone_ratios"))
        )
    if state.get("query_encoder_calls"):
        fatal.append(
            "FATAL_CORRECTNESS: a formal run encodes queries live "
            "(query_encoder_calls={})".format(state["query_encoder_calls"])
        )
    if state.get("optimizer_coverage") is not None and state["optimizer_coverage"] < 1.0:
        fatal.append(
            "FATAL_CORRECTNESS: optimizer coverage {:.6f} < 1.0".format(state["optimizer_coverage"])
        )
    if state.get("two_supervisors"):
        fatal.append(
            "FATAL_CORRECTNESS: {} v9_chain processes are driving this run root"
            .format(state.get("supervisors"))
        )
    if state.get("git_sha_mismatch"):
        fatal.append(
            "FATAL_CORRECTNESS: the checkout is at {} but the run recorded {}"
            .format(state.get("git_sha_now"), state.get("git_sha_recorded"))
        )

    free = state.get("disk_free_gb")
    if isinstance(free, (int, float)):
        if free < DISK_PAUSE_GB:
            findings.append({
                "level": "PAUSE", "kind": "disk",
                "detail": "{:.1f} GB free: below the {:.0f} GB floor, so no new task "
                          "or evaluation may start".format(free, DISK_PAUSE_GB),
            })
        elif free < DISK_WARNING_GB:
            findings.append({
                "level": "WARNING", "kind": "disk",
                "detail": "{:.1f} GB free: below the {:.0f} GB warning line".format(
                    free, DISK_WARNING_GB),
            })

    # A fresh heartbeat is proof of life for a stage that writes no metrics at
    # all, which is exactly the stage a metrics-only watchdog would call a
    # stall.  So a stale metric file is only worth reporting when the heartbeat
    # is stale too.
    beat_age = state.get("heartbeat_age")
    fresh_beat = isinstance(beat_age, (int, float)) and beat_age < STALL_SUSPECT_SECONDS
    metric_age = state.get("metric_age")
    if state.get("hard_stall"):
        findings.append({
            "level": "HARD_STALL", "kind": "stall",
            "detail": "{} consecutive checks with no GPU work and no metric progress"
                      .format(state.get("idle_cycles")),
        })
    elif isinstance(metric_age, (int, float)) and metric_age > STALL_SUSPECT_SECONDS and not fresh_beat:
        findings.append({
            "level": "SUSPECTED_STALL" if metric_age <= STALL_DIAGNOSE_SECONDS else "DIAGNOSE",
            "kind": "metrics",
            "detail": "the last step metric is {:.0f}s old while {} training rank(s) are "
                      "alive and the last heartbeat is {}s old".format(
                          metric_age, state.get("ranks"),
                          "unknown" if beat_age is None else "{:.0f}".format(beat_age)),
        })

    if state.get("recent_errors") and state.get("ranks"):
        findings.append({
            "level": "WARNING", "kind": "logs",
            "detail": "the log holds a recent '{}' (log last written {:.0f}s ago) while "
                      "training is alive".format(
                          state["recent_errors"][-1]["pattern"], state["recent_errors"][-1]["age_seconds"]),
        })

    if state.get("chain_dead") and state.get("ranks"):
        findings.append({
            "level": "WARNING", "kind": "supervisor",
            "detail": "the chain process is gone but {} training rank(s) are alive; "
                      "nothing may start a second training against this task"
                      .format(state["ranks"]),
        })
    elif state.get("chain_dead") and not state.get("ranks") and not state.get("run_finished"):
        findings.append({
            "level": "RECOVERABLE", "kind": "supervisor",
            "detail": "the chain process is gone and no training is running",
        })
        recoverable.append("restart_chain")

    if fatal:
        verdict = "FATAL_CORRECTNESS"
    elif recoverable:
        verdict = "RECOVERABLE"
    elif findings:
        verdict = findings[0]["level"]
    else:
        verdict = "HEALTHY"
    return {"verdict": verdict, "findings": findings, "fatal": fatal,
            "recoverable": sorted(set(recoverable))}


def observe(args) -> Dict[str, Any]:
    """One check's readings, with no judgement in them."""
    root = Path(args.run_root)
    supervisors = _processes("compose.experiments.v9_chain", root)
    # torchrun's own command line contains the training module it launches, so
    # it is subtracted here rather than counted as a fifth rank.
    torchrun = _processes("torch.distributed.run", root)
    launchers = set(torchrun)
    ranks = [pid for pid in _processes("train_compose", root) if pid not in launchers]
    tasks: Dict[str, Any] = {}
    current_task: Optional[int] = None
    for index in range(len(TASK_NAMES)):
        task_root = root / "task{}".format(index)
        if not task_root.is_dir():
            continue
        metric = _last_metric(task_root)
        record: Dict[str, Any] = {
            "task_index": index,
            "task_name": TASK_NAMES[index],
            "metric_age": (time.time() - metric["mtime"]) if metric else None,
            "rows": metric["rows"] if metric else 0,
            "latest_metric_file": str(metric["path"]) if metric else None,
            "checkpoints": sorted(
                path.name for path in (task_root / "training" / "task{}".format(index)).glob("checkpoint-*")
            ),
            "train_phase_complete": (task_root / "data" / "train_phase_complete.json").is_file(),
            "self_eval": (task_root / "evaluation" / "self_eval.json").is_file(),
        }
        row = (metric or {}).get("row") or {}
        record["step"] = row.get("step")
        record["stage"] = row.get("stage")
        record["loss_total"] = row.get("loss_total")
        record["contribution_mean"] = row.get("contribution_mean")
        record["responsibility_mean"] = row.get("responsibility_mean")
        record["mean_offered_experts"] = row.get("mean_offered_experts")
        record["max_offered_experts"] = row.get("max_offered_experts")
        record["global_samples_per_sec"] = row.get("global_samples_per_sec")
        record["wide_micro_steps"] = row.get("wide_micro_steps")
        ratios = {key: row.get(key) for key in RATIO_FIELDS}
        record["backbone_ratios"] = ratios
        record["backbone_invariant_broken"] = any(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and not 0.99 <= float(value) <= 1.01
            for value in ratios.values()
        )
        record["non_finite"] = sorted(
            key for key in FINITE_FIELDS
            if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
            and not math.isfinite(float(row[key]))
        )
        tasks[str(index)] = record
        if not record["train_phase_complete"]:
            current_task = index

    recorded_path = root / "formal_git_sha.txt"
    recorded = recorded_path.read_text(encoding="utf-8").strip() if recorded_path.is_file() else None
    git_sha_now = _git_sha(Path(args.repo))
    current = tasks.get(str(current_task)) if current_task is not None else None
    return {
        "timestamp": time.time(),
        "iso": _now(),
        "run_root": str(root),
        "supervisors": len(supervisors),
        "supervisor_pids": supervisors,
        "two_supervisors": len(supervisors) > 1,
        "ranks": len(ranks),
        "rank_pids": ranks,
        "torchrun_pids": torchrun,
        "chain_dead": not supervisors,
        "current_task": current_task,
        "current": {
            "step": (current or {}).get("step"),
            "stage": (current or {}).get("stage"),
            "rows": (current or {}).get("rows"),
            "checkpoint": ((current or {}).get("checkpoints") or [None])[-1],
        },
        "tasks": tasks,
        "gpus": _gpus(),
        "disk_free_gb": round(shutil.disk_usage(root if root.exists() else Path("/")).free / (1024 ** 3), 2),
        "metric_age": (current or {}).get("metric_age"),
        "heartbeat_age": heartbeat_age(root),
        "heartbeat": heartbeat_read(root),
        "markers": sorted(path.name for path in root.glob("FORMAL_*")),
        "git_sha_now": git_sha_now,
        "git_sha_recorded": recorded,
        "git_sha_mismatch": bool(recorded and git_sha_now and recorded != git_sha_now),
        "recent_errors": [
            error for error in _recent_errors(root)
            if error["age_seconds"] <= 3 * CHECK_INTERVAL
        ],
    }


def maybe_recover(args, state: Mapping[str, Any], verdict: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Undo a runtime fault -- and only a runtime fault.

    The only action taken here is starting the chain again when nothing is
    running at all.  It is deliberately the smallest possible recovery: the
    chain's own gates decide what may be skipped, resumed or retrained, and it
    refuses to start a second training for a task that is already training, so
    the worst case of a spurious restart is a process that exits immediately.
    Everything else -- a dead trainer with a checkpoint beside it, a code bug, a
    correctness fault -- is reported for a human, because acting on it means
    deciding something about the *experiment* rather than about a process.
    """
    if not args.auto_recover:
        return None
    if verdict["verdict"] != "RECOVERABLE" or "restart_chain" not in verdict["recoverable"]:
        return None
    if state.get("run_finished"):
        return None
    root = Path(args.run_root)
    command = [args.python, "-m", "compose.experiments.v9_chain", *args.chain_args]
    if "--takeover" not in command:
        command.append("--takeover")
    _append_event(root, "RECOVERY_START", detail="restarting the chain", command=command)
    log_path = root / "monitor" / "chain_restart.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n===== {} restarting chain =====\n{}\n".format(_now(), " ".join(command)))
        process = subprocess.Popen(command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT)
    payload = {"pid": process.pid, "command": command, "log": str(log_path)}
    _append_event(root, "RECOVERY_COMPLETE", **payload)
    return payload


def transitions(previous: Optional[Mapping[str, Any]], state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Events that describe a *change* since the last check.

    Comparing with the previous cycle is what keeps the event file a record of
    what happened rather than a copy of the status file repeated once a minute.
    """
    events: List[Dict[str, Any]] = []
    before_tasks = (previous or {}).get("tasks") or {}
    after_tasks = state.get("tasks") or {}
    for key, task in sorted(after_tasks.items()):
        before = before_tasks.get(key) or {}
        if task.get("self_eval") and not before.get("self_eval"):
            events.append({"type": "SELF_EVAL_COMPLETE", "task": int(key),
                           "score": (state.get("self_eval_scores") or {}).get(key)})
        if task.get("train_phase_complete") and not before.get("train_phase_complete"):
            events.append({"type": "TASK_COMPLETE", "task": int(key)})
        checkpoints = task.get("checkpoints") or []
        if checkpoints and checkpoints != (before.get("checkpoints") or []):
            events.append({"type": "CHECKPOINT", "task": int(key), "checkpoint": checkpoints[-1]})
    heartbeat = state.get("heartbeat") or {}
    previous_heartbeat = (previous or {}).get("heartbeat") or {}
    if heartbeat.get("stage") in ("self_eval", "final_sweep"):
        if previous_heartbeat.get("stage") != heartbeat.get("stage"):
            events.append({"type": "SELF_EVAL_START", "task": heartbeat.get("task"),
                           "stage": heartbeat.get("stage")})
    return events


def run_check(args, idle_cycles: int, previous: Optional[Mapping[str, Any]]) -> int:
    root = Path(args.run_root)
    state = observe(args)
    state["run_finished"] = bool((root / "FORMAL_COMPLETE").is_file()
                                 or (root / "FORMAL_FAILED").is_file())

    # Five consecutive cycles of nothing happening anywhere.  GPU utilisation
    # counts as "something happening": a long calibration or a 3000-sample
    # evaluation uses the devices hard while writing no step metric.
    busy = any((device.get("utilization") or 0.0) > 5.0 for device in (state.get("gpus") or []))
    current = state.get("current") or {}
    moved = bool(previous) and current.get("rows") != ((previous or {}).get("current") or {}).get("rows")
    idle_cycles = 0 if (busy or moved or state["run_finished"]) else idle_cycles + 1
    state["idle_cycles"] = idle_cycles
    state["hard_stall"] = idle_cycles >= HARD_STALL_CYCLES

    verdict = evaluate_status(state)
    state["verdict"] = verdict
    recovery = maybe_recover(args, state, verdict)
    state["recovery"] = recovery
    _write_json(root / "monitor" / "watchdog_status.json", state)

    for event in transitions(previous, state):
        _append_event(root, event.pop("type"), **event)
    kind = verdict["verdict"]
    if kind == "FATAL_CORRECTNESS":
        _append_event(root, "FATAL", verdict=kind, detail=verdict["fatal"])
    elif kind in ("HEALTHY", "PROGRESS", "WARNING"):
        _append_event(root, "PROGRESS", verdict=kind, task=state.get("current_task"),
                      step=current.get("step"), stage=current.get("stage"),
                      rows=current.get("rows"), metric_age=state.get("metric_age"))
    if verdict["findings"]:
        _append_event(root, "WARNING", verdict=kind, findings=verdict["findings"])

    print(json.dumps({
        "iso": state["iso"], "verdict": kind, "task": state.get("current_task"),
        "step": current.get("step"), "stage": current.get("stage"), "rows": current.get("rows"),
        "metric_age": state.get("metric_age"), "ranks": state.get("ranks"),
        "heartbeat_age": state.get("heartbeat_age"),
        "chain_alive": not state.get("chain_dead"),
        "disk_free_gb": state.get("disk_free_gb"),
        "fatal": verdict["fatal"],
        "findings": [finding["detail"] for finding in verdict["findings"]],
    }, sort_keys=True), flush=True)
    return idle_cycles


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--interval", type=float, default=CHECK_INTERVAL)
    parser.add_argument("--once", action="store_true", help="run one check and exit")
    parser.add_argument("--no-auto-recover", dest="auto_recover", action="store_false", default=True,
                        help="report only; never start a chain")
    parser.add_argument("--chain-args", default="",
                        help="the chain's own arguments, for the restart path")
    args = parser.parse_args(argv)
    args.chain_args = [item for item in args.chain_args.split(" ") if item]

    root = Path(args.run_root)
    (root / "monitor").mkdir(parents=True, exist_ok=True)
    # One watchdog per run root, enforced by the kernel rather than by a
    # convention: a second watchdog would double every warning and race on the
    # event file.
    lock_path = root / "locks" / "watchdog.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another watchdog already holds {}".format(lock_path), file=sys.stderr)
        return 3
    _append_event(root, "WATCHDOG_START", interval=args.interval, pid=os.getpid())

    idle_cycles = 0
    previous: Optional[Mapping[str, Any]] = None
    try:
        while True:
            try:
                idle_cycles = run_check(args, idle_cycles, previous)
                previous = _read_json(root / "monitor" / "watchdog_status.json")
            except Exception as error:  # noqa: BLE001 - a watchdog must outlive its own bugs
                _append_event(root, "WARNING", verdict="WATCHDOG_ERROR",
                              detail="{}: {}".format(type(error).__name__, error))
                print("watchdog check failed: {}: {}".format(type(error).__name__, error),
                      file=sys.stderr, flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        _append_event(root, "WATCHDOG_STOP", reason="interrupted")
        return 0
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
