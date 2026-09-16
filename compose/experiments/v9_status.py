"""Print the real state of a V9-S run, read from its artefacts.

    python -m compose.experiments.v9_status --root RUN_ROOT

This answers the question "where is the run actually up to" without reading a
report that may describe an older attempt, without a shell one-liner per fact,
and without trusting the run's own summary.  Every line comes from an artefact
on disk or the process table; nothing is inferred from a previous answer.

It reads through the same :func:`compose.experiments.v9_watchdog.observe` the
watchdog uses, so the status a person sees and the status the watchdog acts on
cannot disagree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from compose.experiments.v9_watchdog import evaluate_status, observe
from compose.v9.closure import TASK_NAMES, task_completion


def _gb(value: Any) -> str:
    return "n/a" if value is None else "{:.1f}G".format(value)


def _age(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)):
        return "n/a"
    if seconds < 90:
        return "{:.0f}s".format(seconds)
    if seconds < 5400:
        return "{:.1f}m".format(seconds / 60.0)
    return "{:.1f}h".format(seconds / 3600.0)


def render(args) -> Dict[str, Any]:
    # ``observe`` is written against the watchdog's own argument namespace; this
    # entry point names the same thing ``--root`` because that is what every
    # other tool in this directory calls it.  One reading path, two spellings.
    args.run_root = getattr(args, "run_root", None) or args.root
    state = observe(args)
    verdict = evaluate_status(state)
    state["verdict"] = verdict
    return state


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="a V9-S run root")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--json", action="store_true", help="the same readings, as JSON")
    args = parser.parse_args(argv)

    state = render(args)
    if args.json:
        print(json.dumps(state, indent=2, sort_keys=True, default=str))
        return 0

    verdict = state["verdict"]
    root = Path(args.root)
    print("V9-S status  {}".format(root))
    print("  git SHA        {}   (recorded {}, {})".format(
        state.get("git_sha_now") or "unknown",
        state.get("git_sha_recorded") or "unrecorded",
        "MISMATCH" if state.get("git_sha_mismatch") else "match",
    ))
    print("  markers        {}".format(", ".join(state.get("markers") or []) or "(none)"))
    print("  supervisor     {} process(es) {}".format(
        state.get("supervisors"), state.get("supervisor_pids") or ""))
    print("  trainer        {} rank(s)  torchrun {}".format(
        state.get("ranks"), state.get("torchrun_pids") or []))
    print("  current        task {} ({})  step {}  stage {}  rows {}  metric age {}".format(
        state.get("current_task"),
        TASK_NAMES[state["current_task"]] if state.get("current_task") is not None else "-",
        (state.get("current") or {}).get("step"),
        (state.get("current") or {}).get("stage"),
        (state.get("current") or {}).get("rows"),
        _age(state.get("metric_age")),
    ))
    print("  heartbeat      {} ago  {}".format(
        _age(state.get("heartbeat_age")),
        (state.get("heartbeat") or {}).get("stage") or "(never beat)",
    ))
    print("  disk free      {}".format(_gb(state.get("disk_free_gb"))))
    print("  verdict        {}{}".format(
        verdict["verdict"],
        "  " + "; ".join(verdict["fatal"]) if verdict["fatal"] else "",
    ))

    print("\n  GPU      util   memory      temp   power")
    for device in state.get("gpus") or []:
        print("  {index:<8} {util:>4.0f}%  {mem:>8.0f}M  {temp:>4.0f}C  {power:>5.0f}W".format(
            index=device["index"], util=device["utilization"], mem=device["memory_used_mib"],
            temp=device["temperature_c"], power=device["power_w"],
        ))
    if not state.get("gpus"):
        print("  (nvidia-smi unavailable)")

    print("\n  task  name         rows   step   ckpts  self-eval  train-phase  metric age")
    for index in range(len(TASK_NAMES)):
        task = (state.get("tasks") or {}).get(str(index))
        if task is None:
            print("  {:<4}  {:<11}  (not started)".format(index, TASK_NAMES[index]))
            continue
        print("  {:<4}  {:<11}  {:>4}  {:>5}  {:>5}  {:>9}  {:>11}  {:>10}".format(
            index, TASK_NAMES[index], task.get("rows") or 0, task.get("step") or "-",
            len(task.get("checkpoints") or []),
            "yes" if task.get("self_eval") else "no",
            "yes" if task.get("train_phase_complete") else "no",
            _age(task.get("metric_age")),
        ))

    for index in range(len(TASK_NAMES)):
        task_root = root / "task{}".format(index)
        if not task_root.is_dir():
            continue
        completion = task_completion(
            task_root, index, require_full_coverage=False, require_eval=False,
            eval_root=root / "evaluation_matrix",
        )
        if completion["complete"]:
            continue
        print("\n  task {} outstanding: {}".format(index, ", ".join(completion["failed"]) or "nothing"))

    if verdict["findings"]:
        print("\n  findings")
        for finding in verdict["findings"]:
            print("    [{}] {}".format(finding["level"], finding["detail"]))

    errors = state.get("recent_errors") or []
    if errors:
        print("\n  last error (log last written {} ago)".format(_age(errors[-1]["age_seconds"])))
        for line in errors[-1]["context"].splitlines()[-6:]:
            print("    {}".format(line))
    else:
        print("\n  last error     none in the last {} minutes".format(
            int(3 * 120 / 60)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
