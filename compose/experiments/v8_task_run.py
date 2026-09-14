"""Formal V8 task entry point; V7 is reused only as its numerical kernel."""
from __future__ import annotations

import argparse
import sys

from compose.v8.metric_adapter import TaskMetricAdapter


METHOD_SUMMARY = """================ V8 FORMAL METHOD ================
Method:
  Few-shot Answer-Supervised Historical Screening
  + Full-data Query-only Global Top-2 Co-evolution
Teacher: train subset; full historical single + bounded pair; NLL ranking only
Full Training: ALL samples; Oracle evaluations: 0
  Routing supervision: Query/Key only
  Teacher assignments in full training: None
  Reuse-key adaptation: Routed answer-quality weighted
  Historical LoRA trainable: False
  Historical committed keys trainable: False
  Reusable current-task keys trainable: True
  Candidate LoRA trainable: True
  Candidate keys trainable: True
Routing: Fixed query; Multi-key max-per-expert; Distinct expert Top-2
=================================================="""


def _value(argv, name):
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _check_tasks(tasks):
    adapter, failures = TaskMetricAdapter(), []
    for task in sorted(set(tasks)):
        if task:
            try:
                adapter.require_decomposable(task)
            except Exception as error:
                failures.append("Task{}: {}".format(task, error))
    if failures:
        raise SystemExit(
            "V8 formal preflight failed: no verified per-sample Teacher capability proxy for:\n  {}\n"
            "Use TASKS=0,1,3,4 until a validated proxy is implemented."
            .format("\n  ".join(failures))
        )


def main():
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--check-teacher-support", action="store_true")
    parser.add_argument("--tasks", default="0,1,2,3,4,5")
    known, _ = parser.parse_known_args(argv)
    if known.check_teacher_support:
        try:
            tasks = [int(x) for x in known.tasks.split(",") if x.strip()]
        except ValueError as error:
            raise SystemExit("--tasks must be comma-separated task indices") from error
        _check_tasks(tasks)
        print("V8 teacher capability preflight passed for tasks {}".format(tasks))
        return
    if "--help" in argv or "-h" in argv:
        print(METHOD_SUMMARY)
        from compose.experiments.v7_task_run import main as kernel_main
        kernel_main()
        return
    missing = [x for x in ("--compose-v8-config", "--compose-v8-query-cache-root", "--task-index") if x not in argv]
    if missing:
        raise SystemExit("V8 formal task run requires {}".format(", ".join(missing)))
    try:
        task = int(_value(argv, "--task-index"))
    except (TypeError, ValueError) as error:
        raise SystemExit("--task-index must be an integer") from error
    num, ratio = _value(argv, "--v8-teacher-num-samples"), _value(argv, "--v8-teacher-sample-ratio")
    if num is not None and ratio is not None:
        raise SystemExit("set exactly one Teacher subset control")
    if task == 0 and (num is not None or ratio is not None):
        raise SystemExit("Task0 has no history: do not configure a Teacher subset")
    if task > 0:
        if num is None and ratio is None:
            raise SystemExit("Task{} requires explicit bounded Teacher subset".format(task))
        _check_tasks([task])
    print(METHOD_SUMMARY, flush=True)
    from compose.experiments.v7_task_run import main as kernel_main
    kernel_main()


if __name__ == "__main__":
    main()
