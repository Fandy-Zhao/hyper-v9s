"""Per-task stage budget of a frozen V8 continual-learning run.

This is the source of the ``train / RMS / prune`` columns in
``docs/reports/V8_SPEEDUP_REPORT.md`` section 3.  It exists because those columns
were previously hand-derived and drifted by 0.01 h in a few cells; generating
them makes each one checkable against the run's own artifacts.

Two different clocks are involved, and they must not be confused:

* **Training** is measured by the trainer itself: ``train_runtime`` in
  ``logs/training.log``.  It starts after the model is loaded and the dataset is
  built, so it excludes startup -- which is the honest number for a speedup
  claim, since startup is not what the acceleration touches.
* **RMS and pruning** are wall-clock differences between stage-boundary
  artifacts, exactly as ``V8_CURRENT_RUNTIME_AUDIT.md`` section 3.1 defines
  them: ``s4_rms.done - s3_training.done`` and
  ``s5_pruning_commit.done - s4_rms.done``.

The second clock needs a guard.  A pipeline that is repaired and resumed
re-stamps the markers it had already written, and on this run that happened
twice: task 0's ``s0``-``s3`` markers all carry one mtime, and task 5's
``s0``-``s4`` all carry another.  Those stamps are later than the work they
claim to bound (task 0's ``s3`` is ~4.9 h *after* its training log stopped), so
using them silently shortens or lengthens a stage by hours.  The guard is a
sanity bound: a stage marker is trusted only if it lands within
``--marker-tolerance`` of the matching stage log's mtime; otherwise the log's
mtime is used for that boundary.  The tolerance is 600 s against typical gaps
of 4-6 s.

Usage::

    python -m compose.experiments.stage_budget <run-root> [--json out.json]

``<run-root>`` is a run directory holding ``task0`` .. ``task5``, e.g.
``/data/ckpt/.../v7_gpu01_cached_query_formal_20260903``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TASKS = range(6)

#: A stage marker further than this from its own stage log is a re-stamp.
MARKER_TOLERANCE_S = 600.0

#: The fixed-ladder training ratio of the shipped V8-Exact-Accelerated config.
#: Every accelerated `train` cell is `train / TRAIN_RATIO`; RMS and pruning are
#: left at their original values because the acceleration does not touch them.
TRAIN_RATIO = 2.258


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _train_runtime(log: Path) -> float | None:
    """Seconds reported by the HF Trainer, or None.

    ``train_runtime`` appears in the final metrics dict the trainer dumps; it is
    also echoed to the log tail.  Take the last occurrence.
    """
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return None
    hits = re.findall(r"train_runtime['\"]?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text)
    return float(hits[-1]) if hits else None


def stage_budget(run_root: Path, tolerance: float = MARKER_TOLERANCE_S) -> dict:
    """Return the per-task budget, in hours, plus the endpoint provenance."""
    tasks = []
    for t in TASKS:
        d = run_root / f"task{t}"
        stages = d / "stages"
        train_log = d / "logs" / "training.log"
        rms_log = d / "logs" / "rms.log"

        train_runtime = _train_runtime(train_log)
        train_end_log = _mtime(train_log)
        rms_end_log = _mtime(rms_log)
        s3 = _mtime(stages / "s3_training.done")
        s4 = _mtime(stages / "s4_rms.done")
        s5 = _mtime(stages / "s5_pruning_commit.done")

        row = {"task": t, "sources": {}}

        # -- training: the trainer's own clock -------------------------------
        if train_runtime is None or train_end_log is None:
            row["error"] = "no training.log / train_runtime"
            tasks.append(row)
            continue
        row["train_s"] = train_runtime
        row["sources"]["train"] = "trainer train_runtime"

        # -- start of RMS: end of the training stage -------------------------
        if s3 is not None and abs(s3 - train_end_log) <= tolerance:
            rms_start, row["sources"]["rms_start"] = s3, "s3_training.done"
        else:
            rms_start, row["sources"]["rms_start"] = train_end_log, "training.log mtime (s3 re-stamped)"

        # -- end of RMS: s4_rms.done, else the RMS log ------------------------
        if s4 is not None and rms_end_log is not None and abs(s4 - rms_end_log) <= tolerance:
            rms_end, row["sources"]["rms_end"] = s4, "s4_rms.done"
        elif rms_end_log is not None:
            rms_end, row["sources"]["rms_end"] = rms_end_log, "rms.log mtime (s4 re-stamped)"
        else:
            rms_end, row["sources"]["rms_end"] = s4, "s4_rms.done (no rms.log)"

        if rms_start is None or rms_end is None:
            row["error"] = "missing RMS boundary"
            tasks.append(row)
            continue
        row["rms_s"] = rms_end - rms_start

        # -- pruning: s4 -> s5 -----------------------------------------------
        if s5 is None or rms_end is None:
            row["error"] = "missing s5_pruning_commit.done"
            tasks.append(row)
            continue
        row["prune_s"] = s5 - rms_end
        row["sources"]["prune_end"] = "s5_pruning_commit.done"

        for key, secs in (
            ("train_h", row["train_s"]),
            ("rms_h", row["rms_s"]),
            ("prune_h", row["prune_s"]),
        ):
            row[key] = secs / 3600.0
        row["total_h"] = row["train_h"] + row["rms_h"] + row["prune_h"]
        row["train_acc_h"] = row["train_h"] / TRAIN_RATIO
        row["total_acc_h"] = row["train_acc_h"] + row["rms_h"] + row["prune_h"]
        row["speedup"] = row["total_h"] / row["total_acc_h"]
        tasks.append(row)

    ok = [r for r in tasks if "error" not in r]
    totals = {
        k: sum(r[k] for r in ok)
        for k in ("train_h", "rms_h", "prune_h", "total_h", "train_acc_h", "total_acc_h")
    }
    if totals["total_acc_h"]:
        totals["speedup"] = totals["total_h"] / totals["total_acc_h"]
    return {"run_root": str(run_root), "train_ratio": TRAIN_RATIO, "tasks": tasks, "totals": totals}


def _fmt(row: dict) -> str:
    if "error" in row:
        return f"| task{row['task']} | {row['error']} |"
    return (
        f"| task{row['task']} | {row['train_h']:.2f} | {row['rms_h']:.2f} | "
        f"{row['prune_h']:.2f} | {row['total_h']:.2f} | {row['train_acc_h']:.2f} | "
        f"{row['total_acc_h']:.2f} | {row['speedup']:.2f} |"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_root", type=Path)
    ap.add_argument("--json", type=Path, default=None, help="also write the raw result here")
    ap.add_argument("--marker-tolerance", type=float, default=MARKER_TOLERANCE_S)
    args = ap.parse_args()

    result = stage_budget(args.run_root, args.marker_tolerance)

    print("| task | train | RMS | prune | **total** | train' | **total'** | **x** |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in result["tasks"]:
        print(_fmt(row))
    t = result["totals"]
    print(
        f"| **full run** | **{t['train_h']:.2f}** | **{t['rms_h']:.2f}** | "
        f"**{t['prune_h']:.2f}** | **{t['total_h']:.2f}** | **{t['train_acc_h']:.2f}** | "
        f"**{t['total_acc_h']:.2f}** | **{t['speedup']:.2f}** |"
    )
    print()
    print("boundaries used:")
    for row in result["tasks"]:
        print(f"  task{row['task']}: " + ", ".join(f"{k}={v}" for k, v in row["sources"].items()))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
