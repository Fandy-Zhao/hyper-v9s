"""DataLoader sufficiency, decided from the profiler and not from taste.

The V9 efficiency brief states the rule this module implements: a
``prefetch_factor`` / worker-count change is justified only when the measured
loader stall is material, and the threshold is 1% of the step.  Below it the
loader is not the bottleneck, so tuning it buys nothing measurable while adding
a knob whose value then has to be re-validated on every future run.

The module measures the fraction and returns a verdict.  It deliberately never
changes the loader: the decision and the code that would act on it are kept
apart so the reported number is evidence rather than a justification written
after the change.

Two sources, in order of preference:

* ``profile_steps.jsonl`` -- the full phase profiler (``--profile_training``),
  which separates ``data_wait_time`` (the gap between consecutive optimizer
  steps) from ``step_body_time`` and ``optimizer_time``.  This is the more
  informative source and is what the formal launcher enables.
* ``task*_train_steps.rank*.jsonl`` -- the trainer's own per-step log, which
  carries ``training_step_sec`` and ``inter_step_wait_sec`` whether or not the
  profiler is on.  A run that was already taken can therefore be judged without
  being repeated, which is what makes the rule checkable on existing evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

#: The brief's threshold, as a fraction of the optimizer-step period.
DEFAULT_THRESHOLD = 0.01


class DataWaitError(RuntimeError):
    pass


def load_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    """Read every JSON object from every path, skipping unreadable lines.

    A truncated final line is a normal consequence of killing a profiled run,
    and discarding it is right: the alternative is that the loader verdict
    cannot be computed for the run whose loader behaviour is in question.
    """
    rows: List[Dict[str, Any]] = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def measure(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Split the rows by which source they came from and total each.

    The two sources are not summed together: they measure the same seconds at
    different granularities, and adding a profiler row to a trainer row would
    count the wait twice.
    """
    profiler_wait = profiler_period = 0.0
    profiler_seen = 0
    trainer_wait = trainer_period = 0.0
    trainer_seen = 0
    for row in rows:
        if "data_wait_time" in row:
            wait = _finite(row.get("data_wait_time"))
            if wait is None:
                continue
            period = _finite(row.get("window_wall_time"))
            if period is None:
                period = sum(
                    value
                    for value in (
                        _finite(row.get("step_body_time")),
                        _finite(row.get("optimizer_time")),
                        wait,
                    )
                    if value is not None
                )
            profiler_wait += wait
            profiler_period += max(period, wait)
            profiler_seen += 1
        elif "inter_step_wait_sec" in row:
            wait = _finite(row.get("inter_step_wait_sec"))
            step = _finite(row.get("training_step_sec"))
            if wait is None or step is None:
                # The first micro-step of a run has no predecessor to have
                # waited for; it carries ``None`` rather than a zero, and
                # treating that as a measured zero would bias the fraction
                # downward by exactly one step.
                continue
            trainer_wait += wait
            trainer_period += step + wait
            trainer_seen += 1
    if profiler_seen:
        return {
            "source": "profiler",
            "rows": profiler_seen,
            "wait_sec": round(profiler_wait, 6),
            "period_sec": round(profiler_period, 6),
            "fraction": (profiler_wait / profiler_period) if profiler_period > 0 else 0.0,
        }
    if trainer_seen:
        return {
            "source": "trainer_steps",
            "rows": trainer_seen,
            "wait_sec": round(trainer_wait, 6),
            "period_sec": round(trainer_period, 6),
            "fraction": (trainer_wait / trainer_period) if trainer_period > 0 else 0.0,
        }
    raise DataWaitError(
        "no profiler row and no trainer step row with a measured wait; nothing "
        "to decide from (run with --profile_training, or point this at a run "
        "that wrote metrics/task*_train_steps.rank*.jsonl)"
    )


def verdict(fraction: float, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """Turn a measured fraction into the decision the brief prescribes."""
    below = fraction < float(threshold)
    return {
        "data_wait_fraction": round(float(fraction), 6),
        "threshold": float(threshold),
        "below_threshold": bool(below),
        "dataloader_change_warranted": not below,
        "decision": (
            "leave the DataLoader alone: the loader is not the bottleneck, and a "
            "prefetch_factor or worker-count change would be unmeasurable here"
            if below
            else "profile the loader before changing it: the measured stall is "
            "material and a prefetch/worker change is worth evaluating"
        ),
    }


def discover(root: Path) -> List[Path]:
    """Every profiler and trainer step log under a run root."""
    root = Path(root)
    paths = sorted(root.glob("**/profile_steps*.jsonl"))
    paths += sorted(root.glob("**/metrics/task*_train_steps*.jsonl"))
    return paths


def report(root: Path, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    paths = discover(root)
    if not paths:
        raise DataWaitError("no profiler or step metrics under {}".format(root))
    payload = measure(load_rows(paths))
    payload.update(verdict(payload["fraction"], threshold))
    payload["root"] = str(root)
    payload["sources"] = [str(path) for path in paths]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="a run root, or one metrics file")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    target = Path(args.root)
    if target.is_file():
        payload = measure(load_rows([target]))
        payload.update(verdict(payload["fraction"], args.threshold))
        payload["root"] = str(target)
    else:
        payload = report(target, args.threshold)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(
        "data_wait {fraction:.4%} of the step over {rows} rows "
        "({source}): {decision}".format(
            fraction=float(payload["fraction"]),
            rows=payload["rows"],
            source=payload["source"],
            decision=payload["decision"],
        )
    )


if __name__ == "__main__":
    main()
