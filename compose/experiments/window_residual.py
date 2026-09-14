"""Characterise the answer-loss residual a micro-batch change leaves behind.

``compare_recipe --mode window`` answers *whether* two arms agree; when the
answer is no it reports the worst window and stops.  This module answers the
question that decides what the no means: **is the residual a diverging
trajectory or a stationary rounding floor?**  Those two are the same number at
any single window and completely different claims about the run, so the verdict
cannot be read off the comparator's output and needs its own measurement.

Three statistics separate them, and each one is here because it rules out a
specific reading of "MISMATCH":

``sign and bias``
    The mean of the *signed* relative residual, with its standard error.  A
    moved objective leaves a signed offset; rounding around the same objective
    leaves a mean that is zero to within noise.  Reported as a t-statistic,
    because a mean of -1e-3 on its own says nothing without the spread it came
    from.

``growth``
    The residual in consecutive blocks of windows, in *both* the relative and
    the absolute statistic.  A compounding divergence grows in the absolute
    statistic, which cannot be explained away.  A fixed arithmetic floor under a
    decaying loss grows in the relative statistic only, because its denominator
    is shrinking -- so the absolute column is what makes the relative column
    readable.  The two are printed together for that reason.

``extreme-value consistency``
    For a stationary zero-mean process of standard deviation sigma, the largest
    of ``n`` draws sits near ``sigma * sqrt(2 ln n)``.  If the observed worst
    window is far above that, the process is not stationary and the tolerance
    cannot be extrapolated to a longer run.  This -- not the six-step worst case
    -- is what licenses a tolerance for the brief's 100 / 500 / 1000 ladder, and
    the module prints the prediction for all three so the number in the report
    is a projection rather than a fit.

The comparator's own aggregation is imported rather than reimplemented: a second
implementation of "what one window's objective is" would be free to disagree
with the first, and the disagreement would be invisible.

Usage::

    python -m compose.experiments.window_residual \
        --arm S0=/run/s0/metrics --arm C=/run/c/metrics --steps 100 \
        --report docs/reports/data/recipe_ab_window_residual.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

from compose.experiments.compare_recipe import (
    _window_objective,
    load_arm,
    optimizer_windows,
)

#: Fields carried through the same analysis.  ``key_loss`` is included because
#: it is the objective the router itself is trained on: if a routing-relevant
#: discrepancy existed, this is the first place it would show, and it is
#: expected to be orders of magnitude quieter than the answer term.
FIELDS = ("answer_loss", "key_loss", "total_loss")

#: Window counts the brief's equivalence ladder asks about.  The extreme-value
#: prediction is projected to each so a tolerance can be chosen before the run
#: rather than after.
LADDER = (100, 500, 1000)


def _complete_windows(
    left: Dict[int, List[Dict[str, object]]],
    right: Dict[int, List[Dict[str, object]]],
    steps: int,
) -> Tuple[List[int], List[int]]:
    """Windows present, complete and shared by both arms.

    The arms are read from disk and the run may still be appending to them, so
    the final window of a live arm holds a *fraction* of its micro-steps.  Its
    row-mean is then an average over a different set of rows and differs from
    the complete window by whole percent -- a large, purely editorial artefact
    that would dominate a max-over-windows statistic.  Completeness is decided
    against each arm's own modal row count, which is what a full window looks
    like without having to be told the world size or the accumulation.
    """
    sizes: Dict[str, Dict[int, int]] = {}
    for label, arm in (("left", left), ("right", right)):
        sizes[label] = {index: len(rows) for index, rows in arm.items()}
    modal = {
        label: statistics.mode(list(counts.values()))
        for label, counts in sizes.items()
        if counts
    }
    common = sorted(set(sizes["left"]) & set(sizes["right"]))
    complete = [
        index
        for index in common
        if sizes["left"][index] == modal["left"]
        and sizes["right"][index] == modal["right"]
    ]
    if steps:
        complete = [index for index in complete if index < steps]
    dropped = [index for index in common if index not in complete and (not steps or index < steps)]
    return complete, dropped


def _series(
    left: Dict[int, List[Dict[str, object]]],
    right: Dict[int, List[Dict[str, object]]],
    indices: Sequence[int],
    field: str,
) -> List[Tuple[int, float, float, float, float]]:
    """``(step, left value, right value, relative residual, absolute residual)``."""
    out = []
    for index in indices:
        a = _window_objective(left[index], field)
        b = _window_objective(right[index], field)
        if a is None or b is None:
            continue
        out.append((index, a, b, (b - a) / a if a else 0.0, b - a))
    return out


def _blocks(series: Sequence[Tuple[int, float, float, float, float]], count: int = 5):
    """The series in equal consecutive blocks, for the growth reading."""
    if not series:
        return []
    width = max(1, len(series) // count)
    out = []
    for start in range(0, len(series), width):
        block = series[start : start + width]
        if not block:
            continue
        out.append(
            {
                "first_step": block[0][0],
                "last_step": block[-1][0],
                "n": len(block),
                "mean_abs_rel": statistics.mean(abs(row[3]) for row in block),
                "max_abs_rel": max(abs(row[3]) for row in block),
                "mean_abs_abs": statistics.mean(abs(row[4]) for row in block),
                "max_abs_abs": max(abs(row[4]) for row in block),
                "mean_left": statistics.mean(row[1] for row in block),
            }
        )
    return out


def analyse(
    series: Sequence[Tuple[int, float, float, float, float]],
    field: str,
) -> Dict[str, object]:
    """Every statistic the verdict rests on, for one field."""
    rel = [row[3] for row in series]
    absolute = [row[4] for row in series]
    n = len(rel)
    if n < 2:
        return {"field": field, "n": n}
    sd = statistics.pstdev(rel)
    mean = statistics.mean(rel)
    se = sd / math.sqrt(n)
    worst = max(abs(value) for value in rel)
    prediction = {str(k): sd * math.sqrt(2.0 * math.log(k)) for k in LADDER}
    half = n // 2
    return {
        "field": field,
        "n": n,
        "mean_rel": mean,
        "sd_rel": sd,
        "se_rel": se,
        "t_statistic": mean / se if se else None,
        "mean_abs_rel": statistics.mean(abs(value) for value in rel),
        "max_abs_rel": worst,
        "mean_abs_abs": statistics.mean(abs(value) for value in absolute),
        "max_abs_abs": max(abs(value) for value in absolute),
        "extreme_value_prediction": prediction,
        "extreme_value_ratio": worst / prediction[str(LADDER[0])] if prediction[str(LADDER[0])] else None,
        "first_half_mean_rel": statistics.mean(rel[:half]),
        "second_half_mean_rel": statistics.mean(rel[half:]),
        "half_drift_sigma": (
            (statistics.mean(rel[half:]) - statistics.mean(rel[:half]))
            / (sd / math.sqrt(half))
            if sd and half
            else None
        ),
        "blocks": _blocks(series),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--steps", type=int, default=0, help="windows to use (0 = all)")
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv)

    arms: Dict[str, str] = {}
    for spec in args.arm:
        label, _, path = spec.partition("=")
        arms[label] = path
    if len(arms) != 2:
        parser.error("exactly two --arm LABEL=PATH arguments are required")
    (label_a, path_a), (label_b, path_b) = list(arms.items())

    left = optimizer_windows(load_arm(path_a))
    right = optimizer_windows(load_arm(path_b))
    indices, dropped = _complete_windows(left, right, args.steps)

    result: Dict[str, object] = {
        "left": label_a,
        "right": label_b,
        "left_path": path_a,
        "right_path": path_b,
        "windows": len(indices),
        "dropped_incomplete": dropped,
        "fields": {},
    }
    for field in FIELDS:
        series = _series(left, right, indices, field)
        result["fields"][field] = analyse(series, field)
        result["fields"][field]["series"] = [
            {"step": row[0], "left": row[1], "right": row[2],
             "rel": row[3], "abs": row[4]}
            for row in series
        ]

    answer = result["fields"]["answer_loss"]
    print("%s vs %s: %d complete windows%s" % (
        label_a, label_b, len(indices),
        " (%d incomplete dropped)" % len(dropped) if dropped else ""))
    print("  answer_loss  mean rel %+.4e  sd %.4e  t=%.2f  max|rel| %.4e"
          % (answer["mean_rel"], answer["sd_rel"], answer["t_statistic"],
             answer["max_abs_rel"]))
    print("  stationary-Gaussian max|rel| prediction: "
          + "  ".join("n=%d %.3e" % (k, answer["extreme_value_prediction"][str(k)])
                      for k in LADDER))
    print("  observed/predicted(n=%d) = %.3f  (1.0 => stationary)"
          % (LADDER[0], answer["extreme_value_ratio"]))
    print("  blocks (rel / abs), which column grows:")
    for block in answer["blocks"]:
        print("    steps %3d-%3d  mean|rel| %.3e  mean|abs| %.3e  mean loss %.5f"
              % (block["first_step"], block["last_step"], block["mean_abs_rel"],
                 block["mean_abs_abs"], block["mean_left"]))
    key = result["fields"]["key_loss"]
    print("  key_loss     max|rel| %.4e  (the router's own objective)"
          % key["max_abs_rel"])

    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as handle:
            json.dump(result, handle, indent=1, sort_keys=True)
        print("wrote %s" % args.report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
