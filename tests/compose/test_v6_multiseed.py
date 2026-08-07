"""Regression tests for compose.eval.v6_multiseed aggregation.

The degenerate chain (task0 below_tau -> empty registry -> all evals
against the inherited task0 cold-start adapter) makes every seed's
matrix constant: MAA must equal MFT, BWT must be 0, and per-task
cross-seed std must be 0 when seeds share the pinned config seed.
"""

import statistics
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from compose.eval import v6_multiseed as agg  # noqa: E402


def _by_task(accuracies):
    return [
        {
            "task_id": i,
            "task": agg.TASK_NAMES[i],
            "metric_type": "Accuracy",
            "samples": 3000,
            "correct": int(3000 * acc),
            "accuracy": acc,
        }
        for i, acc in enumerate(accuracies)
    ]


def test_compute_aggregates_constant_matrix():
    """Two identical seeds -> zero std, MAA == MFT, BWT == 0."""
    rows = [
        _by_task([0.1643, 0.5420, 0.3858, 0.2037, 0.2000, 0.4217]),
        _by_task([0.1643, 0.5420, 0.3858, 0.2037, 0.2000, 0.4217]),
    ]
    per_task, metrics = agg.compute_aggregates([42, 43], rows)
    assert all(t["std"] == 0.0 for t in per_task)
    assert metrics["MAA"] == metrics["MFT"]
    assert metrics["BWT"] == 0.0
    assert metrics["MFT"] == pytest.approx(0.31958333333333333)
    assert metrics["MFN"] == 0.542


def test_compute_aggregates_single_seed():
    rows = [_by_task([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])]
    per_task, metrics = agg.compute_aggregates([42], rows)
    assert per_task[0]["mean"] == 0.1
    assert per_task[0]["std"] == 0.0  # no cross-seed variance
    assert metrics["MFT"] == 0.35
    assert metrics["MFN"] == 0.6


def test_std_across_seeds_aggregates_per_task_std():
    """Cross-seed spread must be derived from per-task stds, not from
    the spread across tasks (task-mean 31.96% vs task-std ~14.5% must
    not be confused with seed variance)."""
    rows = [
        _by_task([0.1643, 0.5420, 0.3858, 0.2037, 0.2000, 0.4217]),
        _by_task([0.1643, 0.5420, 0.3858, 0.2037, 0.2000, 0.4217]),
    ]
    per_task, metrics = agg.compute_aggregates([42, 43], rows)
    per_task_stds = [t["std"] for t in per_task]
    assert statistics.mean(per_task_stds) == 0.0
    assert all(s == 0.0 for s in per_task_stds)


def test_load_seed_summary_requires_six_tasks():
    """The aggregator must refuse summaries missing any of the six tasks
    (a truncated acceptance run must not aggregate silently)."""
    import json
    import tempfile


    from compose.eval.v6_multiseed import load_seed_summary

    with tempfile.TemporaryDirectory() as tmp:
        seed_dir = Path(tmp)
        payload = {"overall_accuracy": {"by_task": _by_task([0.1] * 5)}}
        (seed_dir / "final_summary.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        try:
            load_seed_summary(seed_dir)
        except AssertionError as exc:
            assert "six task metrics" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected AssertionError")
