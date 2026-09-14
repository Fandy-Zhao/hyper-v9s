"""Retrieval audits and the V8-A comparison metrics.

The failure this module exists to prevent is a misdiagnosis.  If the Multi-Key
Router never recalls the expert that could have solved a sample, the teacher
cannot possibly discover that expert -- and V8 would be blamed for a
*retrieval* failure while the real cause is a *capability* failure (or the other
way round).  PART 26 therefore requires a **full-pool single audit** that skips
the Top-M limit entirely and evaluates every historical expert, so the two
hypotheses can be told apart:

* full-pool oracle finds a solver, Top-M recall misses it
  -> retrieval / key problem (V8's alias keys can help)
* full-pool oracle finds no solver at all
  -> genuine capability gap (aliases cannot help; expert formation must change)

``GapClosed`` and the recall curves below are the numbers PART 35 asks for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


EPSILON = 1.0e-9


def _as_float(value: Any) -> float:
    return float(value)


@dataclass
class RecallCurve:
    ks: Tuple[int, ...]
    teacher_expert_recall: Dict[int, float]
    full_pool_recall: Dict[int, float]
    samples: int
    oracle_expert_pairs: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ks": list(self.ks),
            "teacher_expert_recall_at": {f"@{k}": v for k, v in self.teacher_expert_recall.items()},
            "full_pool_oracle_recall_at": {f"@{k}": v for k, v in self.full_pool_recall.items()},
            "samples": self.samples,
            "oracle_expert_pairs": self.oracle_expert_pairs,
        }


def teacher_expert_recall_at_k(
    solved_experts_by_sample: Mapping[str, Iterable[int]],
    recall_map: Mapping[str, Sequence[int]],
    ks: Sequence[int] = (1, 2, 4, 8),
) -> Dict[int, float]:
    """Fraction of *oracle solving experts* that appear in the Top-K recall.

    This is the metric that separates "the expert is not reusable" from "the
    router did not find it".  When a sample has several solving experts, every
    one of them counts -- missing any of them is lost capability.
    """
    totals = {int(k): 0 for k in ks}
    hits = {int(k): 0 for k in ks}
    for sample_id, solved in solved_experts_by_sample.items():
        solved_set = {int(value) for value in solved}
        if not solved_set:
            continue
        recall = [int(value) for value in recall_map.get(str(sample_id), [])]
        for k in ks:
            k = int(k)
            totals[k] += len(solved_set)
            hits[k] += len(solved_set.intersection(recall[:k]))
    return {k: (hits[k] / totals[k] if totals[k] else 0.0) for k in ks}


def full_pool_oracle_recall_at_k(
    solved_experts_by_sample: Mapping[str, Iterable[int]],
    ranked_experts_by_sample: Mapping[str, Sequence[int]],
    ks: Sequence[int] = (1, 2, 4, 8),
) -> Dict[int, float]:
    """Recall of the *full-pool* oracle ranking, i.e. how deep the router must go.

    ``ranked_experts_by_sample`` is the key-similarity ranking over **all**
    active experts (no Top-M cut), so this curve shows whether the solving expert
    is reachable at all within a larger budget.
    """
    totals = {int(k): 0 for k in ks}
    hits = {int(k): 0 for k in ks}
    for sample_id, solved in solved_experts_by_sample.items():
        solved_set = {int(value) for value in solved}
        if not solved_set:
            continue
        ranked = [int(value) for value in ranked_experts_by_sample.get(str(sample_id), [])]
        for k in ks:
            k = int(k)
            totals[k] += len(solved_set)
            hits[k] += len(solved_set.intersection(ranked[:k]))
    return {k: (hits[k] / totals[k] if totals[k] else 0.0) for k in ks}


def teacher_set_exact_accuracy(
    teacher: Mapping[str, Sequence[int]],
    oracle: Mapping[str, Sequence[int]],
) -> float:
    """Fraction of samples where the teacher's set equals the oracle's best set."""
    if not teacher:
        return 0.0
    matches = 0
    for sample_id, chosen in teacher.items():
        expected = oracle.get(str(sample_id))
        if expected is None:
            continue
        if sorted(int(v) for v in chosen) == sorted(int(v) for v in expected):
            matches += 1
    return matches / len(teacher)


def historical_reuse_rate(
    state_by_sample: Mapping[str, str],
    expert_origin_task: Mapping[int, int],
    selected_by_sample: Mapping[str, Sequence[int]],
    current_task: int,
) -> Dict[str, float]:
    """Share of samples whose *chosen* experts come from an earlier task."""
    if not state_by_sample:
        return {"samples": 0, "reuse_samples": 0, "historical_reuse_rate": 0.0}
    reuse = 0
    total = 0
    for sample_id, state in state_by_sample.items():
        chosen = [int(value) for value in selected_by_sample.get(str(sample_id), [])]
        if not chosen:
            continue
        total += 1
        if any(int(expert_origin_task.get(expert_id, current_task)) != int(current_task)
               for expert_id in chosen):
            reuse += 1
    return {
        "samples": total,
        "reuse_samples": reuse,
        "historical_reuse_rate": (reuse / total) if total else 0.0,
    }


def gap_closed(v7_metric: float, v8_metric: float, teacher_metric: float) -> Dict[str, float]:
    """``GapClosed = (V8 - V7) / max(Teacher - V7, eps)`` plus raw numbers.

    Reported with the absolute quantities as well, because a large ratio over a
    negligible gap is not evidence of anything.
    """
    teacher_gap = float(teacher_metric) - float(v7_metric)
    v8_gain = float(v8_metric) - float(v7_metric)
    closed = v8_gain / max(teacher_gap, EPSILON) if teacher_gap > 0 else (1.0 if v8_gain > 0 else 0.0)
    return {
        "v7_metric": float(v7_metric),
        "v8_metric": float(v8_metric),
        "teacher_metric": float(teacher_metric),
        "teacher_minus_v7": teacher_gap,
        "v8_minus_v7": v8_gain,
        "gap_closed": closed,
        "teacher_minus_v8": float(teacher_metric) - float(v8_metric),
    }


def winning_key_distribution(route_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Origin-vs-alias split of the keys that actually fired (PART 35).

    ``route_rows`` are :meth:`compose.v8.routing.MultiKeyRouteResult.as_rows`
    entries whose ``key_ids`` name the winning key per slot.
    """
    origin = 0
    alias = 0
    per_task: Dict[str, int] = {}
    per_expert: Dict[str, int] = {}
    for row in route_rows:
        for key_id in row.get("key_ids", []) or []:
            if key_id is None:
                continue
            # ``key_type`` carries its own ``_`` in ``task_alias``, so bound the
            # split instead of counting segments -- counting silently dropped
            # every origin key (``e3_t7_origin`` has three), which pinned
            # ``origin_key_wins`` at 0 and ``alias_win_rate`` at 1.0.
            parts = str(key_id).split("_", 2)
            if len(parts) != 3:
                continue
            key_type = parts[2]
            task_id = parts[1]
            expert_id = parts[0]
            if key_type == "origin":
                origin += 1
            elif key_type == "task_alias":
                alias += 1
            else:
                continue
            per_task[task_id] = per_task.get(task_id, 0) + 1
            per_expert[expert_id] = per_expert.get(expert_id, 0) + 1
    total = origin + alias
    return {
        "origin_key_wins": origin,
        "alias_key_wins": alias,
        "alias_win_rate": (alias / total) if total else 0.0,
        "wins_per_key_task": dict(sorted(per_task.items())),
        "wins_per_expert": dict(sorted(per_expert.items())),
    }


def routing_agreement(
    actual: Mapping[str, Sequence[int]],
    expected: Mapping[str, Sequence[int]],
) -> Dict[str, float]:
    """Per-sample exact-set agreement between two routing policies."""
    if not actual:
        return {"samples": 0, "exact_set_agreement": 0.0, "top1_agreement": 0.0}
    exact = 0
    top1 = 0
    compared = 0
    for sample_id, chosen in actual.items():
        other = expected.get(str(sample_id))
        if other is None:
            continue
        compared += 1
        left = sorted(int(v) for v in chosen)
        right = sorted(int(v) for v in other)
        if left == right:
            exact += 1
        if left and right and left[0] == right[0]:
            top1 += 1
    return {
        "samples": compared,
        "exact_set_agreement": (exact / compared) if compared else 0.0,
        "top1_agreement": (top1 / compared) if compared else 0.0,
    }


def diagnose(
    full_pool_oracle_solves: int,
    teacher_positives: int,
    candidate_recall: float,
    v7_metric: Optional[float],
    v8_metric: float,
    samples: int,
) -> Dict[str, Any]:
    """Classify the V8-A outcome into the four cases of PART 36.

    Three of the four cases are statements *about V7* -- "the router misses the
    experts" only means something if the recalled ones are compared against a
    baseline.  ``v7_metric`` is therefore optional, and ``None`` means the
    baseline was measured on a different sample set and cannot be compared
    (see ``V8TaskRun.analyse``).  In that situation only the capability question
    is answerable, so a run with capability and no baseline is reported as
    ``CASE_UNCLASSIFIED_NO_V7_BASELINE`` rather than being pushed into CASE_B or
    CASE_D by comparison with a fabricated zero.
    """
    if samples <= 0:
        raise ValueError("diagnose needs a positive sample count")
    oracle_rate = full_pool_oracle_solves / samples
    if full_pool_oracle_solves == 0:
        case, interpretation = "CASE_C", (
            "the full-pool teacher finds essentially no reusable historical "
            "capability: the bottleneck is expert formation, not the router"
        )
    elif v7_metric is None:
        case, interpretation = "CASE_UNCLASSIFIED_NO_V7_BASELINE", (
            "reusable historical capability exists, but this run has no "
            "comparable V7 baseline (the diagnostic was measured on a different "
            "sample set), so the A/B/D cases -- which are all comparisons "
            "against V7 -- are not decidable from it"
        )
    elif candidate_recall < 0.5:
        case, interpretation = "CASE_A", (
            "reusable experts exist but the router misses them: this is a "
            "retrieval/key-representation problem, which alias keys target"
        )
    elif v8_metric <= v7_metric + EPSILON:
        case, interpretation = "CASE_B", (
            "reusable experts exist and are recalled, yet the alias keys do not "
            "convert that into a better metric: fixed-query geometry or key "
            "learning is still insufficient"
        )
    else:
        case, interpretation = "CASE_D_OR_IMPROVED", (
            "recall is high and the metric improved; check train-vs-validation "
            "recall for alias-key overfitting before concluding"
        )
    return {
        "case": case,
        "interpretation": interpretation,
        "full_pool_oracle_solve_rate": oracle_rate,
        "teacher_positives": int(teacher_positives),
        "candidate_recall": float(candidate_recall),
        "v7_metric": None if v7_metric is None else float(v7_metric),
        "v7_metric_comparable": v7_metric is not None,
        "v8_metric": float(v8_metric),
    }


__all__ = [
    "EPSILON",
    "RecallCurve",
    "diagnose",
    "full_pool_oracle_recall_at_k",
    "gap_closed",
    "historical_reuse_rate",
    "routing_agreement",
    "teacher_expert_recall_at_k",
    "teacher_set_exact_accuracy",
    "winning_key_distribution",
]
