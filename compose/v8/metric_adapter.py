"""Task-metric adapter: the ``M`` signal that decides ``solved``.

V8's hard gate is *task correctness*, never a loss threshold.  The official UCIT
scoring splits cleanly into two families:

* **Accuracy** (tasks 0, 1, 3, 4) -- ``llava/eval/eval_deepseek_r1.py:57``
  compares ``pred.upper() == ground_truth.upper()`` per sample and reports the
  percentage.  That is *exactly* per-sample decomposable, so the adapter can
  reproduce the official number bit-for-bit.
* **Average** (tasks 2, 5) -- the mean of Bleu_1..4 / METEOR / ROUGE_L / CIDEr.
  Those are corpus-level statistics; no per-sample decomposition reproduces
  them.  V8 therefore refuses to pretend: caption tasks are marked
  ``decomposable=False`` and the adapter reports an explicitly named
  training-time capability proxy instead, which must never be reported as the
  official metric.

``assert_matches_official`` is the acceptance gate for the decomposable family:
the adapter's own aggregate must equal the number the official scorer printed in
``Result.text``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


#: UCIT task index -> dataset name (matches the V7 run layout).
TASK_DATASETS: Dict[int, str] = {
    0: "ImageNet-R",
    1: "ArxivQA",
    2: "VizWiz",
    3: "IconQA",
    4: "CLEVR-Math",
    5: "Flickr30k",
}

#: Tasks the official scorer grades by case-insensitive exact match.
EXACT_MATCH_TASKS: Tuple[int, ...] = (0, 1, 3, 4)
#: Tasks the official scorer grades by the corpus-level caption Average.
CAPTION_TASKS: Tuple[int, ...] = (2, 5)

_RESULT_VALUE = re.compile(r"^\s*(Accuracy|Average)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?\s*$", re.M)


class MetricAdapterError(RuntimeError):
    """Raised when a metric cannot be evaluated the way V8 requires."""


@dataclass(frozen=True)
class TaskMetricSpec:
    task_id: int
    dataset: str
    metric_name: str
    decomposable: bool
    #: True when a larger value means a better answer.
    higher_is_better: bool = True
    #: Value at which a *single sample* counts as solved (decomposable tasks).
    solved_value: float = 1.0
    #: Name of the training-time proxy used when ``decomposable`` is False.
    proxy_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.metric_name not in ("Accuracy", "Average"):
            raise MetricAdapterError(f"unknown official metric {self.metric_name!r}")
        if not self.decomposable and self.proxy_name is None:
            raise MetricAdapterError(
                f"task {self.task_id} is not decomposable and must name an "
                "explicit capability proxy rather than reuse the official name"
            )


def default_specs() -> Dict[int, TaskMetricSpec]:
    specs: Dict[int, TaskMetricSpec] = {}
    for task_id, dataset in TASK_DATASETS.items():
        if task_id in EXACT_MATCH_TASKS:
            specs[task_id] = TaskMetricSpec(
                task_id=task_id,
                dataset=dataset,
                metric_name="Accuracy",
                decomposable=True,
                solved_value=1.0,
            )
        else:
            specs[task_id] = TaskMetricSpec(
                task_id=task_id,
                dataset=dataset,
                metric_name="Average",
                decomposable=False,
                proxy_name="v8_caption_capability_proxy_v1",
            )
    return specs


@dataclass
class AnswerScore:
    """One sample's task-metric outcome."""

    sample_id: str
    value: float
    solved: bool
    metric_name: str
    decomposable: bool


class TaskMetricAdapter:
    """Per-sample task correctness for the decomposable task family."""

    def __init__(self, specs: Optional[Mapping[int, TaskMetricSpec]] = None) -> None:
        self.specs: Dict[int, TaskMetricSpec] = dict(specs or default_specs())

    # ------------------------------------------------------------------
    def spec(self, task_id: int) -> TaskMetricSpec:
        try:
            return self.specs[int(task_id)]
        except KeyError as exc:
            raise MetricAdapterError(f"unknown task {task_id}") from exc

    def require_decomposable(self, task_id: int) -> TaskMetricSpec:
        spec = self.spec(task_id)
        if not spec.decomposable:
            raise MetricAdapterError(
                f"task {task_id} ({spec.dataset}) is graded by the corpus-level "
                f"{spec.metric_name} metric, which has no per-sample "
                "decomposition; V8 must not use it to decide `solved`"
            )
        return spec

    # ------------------------------------------------------------------
    def sample_value(self, task_id: int, prediction: str, ground_truth: str) -> float:
        """Per-sample metric value.

        For the Accuracy family this is ``1.0`` / ``0.0`` and is byte-identical
        to the official comparison (no extra stripping, same ``upper()``).
        """
        spec = self.require_decomposable(task_id)
        if spec.metric_name != "Accuracy":
            raise MetricAdapterError(f"unhandled metric {spec.metric_name!r}")
        left = "" if prediction is None else str(prediction)
        right = "" if ground_truth is None else str(ground_truth)
        return 1.0 if left.upper() == right.upper() else 0.0

    def score_sample(
        self, task_id: int, sample_id: str, prediction: str, ground_truth: str
    ) -> AnswerScore:
        spec = self.require_decomposable(task_id)
        value = self.sample_value(task_id, prediction, ground_truth)
        return AnswerScore(
            sample_id=str(sample_id),
            value=value,
            solved=bool(value >= spec.solved_value),
            metric_name=spec.metric_name,
            decomposable=True,
        )

    def is_solved(self, task_id: int, prediction: str, ground_truth: str) -> bool:
        """The `solved` predicate.  Metric only -- no loss, no threshold on NLL."""
        spec = self.require_decomposable(task_id)
        return bool(self.sample_value(task_id, prediction, ground_truth) >= spec.solved_value)

    # ------------------------------------------------------------------
    def aggregate(self, task_id: int, values: Sequence[float]) -> float:
        """Adapter-side aggregate on the official percentage scale."""
        self.require_decomposable(task_id)
        if not values:
            raise MetricAdapterError("cannot aggregate an empty result set")
        return 100.0 * float(sum(values)) / float(len(values))

    def assert_matches_official(
        self,
        task_id: int,
        values: Sequence[float],
        official_result_text: str,
        tolerance: float = 0.005,
    ) -> Dict[str, object]:
        """Gate: the adapter must reproduce the official scorer's own number."""
        spec = self.spec(task_id)
        expected = parse_result_value(official_result_text, spec.metric_name)
        actual = self.aggregate(task_id, values)
        if abs(actual - expected) > tolerance:
            raise MetricAdapterError(
                f"adapter aggregate {actual:.4f} does not reproduce the official "
                f"{spec.metric_name} {expected:.4f} for task {task_id}"
            )
        return {
            "task_id": int(task_id),
            "metric_name": spec.metric_name,
            "adapter_value": actual,
            "official_value": expected,
            "abs_delta": abs(actual - expected),
            "samples": len(values),
            "within_tolerance": True,
        }

    # ------------------------------------------------------------------
    def capability_proxy(self, task_id: int) -> Tuple[str, str]:
        """Explicitly named proxy for a non-decomposable task."""
        spec = self.spec(task_id)
        if spec.decomposable:
            raise MetricAdapterError(
                f"task {task_id} is decomposable; use sample_value, not a proxy"
            )
        return str(spec.proxy_name), spec.metric_name


def parse_result_value(result_text: str, metric_name: str) -> float:
    """Extract ``Accuracy: 66.00%`` / ``Average: 58.04`` from ``Result.text``."""
    matches = _RESULT_VALUE.findall(result_text or "")
    for name, value in matches:
        if name == metric_name:
            return float(value)
    raise MetricAdapterError(
        f"no {metric_name!r} line found in the official result text:\n{result_text!r}"
    )


def exact_match_values(
    adapter: TaskMetricAdapter,
    task_id: int,
    predictions: Mapping[str, str],
    ground_truth: Mapping[str, str],
    sample_ids: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Per-sample values keyed by sample id, order-insensitive."""
    ids = list(sample_ids) if sample_ids is not None else sorted(predictions)
    values: Dict[str, float] = {}
    for sample_id in ids:
        sample_id = str(sample_id)
        if sample_id not in ground_truth:
            raise MetricAdapterError(f"no ground truth for sample {sample_id}")
        if sample_id not in predictions:
            raise MetricAdapterError(f"no prediction for sample {sample_id}")
        values[sample_id] = adapter.sample_value(
            task_id, predictions[sample_id], ground_truth[sample_id]
        )
    return values


__all__ = [
    "AnswerScore",
    "CAPTION_TASKS",
    "EXACT_MATCH_TASKS",
    "MetricAdapterError",
    "TASK_DATASETS",
    "TaskMetricAdapter",
    "TaskMetricSpec",
    "default_specs",
    "exact_match_values",
    "parse_result_value",
]
