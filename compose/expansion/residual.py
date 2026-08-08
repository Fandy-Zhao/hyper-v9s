"""Residual splitting driven by the answer-supervised old-expert teacher.

New core definition (replaces the old gain-floor rule):

    old_teacher_loss = loss(S_old_star)     # answer NLL of the best
                                             # empty/single/pair old set

    is_residual = old_teacher_loss > tau_res

A sample whose best old-expert teacher set still leaves a loss above
``tau_res`` is residual material (old experts did not explain it). A
sample at or below the threshold is reused.

Rules:

- ``tau_res`` is never lowered to force new experts;
- an empty expert pool is a legal cold start: every training sample is
  residual material for the first task (no rejected/rebootstrap concept);
- a historical Top-M recall miss is recorded as a retrieval diagnostic,
  never wrapped into "need a new capability expert".

Outputs: reuse train, residual train. Validation residuals never update
cluster training.
"""

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from compose.teacher.types import stable_hash

RESIDUAL_SCHEMA_VERSION = 1

VALID_SPLITS = ("train", "val")
RESIDUAL_REASON_BELOW_TAU = "old_teacher_loss_below_tau_res"  # reuse
RESIDUAL_REASON_ABOVE_TAU = "old_teacher_loss_above_tau_res"  # residual
RESIDUAL_REASON_COLD_START = "cold_start_empty_pool"  # residual (task 0)
RESIDUAL_REASON_RECALL_MISS = "recall_miss_diagnostic"  # diagnostic only


@dataclass(frozen=True)
class ComposeResidualRecord:
    sample_id: str
    task_id: int
    old_teacher_set: Tuple[int, ...]
    empty_loss: float
    old_teacher_loss: float
    residual_reason: str
    split: str
    retrieved_top_m: Tuple[int, ...] = ()
    teacher_multi_hot: Dict[int, int] = field(default_factory=dict)
    retrieval_diagnostic: bool = False

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id is required")
        if self.split not in VALID_SPLITS:
            raise ValueError("split must be one of {}".format(VALID_SPLITS))
        if self.residual_reason not in (
            RESIDUAL_REASON_BELOW_TAU,
            RESIDUAL_REASON_ABOVE_TAU,
            RESIDUAL_REASON_COLD_START,
            RESIDUAL_REASON_RECALL_MISS,
        ):
            raise ValueError(
                "unknown residual_reason: {!r}".format(self.residual_reason)
            )
        if not all(value >= 0 for value in (self.empty_loss, self.old_teacher_loss)):
            raise ValueError("losses must be non-negative")
        object.__setattr__(
            self,
            "teacher_multi_hot",
            {int(key): int(value) for key, value in self.teacher_multi_hot.items()},
        )
        object.__setattr__(
            self, "retrieved_top_m", tuple(int(value) for value in self.retrieved_top_m)
        )

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["old_teacher_set"] = list(self.old_teacher_set)
        data["retrieved_top_m"] = list(self.retrieved_top_m)
        return data

    @classmethod
    def from_dict(cls, value: Dict[str, object]) -> "ComposeResidualRecord":
        data = dict(value)
        data["old_teacher_set"] = tuple(int(item) for item in data["old_teacher_set"])
        data["retrieved_top_m"] = tuple(
            int(item) for item in data.get("retrieved_top_m", ())
        )
        return cls(**data)


def is_residual(
    old_teacher_loss: float,
    tau_res: float,
    empty_pool: bool = False,
    recall_covered: bool = True,
) -> Tuple[bool, str]:
    """Core residual judgment.

    ``empty_pool=True`` (task 0 / cold start): every sample is residual
    material (the frozen backbone is the current system and nothing old
    could explain the sample).

    ``recall_covered=False`` marks a retrieval diagnostic: the Top-M missed
    the contributing expert. It is recorded, never turned into a new
    capability expert.
    """
    if empty_pool:
        return True, RESIDUAL_REASON_COLD_START
    if not recall_covered:
        return False, RESIDUAL_REASON_RECALL_MISS
    if old_teacher_loss > tau_res:
        return True, RESIDUAL_REASON_ABOVE_TAU
    return False, RESIDUAL_REASON_BELOW_TAU


def build_residual_records(
    teacher_records: Sequence["Any"],
    tau_res: float,
    split: str,
    top_m_covered: Optional[Sequence[bool]] = None,
    empty_pool: bool = False,
) -> Tuple[List[ComposeResidualRecord], List[ComposeResidualRecord]]:
    """Classify teacher records into (reuse, residual).

    ``teacher_records`` carries ``sample_id``, ``teacher_set``,
    ``teacher_loss``, ``empty_loss`` and (optionally) ``candidate_experts``
    (the per-sample retrieved Top-M). ``top_m_covered[i]`` is True when the
    contributing expert of record ``i`` was inside its per-sample Top-M; a
    recall miss is recorded as a diagnostic residual record that is never
    used for clustering or training.
    """
    if top_m_covered is not None and len(top_m_covered) != len(teacher_records):
        raise ValueError("teacher records and recall coverage must align")
    reuse = []
    residual = []
    for index, record in enumerate(teacher_records):
        sample_id = str(record["sample_id"])
        teacher_set = tuple(int(value) for value in record["teacher_set"])
        teacher_loss = float(record["teacher_loss"])
        empty_loss = float(record["empty_loss"])
        retrieved = tuple(
            int(value) for value in record.get("candidate_experts", ())
        )
        covered = True if top_m_covered is None else bool(top_m_covered[index])
        is_res, reason = is_residual(
            teacher_loss, tau_res, empty_pool=empty_pool, recall_covered=covered
        )
        record_payload = {
            "sample_id": sample_id,
            "task_id": int(record["task_id"]),
            "old_teacher_set": teacher_set,
            "empty_loss": empty_loss,
            "old_teacher_loss": teacher_loss,
            "residual_reason": reason,
            "split": split,
            "retrieved_top_m": retrieved,
            "teacher_multi_hot": record.get("teacher_multi_hot", {}),
            "retrieval_diagnostic": reason == RESIDUAL_REASON_RECALL_MISS,
        }
        if reason == RESIDUAL_REASON_RECALL_MISS:
            residual.append(ComposeResidualRecord(**record_payload))
        elif is_res:
            residual.append(ComposeResidualRecord(**record_payload))
        else:
            reuse.append(ComposeResidualRecord(**record_payload))
    return reuse, residual


def should_create_experts(
    residual_train_count: int, min_residual_samples: int
) -> bool:
    """Never lower ``tau_res`` to force new experts; the count gate is the
    only remaining constraint."""
    return residual_train_count >= min_residual_samples


def write_residual_split(
    root: Path,
    reuse: Sequence[ComposeResidualRecord],
    residual: Sequence[ComposeResidualRecord],
    tau_res: float,
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist reuse/residual records plus a summary (runner helper)."""
    (root / "residual").mkdir(parents=True, exist_ok=True)
    with open(str(root / "residual" / "reuse.json"), "w", encoding="utf-8") as handle:
        json.dump([record.to_dict() for record in reuse], handle, indent=2)
    with open(
        str(root / "residual" / "residual.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump([record.to_dict() for record in residual], handle, indent=2)
    base_summary = {
        "schema_version": RESIDUAL_SCHEMA_VERSION,
        "tau_res": tau_res,
        "reuse_count": len(reuse),
        "residual_count": len(residual),
        "cold_start_count": sum(
            1
            for record in residual
            if record.residual_reason == RESIDUAL_REASON_COLD_START
        ),
        "recall_miss_diagnostics": sum(
            1
            for record in residual
            if record.residual_reason == RESIDUAL_REASON_RECALL_MISS
        ),
    }
    base_summary.update(summary or {})
    with open(
        str(root / "residual" / "summary.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(base_summary, handle, indent=2, sort_keys=True)
    return base_summary


def hash_residual_manifest(records: Sequence[ComposeResidualRecord]) -> str:
    payload = {"records": [record.to_dict() for record in records]}
    return stable_hash(payload)
