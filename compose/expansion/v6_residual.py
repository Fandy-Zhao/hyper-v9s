"""V6 residual buffers and reuse/residual splitting (Stage E5).

Residual judgment never uses the Router's predicted set; it is driven by
the answer-supervised teacher:

  a residual sample is one where the best old-expert teacher set is not
  empty (old experts help) yet its loss is still insufficient
  (``old_gain = empty_loss - old_teacher_loss`` below the floor), and the
  contributing expert was inside the Router Top-M (not a recall miss).

Outputs: reuse train, residual train, residual validation and a router
calibration subset. Validation residuals never update candidates.

When residual train is below ``min_residual_samples`` no candidate is
created: ``should_create_candidates`` returns False and the task moves
straight to global Router calibration; thresholds are never lowered to
force a candidate.
"""

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from compose.teacher.v6_teacher import V6TeacherRecord

V6_BUFFER_SCHEMA_VERSION = 1

VALID_SPLITS = ("train", "val")
RESIDUAL_REASON_GAIN_FLOOR = "teacher_gain_below_floor"
RESIDUAL_REASON_EMPTY_TEACHER = "empty_teacher_no_residual"
RESIDUAL_REASON_RECALL_MISS = "recall_miss_not_residual"
RESIDUAL_REASON_SUFFICIENT = "teacher_sufficient"
RESIDUAL_REASON_BASE_ONLY = "base_only_insufficient"


@dataclass(frozen=True)
class V6ResidualRecord:
    sample_id: str
    task_id: int
    old_teacher_set: Tuple[int, ...]
    empty_loss: float
    old_teacher_loss: float
    old_gain: float
    residual_reason: str
    query_feature_path: str
    split: str
    candidate_experts: Tuple[int, ...] = ()
    teacher_multi_hot: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id is required")
        if self.split not in VALID_SPLITS:
            raise ValueError("split must be one of {}".format(VALID_SPLITS))
        if self.residual_reason not in (
            RESIDUAL_REASON_GAIN_FLOOR,
            RESIDUAL_REASON_EMPTY_TEACHER,
            RESIDUAL_REASON_RECALL_MISS,
            RESIDUAL_REASON_SUFFICIENT,
            RESIDUAL_REASON_BASE_ONLY,
        ):
            raise ValueError("unknown residual_reason: {!r}".format(self.residual_reason))
        if not all(value >= 0 for value in (self.empty_loss, self.old_teacher_loss)):
            raise ValueError("losses must be non-negative")
        object.__setattr__(
            self,
            "teacher_multi_hot",
            {int(key): int(value) for key, value in self.teacher_multi_hot.items()},
        )

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["old_teacher_set"] = list(self.old_teacher_set)
        data["candidate_experts"] = list(self.candidate_experts)
        return data

    @classmethod
    def from_dict(cls, value: Dict[str, object]) -> "V6ResidualRecord":
        data = dict(value)
        data["old_teacher_set"] = tuple(int(item) for item in data["old_teacher_set"])
        data["candidate_experts"] = tuple(int(item) for item in data.get("candidate_experts", ()))
        return cls(**data)


class V6ResidualBuffer:
    """Atomic, sharded, resumable residual buffer for train and val splits."""

    def __init__(self) -> None:
        self._records = {}  # type: Dict[str, V6ResidualRecord]

    @property
    def records(self) -> Tuple[V6ResidualRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    @property
    def sample_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._records))

    def add(self, record: V6ResidualRecord) -> bool:
        if record.sample_id in self._records:
            if self._records[record.sample_id] != record:
                raise ValueError("conflicting duplicate residual sample")
            return False
        self._records[record.sample_id] = record
        return True

    def split_records(self, split: str) -> Tuple[V6ResidualRecord, ...]:
        return tuple(record for record in self.records if record.split == split)

    def write_shard(self, path, rank: int) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": V6_BUFFER_SCHEMA_VERSION,
            "rank": int(rank),
            "records": [record.to_dict() for record in self.records],
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load_shard(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != V6_BUFFER_SCHEMA_VERSION:
            raise ValueError("unsupported V6 residual buffer schema")
        result = cls()
        for value in payload["records"]:
            result.add(V6ResidualRecord.from_dict(value))
        return result, int(payload["rank"])

    @classmethod
    def merge_shards(
        cls,
        paths: Sequence[str],
        output_path: str,
        expected_sample_ids: Iterable[str],
    ) -> int:
        expected = set(expected_sample_ids)
        seen_ranks = set()
        merged = cls()
        seen_ids = set()
        for path in paths:
            shard, rank = cls.load_shard(path)
            if rank in seen_ranks:
                raise ValueError("duplicate shard rank {}".format(rank))
            seen_ranks.add(rank)
            for record in shard.records:
                if record.sample_id in seen_ids:
                    raise ValueError(
                        "cross-shard duplicate sample {}".format(record.sample_id)
                    )
                seen_ids.add(record.sample_id)
                merged.add(record)
        if seen_ids != expected:
            missing = sorted(expected - seen_ids)
            unexpected = sorted(seen_ids - expected)
            raise ValueError(
                "shard merge mismatch: missing {}, unexpected {}".format(
                    missing, unexpected
                )
            )
        merged.write_shard(output_path, rank=0)
        return len(seen_ranks)


def is_residual(
    record: V6TeacherRecord,
    min_old_gain: float,
    recall_covered: bool,
    base_only_mode: bool = False,
) -> Tuple[bool, str]:
    """Answer-teacher residual judgment (no Router prediction involved).

    Returns ``(is_residual, reason)`` where reason is one of the
    RESIDUAL_REASON_* constants.

    ``base_only_mode`` is the empty-registry fix (Stage R5): when the active
    expert registry is empty the current system is the frozen backbone, so a
    sample with an empty teacher set is judged by the SAME gain-floor test
    against the base's own loss (``old_gain = empty_loss - teacher_loss``
    with ``teacher_loss == empty_loss`` -> 0).  The existing
    ``min_old_gain`` threshold is reused verbatim; no new threshold is
    introduced for the empty-registry case.
    """
    if not record.teacher_set:
        if base_only_mode:
            # Current system = Base; old_gain over Base is 0, which is below
            # every positive min_old_gain floor -> Base not sufficient ->
            # residual candidate material.  The commit gate still filters
            # candidates that cannot produce real gain on these samples.
            return True, RESIDUAL_REASON_BASE_ONLY
        return False, RESIDUAL_REASON_EMPTY_TEACHER
    old_gain = record.empty_loss - record.teacher_loss
    if old_gain < min_old_gain:
        # Teacher picked old experts but the gain is below the floor:
        # old experts are not sufficient -> residual candidate material.
        if not recall_covered:
            return False, RESIDUAL_REASON_RECALL_MISS
        return True, RESIDUAL_REASON_GAIN_FLOOR
    return False, RESIDUAL_REASON_SUFFICIENT


def build_residual_records(
    teacher_records: Sequence[V6TeacherRecord],
    query_feature_path: str,
    min_old_gain: float,
    top_m_covered: Sequence[bool],
    split: str,
    base_only_mode: bool = False,
) -> Tuple[List[V6ResidualRecord], List[V6ResidualRecord]]:
    """Classify teacher records into (reuse, residual).

    ``top_m_covered[i]`` is True when the contributing expert of record
    ``i`` was inside the Router Top-M (recall audit); a recall miss is
    recorded as a residual record with RESIDUAL_REASON_RECALL_MISS so it
    surfaces in audits but is never used for candidate training.

    ``base_only_mode=True`` (empty active registry): empty-teacher samples
    are no longer dropped; they become residual material via the existing
    gain-floor test (see :func:`is_residual`).  This is what makes the
    empty-registry state non-absorbing: re-bootstrap candidate training can
    restart from the frozen backbone.
    """
    if len(teacher_records) != len(top_m_covered):
        raise ValueError("teacher records and recall coverage must align")
    reuse = []
    residual = []

    def _to_record(record, feature_path, old_gain, reason, split):
        return V6ResidualRecord(
            sample_id=record.sample_id,
            task_id=record.task_id,
            old_teacher_set=record.teacher_set,
            empty_loss=record.empty_loss,
            old_teacher_loss=record.teacher_loss,
            old_gain=old_gain,
            residual_reason=reason,
            query_feature_path=feature_path,
            split=split,
            candidate_experts=record.candidate_experts,
            teacher_multi_hot=record.teacher_multi_hot,
        )
    for record, covered in zip(teacher_records, top_m_covered):
        if not record.teacher_set:
            if base_only_mode:
                old_gain = (record.empty_loss or 0.0) - (record.teacher_loss or 0.0)
                residual.append(
                    _to_record(record, query_feature_path, old_gain,
                               RESIDUAL_REASON_BASE_ONLY, split)
                )
            continue  # non-base-only: empty teacher means no old expert involved
        old_gain = record.empty_loss - record.teacher_loss
        is_res, reason = is_residual(record, min_old_gain, covered)
        if reason == RESIDUAL_REASON_RECALL_MISS:
            # A recall miss is not usable residual material (the Router
            # missed the contributor), but it is recorded as a residual
            # record so audits surface it; never used for training.
            residual.append(_to_record(record, query_feature_path, old_gain, reason, split))
        elif is_res:
            residual.append(_to_record(record, query_feature_path, old_gain, reason, split))
        else:
            reuse.append(_to_record(record, query_feature_path, old_gain, reason, split))
    return reuse, residual


def should_create_candidates(
    residual_train_count: int, min_residual_samples: int
) -> bool:
    """Never lower the threshold to force a candidate."""
    return residual_train_count >= min_residual_samples


def router_calibration_subset(
    records: Sequence[V6ResidualRecord],
    capacity: int,
    seed: int = 42,
) -> Tuple[str, ...]:
    """Seeded stratified subset (by residual reason and split) for Router
    calibration; returns sample ids."""
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    import random

    rng = random.Random(seed)
    buckets = {}
    for record in records:
        key = (record.split, record.residual_reason)
        buckets.setdefault(key, []).append(record.sample_id)
    selected = []
    for key in sorted(buckets):
        pool = buckets[key]
        rng.shuffle(pool)
        quota = max(1, capacity // len(buckets)) if buckets else 0
        selected.extend(pool[:quota])
    rng.shuffle(selected)
    return tuple(selected[:capacity])
