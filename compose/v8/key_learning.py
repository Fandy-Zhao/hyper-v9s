"""Alias-key creation and the V8 key loss.

Two responsibilities, both driven entirely by the teacher's *metric* verdicts:

**Lazy creation** (PART 11).  An alias key ``K(k, t)`` is created only when the
teacher selected expert ``k`` for at least ``alias_support_threshold`` samples of
the current task ``t`` -- that is, only when ``|P(k, t)| > 0``.  Experts with zero
support get no key at all, so the pool never grows keys that encode nothing.
Initialisation is the centroid of the queries this expert actually solved::

    K_init(k, t) = Normalize(mean(q_i for i in P(k, t)))

**The loss** (PART 13, v1 -- deliberately minimal)::

    L_key = lambda_pos * L_pos + lambda_rank * L_rank
    L_pos  = 1 - cos(q_i, K(k, t))                       over positives
    L_rank = max(0, margin - sim(q_i, K_pos) + sim(q_i, K_neg))

No BCE, no temperature, no orthogonality/entropy/load-balancing terms: the
specification defers those until an experiment shows the simple form is
insufficient.

The three-valued target is what makes this safe.  A sample where *two* experts
solve the task has one POSITIVE and one IGNORE -- never a negative.  Pushing an
expert's key away from a query it demonstrably solves would delete real
capability, so :func:`build_key_targets` refuses to emit that pairing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F

from compose.v8.config import (
    TARGET_IGNORE,
    TARGET_NEGATIVE,
    TARGET_POSITIVE,
    V8KeyConfig,
    V8PruningConfig,
)
from compose.v8.pool import MultiKeyExpertPool, MultiKeyPoolError, alias_key_init
from compose.v8.teacher import TeacherResult


class KeyLearningError(RuntimeError):
    """Raised when key training is asked to do something V8 forbids."""


@dataclass
class KeyTargets:
    """Three-valued supervision for one alias key.

    ``positive_ids`` are sample ids the key must move toward; ``negative_ids``
    are ids the key claimed but the metric says it does not solve; ``ignored_ids``
    are ids where the expert genuinely solves the sample but another expert was
    selected -- they must contribute no gradient at all.
    """

    key_id: str
    expert_id: int
    task_id: int
    positive_ids: List[str] = field(default_factory=list)
    negative_ids: List[str] = field(default_factory=list)
    ignored_ids: List[str] = field(default_factory=list)

    @property
    def support(self) -> int:
        return len(self.positive_ids)


@dataclass
class KeyLossReport:
    total: torch.Tensor
    positive: torch.Tensor
    ranking: torch.Tensor
    positive_pairs: int
    ranking_pairs: int
    mean_positive_similarity: float
    keys_used: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "loss_key": float(self.total.detach()),
            "loss_key_positive": float(self.positive.detach()),
            "loss_key_ranking": float(self.ranking.detach()),
            "key_positive_pairs": int(self.positive_pairs),
            "key_ranking_pairs": int(self.ranking_pairs),
            "key_mean_positive_similarity": float(self.mean_positive_similarity),
            "key_keys_used": int(self.keys_used),
        }


def build_key_targets(
    teacher_result: TeacherResult,
    pool: MultiKeyExpertPool,
    task_id: int,
) -> Dict[str, KeyTargets]:
    """Group the teacher's per-sample verdicts into per-alias-key targets."""
    targets: Dict[str, KeyTargets] = {}
    for record in teacher_result.records:
        for expert_id, target in record.key_targets.items():
            expert_id = int(expert_id)
            if not pool.has_alias(expert_id, int(task_id)):
                continue
            key_id = pool.alias_key_id(expert_id, int(task_id))
            bucket = targets.setdefault(
                key_id,
                KeyTargets(key_id=key_id, expert_id=expert_id, task_id=int(task_id)),
            )
            if target == TARGET_POSITIVE:
                bucket.positive_ids.append(record.sample_id)
            elif target == TARGET_NEGATIVE:
                bucket.negative_ids.append(record.sample_id)
            elif target == TARGET_IGNORE:
                bucket.ignored_ids.append(record.sample_id)
            else:
                raise KeyLearningError(f"unknown target {target!r}")
    for bucket in targets.values():
        bucket.positive_ids.sort()
        bucket.negative_ids.sort()
        bucket.ignored_ids.sort()
    return targets


def create_alias_keys(
    teacher_result: TeacherResult,
    pool: MultiKeyExpertPool,
    task_id: int,
    queries_by_sample: Mapping[str, torch.Tensor],
    config: Optional[V8PruningConfig] = None,
    trainable: bool = True,
) -> Dict[str, object]:
    """Create one alias key per expert the teacher actually selected.

    Experts with ``|P(k, t)| == 0`` receive **no** key: unconditional per-expert
    creation would add keys that encode nothing and silently dilute the router.
    """
    config = config or V8PruningConfig()
    positives = teacher_result.positives_by_sample()
    support: Dict[int, List[str]] = {}
    for sample_id, expert_ids in positives.items():
        for expert_id in expert_ids:
            support.setdefault(int(expert_id), []).append(str(sample_id))

    created: Dict[str, Dict[str, object]] = {}
    skipped: Dict[int, str] = {}
    for expert_id in sorted(support):
        sample_ids = sorted(support[expert_id])
        if len(sample_ids) < int(config.alias_support_threshold):
            skipped[expert_id] = (
                f"support {len(sample_ids)} below threshold "
                f"{config.alias_support_threshold}"
            )
            continue
        if pool.has_alias(expert_id, int(task_id)):
            skipped[expert_id] = "alias key already exists"
            continue
        vectors: List[torch.Tensor] = []
        for sample_id in sample_ids:
            if sample_id not in queries_by_sample:
                raise KeyLearningError(f"missing query for positive sample {sample_id}")
            vectors.append(queries_by_sample[sample_id].detach().float().reshape(-1))
        value = alias_key_init(torch.stack(vectors, dim=0))
        key_id = pool.add_key(
            expert_id=expert_id,
            task_id=int(task_id),
            key_type="task_alias",
            value=value,
            lifecycle="candidate",
            trainable=bool(trainable),
            support_count=len(sample_ids),
            extra={
                "init": "centroid_of_teacher_positives",
                "support_sample_ids": sample_ids[:256],
                "support_truncated": len(sample_ids) > 256,
            },
        )
        created[key_id] = {
            "expert_id": expert_id,
            "support": len(sample_ids),
            "init_similarity_max": float(
                F.cosine_similarity(
                    F.normalize(value.reshape(1, -1), dim=-1),
                    F.normalize(torch.stack(vectors, dim=0), dim=-1),
                    dim=-1,
                ).max().item()
            ),
        }
    return {
        "task_id": int(task_id),
        "created": created,
        "skipped": skipped,
        "num_created": len(created),
        "num_skipped": len(skipped),
        "experts_with_support": len(support),
    }


def alias_key_loss(
    queries_by_sample: Mapping[str, torch.Tensor],
    pool: MultiKeyExpertPool,
    targets: Mapping[str, KeyTargets],
    config: Optional[V8KeyConfig] = None,
) -> KeyLossReport:
    """``lambda_pos * L_pos + lambda_rank * L_rank`` over the created alias keys."""
    config = config or V8KeyConfig()
    positive_terms: List[torch.Tensor] = []
    ranking_terms: List[torch.Tensor] = []
    similarity_sum = 0.0
    keys_used = 0
    reference = None
    for parameter in pool.parameters():
        reference = parameter
        break

    for key_id in sorted(targets):
        bucket = targets[key_id]
        if key_id not in pool.key_records:
            raise KeyLearningError(f"targets reference a missing key {key_id}")
        if not pool.key_records[key_id].get("trainable", False):
            continue
        if not bucket.positive_ids:
            continue
        key = F.normalize(pool.keys[key_id].float().reshape(-1), dim=-1)
        keys_used += 1

        positive_queries = []
        for sample_id in bucket.positive_ids:
            positive_queries.append(
                F.normalize(queries_by_sample[sample_id].detach().float().reshape(-1), dim=-1)
            )
        if positive_queries:
            stacked = torch.stack(positive_queries, dim=0)
            similarities = stacked @ key
            positive_terms.append((1.0 - similarities).mean())
            similarity_sum += float(similarities.mean().detach().item())

        # Hardest negative: the non-selected key that most claims this query.
        negative_ids = [
            sample_id for sample_id in bucket.negative_ids
            if sample_id not in set(bucket.positive_ids)
        ]
        for sample_id in negative_ids:
            query = F.normalize(
                queries_by_sample[sample_id].detach().float().reshape(-1), dim=-1
            )
            positive_similarity = float((query @ key).detach().item())
            hardest = None
            hardest_similarity = float("-inf")
            for other_id in pool.key_ids():
                if other_id == key_id:
                    continue
                other_record = pool.key_records[other_id]
                if other_record["lifecycle"] == "pruned":
                    continue
                other = F.normalize(pool.keys[other_id].float().reshape(-1), dim=-1)
                similarity = float((query @ other).detach().item())
                if similarity > hardest_similarity:
                    hardest_similarity = similarity
                    hardest = other
            if hardest is None:
                continue
            ranking_terms.append(
                torch.clamp(
                    torch.tensor(
                        config.ranking_margin - positive_similarity + hardest_similarity,
                        dtype=torch.float32,
                    ),
                    min=0.0,
                )
            )

    if reference is None:  # pragma: no cover - a pool always has keys
        raise KeyLearningError("the pool has no parameters")

    zero = reference.new_zeros(())
    positive = (torch.stack(positive_terms).mean() if positive_terms else zero)
    ranking = (torch.stack(ranking_terms).mean() if ranking_terms else zero)
    total = float(config.lambda_pos) * positive + float(config.lambda_rank) * ranking
    return KeyLossReport(
        total=total,
        positive=positive,
        ranking=ranking,
        positive_pairs=sum(len(bucket.positive_ids) for bucket in targets.values()),
        ranking_pairs=len(ranking_terms),
        mean_positive_similarity=(similarity_sum / keys_used) if keys_used else 0.0,
        keys_used=keys_used,
    )


def assert_no_ignore_is_negative(targets: Mapping[str, KeyTargets]) -> None:
    """A sample may not be positive and negative for the same key at once."""
    for key_id, bucket in targets.items():
        overlap = set(bucket.positive_ids) & set(bucket.negative_ids)
        if overlap:
            raise KeyLearningError(
                f"key {key_id} has {len(overlap)} samples both positive and "
                f"negative: {sorted(overlap)[:5]}"
            )
        ignored_overlap = set(bucket.ignored_ids) & set(bucket.negative_ids)
        if ignored_overlap:
            raise KeyLearningError(
                f"key {key_id} has samples both ignored and negative: "
                f"{sorted(ignored_overlap)[:5]}"
            )


__all__ = [
    "KeyLearningError",
    "KeyLossReport",
    "KeyTargets",
    "alias_key_loss",
    "assert_no_ignore_is_negative",
    "build_key_targets",
    "create_alias_keys",
]
