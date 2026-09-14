"""Key and expert pruning for V8.

Two different things get pruned, and confusing them would be dangerous:

**Keys.**  A historical alias key can become useless -- its expert stopped being
selected, or it grew so close to another key of the same expert that it can no
longer discriminate.  Pruning it is a *routing-table* edit, and it is reversible
in the sense that the expert and its origin key are untouched.

**Experts.**  A candidate expert can be redundant with an existing one.  That is
a *capability* edit and much more consequential, so it is gated behind an
explicit gain requirement.

The invariant both paths share: a key or expert that is still the only carrier
of some capability must not be removed.  Every prune decision therefore reports
its evidence, and :func:`assert_pool_not_emptied` refuses the degenerate outcome
where pruning leaves an expert with no live key at all -- that expert would
become unreachable while still occupying its slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from compose.v8.config import V8PruningConfig
from compose.v8.pool import (
    LIFECYCLE_PRUNED,
    MultiKeyExpertPool,
    MultiKeyPoolError,
)


class PruningError(RuntimeError):
    """Raised when a prune would leave the pool in an incoherent state."""


@dataclass
class PruneDecision:
    """One prune recommendation with the evidence behind it."""

    key_id: str
    expert_id: int
    task_id: int
    key_type: str
    reason: str
    support: int
    teacher_gain: float
    max_similarity_to_sibling: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_id": self.key_id,
            "expert_id": self.expert_id,
            "task_id": self.task_id,
            "key_type": self.key_type,
            "reason": self.reason,
            "support": self.support,
            "teacher_gain": self.teacher_gain,
            "max_similarity_to_sibling": self.max_similarity_to_sibling,
        }


def _sibling_similarity(pool: MultiKeyExpertPool, key_id: str) -> Tuple[Optional[str], float]:
    """Highest cosine to another *live* key of the same expert."""
    record = pool.key_records[key_id]
    expert_id = int(record["expert_id"])
    target = F.normalize(pool.keys[key_id].detach().float().reshape(-1), dim=-1)
    best_id: Optional[str] = None
    best = float("-inf")
    for other_id in pool.active_key_ids_for_expert(expert_id):
        if other_id == key_id:
            continue
        other = F.normalize(pool.keys[other_id].detach().float().reshape(-1), dim=-1)
        similarity = float((target @ other).item())
        if similarity > best:
            best, best_id = similarity, other_id
    return best_id, (0.0 if best == float("-inf") else best)


def plan_key_pruning(
    pool: MultiKeyExpertPool,
    config: Optional[V8PruningConfig] = None,
    support_by_key: Optional[Mapping[str, int]] = None,
    gain_by_key: Optional[Mapping[str, float]] = None,
) -> List[PruneDecision]:
    """Enumerate alias keys that lost their support or became redundant.

    Origin keys are never proposed: they *are* the expert's identity in the
    pool, and removing one would make the expert unreachable through the
    migration path that produced it.  Only ``task_alias`` keys are candidates,
    and never one whose removal would leave its expert with no live key.
    """
    config = config or V8PruningConfig()
    decisions: List[PruneDecision] = []
    for key_id in pool.key_ids(key_type="task_alias"):
        record = pool.key_records[key_id]
        if record["lifecycle"] == LIFECYCLE_PRUNED:
            continue
        expert_id = int(record["expert_id"])
        support = int(
            support_by_key.get(key_id, record["support_count"])
            if support_by_key else record["support_count"]
        )
        gain = float(
            gain_by_key.get(key_id, record["teacher_gain"])
            if gain_by_key else record["teacher_gain"]
        )
        _sibling_id, similarity = _sibling_similarity(pool, key_id)

        reason = None
        if support < int(config.alias_support_threshold):
            reason = (
                f"support {support} below threshold {config.alias_support_threshold}"
            )
        elif gain < float(config.alias_gain_threshold):
            # The threshold is the *minimum* gain that justifies a key, so a
            # gain exactly at the threshold is kept.  This matters because
            # ``teacher_gain`` defaults to 0.0: with a 0.0 threshold and a
            # non-strict comparison, the default configuration would retire
            # every key whose gain had simply not been measured yet.
            reason = (
                f"teacher gain {gain:.6f} below threshold "
                f"{config.alias_gain_threshold}"
            )
        elif similarity >= float(config.alias_redundancy_threshold):
            reason = (
                f"redundant with a sibling key at cosine {similarity:.6f} >= "
                f"{config.alias_redundancy_threshold}"
            )
        if reason is None:
            continue

        remaining = [
            other for other in pool.active_key_ids_for_expert(expert_id)
            if other != key_id
        ]
        if not remaining:
            # Only reachable if a caller has already pruned this expert's origin
            # key out of band; in that state the pool is incoherent anyway and
            # `assert_pool_not_emptied` will say so.  Never remove the last key.
            continue
        decisions.append(PruneDecision(
            key_id=key_id,
            expert_id=expert_id,
            task_id=int(record["task_id"]),
            key_type=str(record["key_type"]),
            reason=reason,
            support=support,
            teacher_gain=gain,
            max_similarity_to_sibling=similarity,
        ))
    return decisions


def apply_pruning(
    pool: MultiKeyExpertPool,
    decisions: Iterable[PruneDecision],
) -> Dict[str, Any]:
    """Apply prune decisions, then verify the pool is still coherent."""
    pruned: List[str] = []
    for decision in decisions:
        if decision.key_id not in pool.key_records:
            raise PruningError(f"prune targets a missing key {decision.key_id}")
        pool.set_key_lifecycle(decision.key_id, LIFECYCLE_PRUNED)
        pool.set_key_trainable(decision.key_id, False)
        pruned.append(decision.key_id)
    assert_pool_not_emptied(pool)
    return {"pruned": pruned, "num_pruned": len(pruned), "audit": pool.audit()}


def assert_pool_not_emptied(pool: MultiKeyExpertPool) -> None:
    """Every *live* expert must keep a live key, or routing cannot reach it.

    A migrated V7 pool may legitimately retain tombstone records and weights
    for experts whose lifecycle is already ``pruned``.  Routing excludes those
    experts, so their deliberately-pruned origin keys are not a new V8 pruning
    failure.
    """
    stranded = [
        expert_id for expert_id in pool.live_expert_ids()
        if not pool.active_key_ids_for_expert(expert_id)
    ]
    if stranded:
        raise PruningError(
            f"pruning would strand {len(stranded)} experts with no live key: "
            f"{stranded[:8]}"
        )


def plan_candidate_pruning(
    candidate_vectors: Mapping[str, torch.Tensor],
    existing_vectors: Mapping[str, torch.Tensor],
    config: Optional[V8PruningConfig] = None,
) -> Dict[str, Any]:
    """Find candidate experts that duplicate an existing one.

    Redundancy is decided on the candidate's *routing vector* (its key), not on
    its weights: two experts that occupy the same region of query space would be
    selected for the same samples, so one of them adds cost without adding
    capability.
    """
    config = config or V8PruningConfig()
    if not config.candidate_prune_enabled:
        return {"enabled": False, "redundant": {}, "num_redundant": 0}
    redundant: Dict[str, Dict[str, Any]] = {}
    for candidate_id, vector in candidate_vectors.items():
        candidate = F.normalize(torch.as_tensor(vector).float().reshape(-1), dim=-1)
        best_id = None
        best = float("-inf")
        for existing_id, other in existing_vectors.items():
            other = F.normalize(torch.as_tensor(other).float().reshape(-1), dim=-1)
            similarity = float((candidate @ other).item())
            if similarity > best:
                best, best_id = similarity, str(existing_id)
        if best >= float(config.candidate_redundancy_cosine):
            redundant[str(candidate_id)] = {
                "duplicates": best_id,
                "cosine": best,
            }
    return {
        "enabled": True,
        "redundant": redundant,
        "num_redundant": len(redundant),
        "threshold": float(config.candidate_redundancy_cosine),
    }


def plan_expert_removal(
    contribution_by_expert: Mapping[int, float],
    config: Optional[V8PruningConfig] = None,
) -> List[Dict[str, Any]]:
    """Experts whose measured contribution does not justify their slot.

    Same boundary convention as :func:`plan_key_pruning`: the configured value is
    the minimum contribution worth keeping, so an expert sitting exactly on it is
    kept and only a contribution strictly below it triggers removal.
    """
    config = config or V8PruningConfig()
    removals = [
        {
            "expert_id": int(expert_id),
            "contribution": float(contribution),
            "reason": (
                f"contribution {float(contribution):.6f} below threshold "
                f"{config.candidate_min_removal_gain}"
            ),
        }
        for expert_id, contribution in sorted(contribution_by_expert.items())
        if float(contribution) < float(config.candidate_min_removal_gain)
    ]
    return removals


__all__ = [
    "PruneDecision",
    "PruningError",
    "apply_pruning",
    "assert_pool_not_emptied",
    "plan_candidate_pruning",
    "plan_expert_removal",
    "plan_key_pruning",
]
