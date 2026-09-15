"""Task-end audits: which task keys to keep, which candidates to commit.

Two decisions, both taken from **globally reduced** statistics (spec §23) so
every rank reaches the same answer from the same numbers:

* **Historical current-task keys** (spec §18).  A historical expert that was
  recalled but never helped has its task key reset to its base identity, so the
  expert returns to the key it had before this task ran.  A key that earned its
  place stays in the expert's key memory, which is what makes the learned
  direction reachable on later tasks.
* **Current candidates** (spec §19).  A candidate becomes a formal expert if it
  was used, contributed positively, and is not a redundant copy of an expert the
  pool already has.  Committing freezes its LoRA and its learned key, at which
  point the key *is* its base functional identity and the expert never trains
  key memory -- if a later task needs it, it may only grow a new task key.

Both audits are pure functions of the statistics plus the pool's key geometry.
No answer text and no per-sample routing rows are stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from compose.v8.pool import (
    LIFECYCLE_CANDIDATE,
    LIFECYCLE_HISTORICAL,
    LIFECYCLE_PRUNED,
)

from .config import V9AuditConfig
from .keys import V9KeyPool


class V9AuditError(RuntimeError):
    """Raised when a task-end decision would leave an illegal pool."""


@dataclass
class ExpertAuditRecord:
    """One expert's task-end evidence, as written to the audit artefact."""

    expert_id: int
    lifecycle: str
    usage: int = 0
    usage_rate: float = 0.0
    selected: int = 0
    selected_rate: float = 0.0
    effective_support: float = 0.0
    mean_contribution: float = 0.0
    mean_positive_contribution: float = 0.0
    positive_contribution_rate: float = 0.0
    validation_gain: Optional[float] = None
    redundancy: Optional[float] = None
    decision: str = "pending"
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_id": int(self.expert_id),
            "lifecycle": self.lifecycle,
            "usage": int(self.usage),
            "usage_rate": float(self.usage_rate),
            "selected": int(self.selected),
            "selected_rate": float(self.selected_rate),
            "effective_support": float(self.effective_support),
            "mean_contribution": float(self.mean_contribution),
            "mean_positive_contribution": float(self.mean_positive_contribution),
            "positive_contribution_rate": float(self.positive_contribution_rate),
            "validation_gain": (
                None if self.validation_gain is None else float(self.validation_gain)
            ),
            "redundancy": None if self.redundancy is None else float(self.redundancy),
            "decision": self.decision,
            "reasons": list(self.reasons),
        }


def _record_from_statistics(expert_id: int, entry: Mapping[str, Any]) -> ExpertAuditRecord:
    return ExpertAuditRecord(
        expert_id=int(expert_id),
        lifecycle=str(entry.get("lifecycle", "unknown")),
        usage=int(entry.get("usage", 0)),
        usage_rate=float(entry.get("usage_rate", 0.0)),
        selected=int(entry.get("selected", 0)),
        selected_rate=float(entry.get("selected_rate", 0.0)),
        effective_support=float(entry.get("effective_support", 0.0)),
        mean_contribution=float(entry.get("mean_contribution", 0.0)),
        mean_positive_contribution=float(entry.get("mean_positive_contribution", 0.0)),
        positive_contribution_rate=float(entry.get("positive_contribution_rate", 0.0)),
    )


def max_key_cosine(
    key_pool: V9KeyPool, expert_id: int, other_expert_ids: Iterable[int]
) -> float:
    """The candidate's nearest neighbour in the pool, by effective-key cosine.

    Computed on the unit base keys, which is the direction routing compares.
    """
    others = [int(value) for value in other_expert_ids if int(value) != int(expert_id)]
    if not others:
        return 0.0
    mine = key_pool.effective_key_matrix([key_pool.base_key_id(expert_id)], detach=True)
    theirs = key_pool.effective_key_matrix(
        [key_pool.base_key_id(value) for value in others], detach=True
    )
    return float((mine @ theirs.T).max().item())


def pairwise_key_cosine(
    key_pool: V9KeyPool, expert_ids: Sequence[int]
) -> Dict[str, Dict[str, float]]:
    """All-pairs cosine among the named experts, keyed by expert id."""
    ids = [int(value) for value in expert_ids]
    if not ids:
        return {}
    matrix = key_pool.effective_key_matrix(
        [key_pool.base_key_id(value) for value in ids], detach=True
    )
    similarity = matrix @ matrix.T
    return {
        str(left): {
            str(right): float(similarity[i, j].item())
            for j, right in enumerate(ids)
            if j != i
        }
        for i, left in enumerate(ids)
    }


def audit_historical_task_keys(
    key_pool: V9KeyPool,
    statistics: Mapping[str, Mapping[str, Any]],
    config: V9AuditConfig,
    task_index: int,
) -> Dict[str, Any]:
    """Decide which of this task's historical task keys survive (spec §18).

    An expert with no task key on this task is not audited: it was not recalled
    through an adapted key, so there is nothing to retire.

    A key that is *rejected* is not deleted and the expert is not hidden -- it
    is reset to the expert's base identity, so the expert remains recallable
    exactly as it was before this task ran.  Rejecting a key means "this task
    taught this key nothing worth keeping", not "this expert was useless".
    """
    task_index = int(task_index)
    retain: List[int] = []
    reset: List[int] = []
    records: Dict[str, Any] = {}
    for expert_id in key_pool.historical_ids:
        if not key_pool.has_task_key(expert_id, task_index):
            continue
        entry = statistics.get(str(int(expert_id)), {})
        record = _record_from_statistics(
            expert_id, {**entry, "lifecycle": LIFECYCLE_HISTORICAL}
        )
        # The criterion is what the *key* earned, not what the expert is worth:
        # a historical expert that was recalled but never contributed has no
        # evidence that its new key is an improvement on its base identity, and
        # keeping the new key would silently re-anchor the expert.
        if record.usage_rate < float(config.min_key_usage_rate):
            record.reasons.append(
                "usage_rate {:.4f} < {:.4f}".format(
                    record.usage_rate, float(config.min_key_usage_rate)
                )
            )
        if record.mean_positive_contribution <= float(
            config.min_key_mean_positive_contribution
        ):
            record.reasons.append(
                "mean_positive_contribution {:.6f} <= {:.6f}".format(
                    record.mean_positive_contribution,
                    float(config.min_key_mean_positive_contribution),
                )
            )
        if record.reasons:
            record.decision = "reset_task_key"
            reset.append(int(expert_id))
        else:
            record.decision = "retain_task_key"
            retain.append(int(expert_id))
        records[str(int(expert_id))] = record.to_dict()
    total = len(retain) + len(reset)
    return {
        "task_index": task_index,
        "retain_task_key": sorted(retain),
        "reset_task_key": sorted(reset),
        #: Spec §34: the fraction of this task's new historical keys that earned
        #: their place.  Near 1 means every recalled expert improved; near 0
        #: means the recall is bringing in experts the answer never wanted.
        "new_key_retention_rate": (float(len(retain)) / total) if total else 0.0,
        "records": records,
    }


def apply_historical_task_key_audit(
    key_pool: V9KeyPool, decisions: Mapping[str, Any], task_index: int
) -> Dict[str, Any]:
    """Apply the task-key decisions and retire both sets.

    A retained key is marked ``historical``, which is what puts it inside
    :meth:`V9KeyPool.historical_key_ids` -- and therefore inside the freeze
    checksums that later tasks assert against.  Leaving it labelled a candidate
    would exclude the one key a later task is most likely to corrupt from the
    integrity audit entirely.

    A rejected key is reset to the base identity, frozen, and marked pruned:
    the value is now identical to the expert's base key, so keeping it live
    would put the same direction in the key memory twice.
    """
    applied = {"reset": [], "retained": []}
    for expert_id in decisions.get("reset_task_key", ()):
        key_id = key_pool.task_key_id(int(expert_id), int(task_index))
        key_pool.reset_task_key(int(expert_id), int(task_index))
        key_pool.set_key_trainable(key_id, False)
        key_pool.set_key_lifecycle(key_id, LIFECYCLE_PRUNED)
        applied["reset"].append(int(expert_id))
    for expert_id in decisions.get("retain_task_key", ()):
        key_id = key_pool.task_key_id(int(expert_id), int(task_index))
        key_pool.set_key_trainable(key_id, False)
        key_pool.set_key_lifecycle(key_id, LIFECYCLE_HISTORICAL)
        applied["retained"].append(int(expert_id))
    return applied


def audit_candidates(
    key_pool: V9KeyPool,
    statistics: Mapping[str, Mapping[str, Any]],
    config: V9AuditConfig,
    task_index: int,
    validation_gain: Optional[Mapping[int, float]] = None,
    redundancy_against: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """Decide which current candidates become formal experts (spec §19).

    ``validation_gain[k]`` is the mean exact answer-loss reduction the candidate
    buys on held-out samples under the deployment routing rule; it is the one
    piece of evidence that does not come from the training statistics.  It is
    optional: if the caller did not run the validation pass, the decision falls
    back to the training evidence and says so in the record.
    """
    task_index = int(task_index)
    candidate_ids = key_pool.current_ids
    historical_ids = [int(value) for value in (redundancy_against or key_pool.historical_ids)]
    gain = {int(k): float(v) for k, v in (validation_gain or {}).items()}
    commit: List[int] = []
    delete: List[int] = []
    records: Dict[str, Any] = {}
    for expert_id in candidate_ids:
        entry = statistics.get(str(int(expert_id)), {})
        record = _record_from_statistics(
            expert_id, {**entry, "lifecycle": LIFECYCLE_CANDIDATE}
        )
        record.validation_gain = gain.get(int(expert_id))
        record.redundancy = max_key_cosine(key_pool, int(expert_id), historical_ids)
        low_usage = record.usage_rate < float(config.min_candidate_usage_rate)
        no_contribution = record.mean_positive_contribution <= float(
            config.min_candidate_positive_contribution
        )
        never_positive = record.positive_contribution_rate <= float(
            config.min_candidate_positive_rate
        )
        weak_gain = (
            record.validation_gain is not None
            and record.validation_gain <= float(config.min_validation_answer_gain)
        )
        duplicated = (
            record.redundancy is not None
            and record.redundancy >= float(config.redundancy_cosine_threshold)
        )
        if low_usage:
            record.reasons.append(
                "usage_rate {:.4f} < {:.4f}".format(
                    record.usage_rate, float(config.min_candidate_usage_rate)
                )
            )
        if no_contribution:
            record.reasons.append(
                "mean_positive_contribution {:.6f} <= {:.6f}".format(
                    record.mean_positive_contribution,
                    float(config.min_candidate_positive_contribution),
                )
            )
        if never_positive:
            record.reasons.append(
                "positive_contribution_rate {:.4f} <= {:.4f}".format(
                    record.positive_contribution_rate,
                    float(config.min_candidate_positive_rate),
                )
            )
        if weak_gain:
            record.reasons.append(
                "validation_gain {:.6f} <= {:.6f}".format(
                    record.validation_gain, float(config.min_validation_answer_gain)
                )
            )
        if duplicated:
            record.reasons.append(
                "redundancy {:.4f} >= {:.4f} against an existing expert".format(
                    record.redundancy, float(config.redundancy_cosine_threshold)
                )
            )
        if low_usage or no_contribution or never_positive or weak_gain:
            record.decision = "delete"
            delete.append(int(expert_id))
        elif duplicated:
            # Highly duplicated *and* carrying no independent evidence: this is
            # the V8 redundancy case, and it is resolved the same way -- the pool
            # keeps the established expert and drops the copy.
            record.decision = "delete_redundant"
            delete.append(int(expert_id))
        else:
            record.decision = "commit"
            commit.append(int(expert_id))
        records[str(int(expert_id))] = record.to_dict()

    forced: List[int] = []
    return {
        "task_index": task_index,
        "commit": sorted(commit),
        "delete": sorted(delete),
        "forced_commit": sorted(forced),
        "records": records,
    }


def apply_candidate_commit(
    key_pool: V9KeyPool,
    manager,
    decisions: Mapping[str, Any],
    task_index: int,
) -> Dict[str, Any]:
    """Promote the passing candidates to formal, frozen experts (spec §19).

    The candidate's learned key *becomes* its base functional key: it is frozen
    in place rather than copied, so the direction it was trained to have is the
    direction inference will compare against.  Its LoRA is frozen at the same
    moment -- from here on, reuse is only possible through a new task key.

    ``manager`` may be ``None``.  The task-end audit runs in its own process,
    after the training process has exited, so there is no model to freeze there;
    the LoRA half of the commit is applied by the *next* task's startup, which
    calls ``pool.train_only(current candidates)`` and therefore freezes every
    expert the committed lifecycle now excludes -- and asserts it.  The
    lifecycle move is the part that has to persist, because that is what the
    next task reads when it decides which experts are historical.
    """
    committed: List[int] = []
    deleted: List[int] = []
    for expert_id in decisions.get("commit", ()):
        expert_id = int(expert_id)
        record = key_pool.expert_record(expert_id)
        record["lifecycle"] = LIFECYCLE_HISTORICAL
        # Bookkeeping goes in ``extra``, which is the record's declared place for
        # it.  A new top-level field would be splatted into ``add_expert`` on the
        # next load -- ``from_state`` forwards every key it does not recognise --
        # and the committed pool would fail to load in the task that needs it.
        record.setdefault("extra", {})["committed_task"] = int(task_index)
        key_id = key_pool.origin_key_id(expert_id)
        key_pool.set_key_trainable(key_id, False)
        key_pool.set_key_lifecycle(key_id, LIFECYCLE_HISTORICAL)
        if manager is not None:
            for layer in manager.layers.values():
                for parameter in layer.experts[str(expert_id)].parameters():
                    parameter.requires_grad_(False)
                    parameter.grad = None
        committed.append(expert_id)
    for expert_id in decisions.get("delete", ()):
        expert_id = int(expert_id)
        key_pool.expert_record(expert_id)["lifecycle"] = LIFECYCLE_PRUNED
        for key_id in key_pool.key_ids(expert_id=expert_id):
            key_pool.set_key_trainable(key_id, False)
            key_pool.set_key_lifecycle(key_id, LIFECYCLE_PRUNED)
        deleted.append(expert_id)
    if len(committed) + len(key_pool.historical_ids) < 1:
        raise V9AuditError("a committed pool must keep at least one expert")
    key_pool.validate()
    return {
        "committed": sorted(committed),
        "deleted": sorted(deleted),
        "historical_ids": sorted(key_pool.historical_ids),
    }


def validation_answer_gain(
    per_candidate_removal: Mapping[int, torch.Tensor],
    per_candidate_baseline: torch.Tensor,
) -> Dict[int, float]:
    """Mean exact answer-loss reduction per candidate, on held-out samples.

    ``per_candidate_removal[k][i]`` is the loss on sample ``i`` with candidate
    ``k`` removed from the deployment selection and ``per_candidate_baseline[i]``
    is the loss with the full selection.  A positive value means the candidate
    lowered the loss, i.e. it helped.
    """
    baseline = per_candidate_baseline.detach().float()
    return {
        int(expert_id): float((values.detach().float() - baseline).mean().item())
        for expert_id, values in per_candidate_removal.items()
        if values.numel() == baseline.numel()
    }


def candidate_usage_entropy(
    statistics: Mapping[str, Mapping[str, Any]], expert_ids: Sequence[int]
) -> float:
    """Normalised entropy of candidate exposure -- the collapse detector.

    Task 0's dominant failure mode is one candidate winning early and never
    being challenged.  Entropy near 1 means exposure is spread; entropy near 0
    means a single candidate is taking everything.
    """
    counts = torch.tensor(
        [float(statistics.get(str(int(value)), {}).get("usage", 0.0)) for value in expert_ids],
        dtype=torch.float64,
    )
    total = float(counts.sum().item())
    if counts.numel() < 2 or total <= 0:
        return 0.0
    probabilities = counts / total
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * probabilities.log()).sum().item())
    return entropy / float(torch.log(torch.tensor(float(counts.numel()))).item())


__all__ = [
    "ExpertAuditRecord",
    "V9AuditError",
    "apply_candidate_commit",
    "apply_historical_task_key_audit",
    "audit_candidates",
    "audit_historical_task_keys",
    "candidate_usage_entropy",
    "max_key_cosine",
    "pairwise_key_cosine",
    "validation_answer_gain",
]
