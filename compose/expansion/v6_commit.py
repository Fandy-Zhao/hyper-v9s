"""V6 candidate validation and transactional commit (Stage E7).

On the held-out residual validation set, each candidate slot m is scored:

    G_im = loss_i(old_teacher_set) - loss_i(old_teacher_set + candidate_m)

Statistics per slot: support count (G > 0), mean/median gain, positive
gain rate, key accuracy (assignment agreement with positive samples),
false activation on non-support samples, parameter cosine, key cosine and
positive-sample overlap. The joint effect of both candidates together is
also evaluated.

Commit conditions come from the configuration:
    support_count >= tau_support, mean_gain >= tau_gain,
    key_accuracy >= tau_key.

Redundant candidates (high parameter cosine AND high positive overlap)
keep the one with the higher validation gain; no merge distillation in
this batch. 0, 1 or 2 experts may be committed.

The commit itself is the two-phase transaction from Stage E2:
pending marker -> write LoRA checkpoint / key / validation report ->
hash verification -> atomic registry update (provisional) -> pool_version
bump -> marker removal. Recovery never double-commits.
"""

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from compose.experts.transaction import CommitTransaction, file_sha256

from .candidate_pool import CandidateExpertPool
from .v6_candidate import V6CandidateConfig


@dataclass
class CandidateValidationStats:
    slot_id: int
    support_count: int
    mean_gain: float
    median_gain: float
    positive_gain_rate: float
    key_accuracy: float
    false_activation_rate: float
    param_cosine: Optional[float]  # cosine with the other slot
    key_cosine: Optional[float]
    positive_sample_ids: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "support_count": self.support_count,
            "mean_gain": self.mean_gain,
            "median_gain": self.median_gain,
            "positive_gain_rate": self.positive_gain_rate,
            "key_accuracy": self.key_accuracy,
            "false_activation_rate": self.false_activation_rate,
            "param_cosine": self.param_cosine,
            "key_cosine": self.key_cosine,
            "positive_sample_ids": list(self.positive_sample_ids),
        }


@dataclass
class CommitDecision:
    slot_id: int
    status: str  # provisional | rejected | redundant
    reason: str
    stats: CandidateValidationStats
    committed_expert_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "status": self.status,
            "reason": self.reason,
            "stats": self.stats.to_dict(),
            "committed_expert_id": self.committed_expert_id,
        }


def _params_flat(module: nn.Module) -> Optional[torch.Tensor]:
    tensors = [parameter.detach().float().reshape(-1) for parameter in module.parameters()]
    if not tensors:
        return None
    return torch.cat(tensors)


def evaluate_candidate(
    pool: CandidateExpertPool,
    slot_id: int,
    records: Sequence[Dict[str, Any]],
    old_expert_ids: Sequence[int],
    loss_fn: Callable[[Dict[str, Any], Tuple[int, ...]], float],
    queries: torch.Tensor,
) -> CandidateValidationStats:
    """Per-slot validation on held-out residual records.

    ``records[i]`` carries at least ``sample_id`` and ``old_teacher_set``;
    ``loss_fn(record, expert_set)`` returns the answer loss under that set
    (``old_teacher_set + (candidate,)`` for the candidate condition).
    ``queries[i]`` is the frozen query feature of record ``i``.
    """
    if queries.ndim != 2 or queries.shape[0] != len(records):
        raise ValueError("queries must align with records")
    gains = []
    positive_ids = []
    assigned_to_slot = pool.assign(queries).eq(slot_id)
    for index, record in enumerate(records):
        old_set = tuple(sorted(int(value) for value in record["old_teacher_set"]))
        candidate_set = tuple(sorted(set(old_set) | {slot_id}))
        baseline = loss_fn(record, old_set)
        with_candidate = loss_fn(record, candidate_set)
        gain = baseline - with_candidate
        gains.append(gain)
        if gain > 0:
            positive_ids.append(str(record["sample_id"]))
    support = len(positive_ids)
    mean_gain = float(statistics.mean(gains)) if gains else 0.0
    median_gain = float(statistics.median(gains)) if gains else 0.0
    positive_rate = support / len(records) if records else 0.0

    # Key accuracy: how often the slot's key assignment matches its support.
    positive_set = set(positive_ids)
    positive_indices = [index for index, record in enumerate(records)
                        if str(record["sample_id"]) in positive_set]
    key_accuracy = 0.0
    if positive_indices:
        key_accuracy = float(
            assigned_to_slot[torch.tensor(positive_indices, dtype=torch.long)].float().mean()
        )
    false_activation = 0.0
    negative_indices = [index for index, record in enumerate(records)
                        if str(record["sample_id"]) not in positive_set]
    if negative_indices:
        false_activation = float(
            assigned_to_slot[torch.tensor(negative_indices, dtype=torch.long)].float().mean()
        )

    param_cosine = None
    key_cosine = None
    if pool.config.slot_count == 2:
        other = 1 - slot_id
        flat_self = _params_flat(pool.slots[slot_id].adapter)
        flat_other = _params_flat(pool.slots[other].adapter)
        if flat_self is not None and flat_other is not None:
            param_cosine = float(
                torch.nn.functional.cosine_similarity(flat_self, flat_other, dim=0)
            )
        key_cosine = float(
            torch.nn.functional.normalize(pool.keys[slot_id].detach(), dim=0)
            @ torch.nn.functional.normalize(pool.keys[other].detach(), dim=0)
        )
    return CandidateValidationStats(
        slot_id=slot_id,
        support_count=support,
        mean_gain=mean_gain,
        median_gain=median_gain,
        positive_gain_rate=positive_rate,
        key_accuracy=key_accuracy,
        false_activation_rate=false_activation,
        param_cosine=param_cosine,
        key_cosine=key_cosine,
        positive_sample_ids=tuple(sorted(positive_ids)),
    )


def decide_commits(
    stats_list: Sequence[CandidateValidationStats],
    config: V6CandidateConfig,
    tau_support: int,
    tau_gain: float,
    tau_key: float,
    tau_param_cosine: float = 0.9,
    tau_overlap: float = 0.8,
) -> List[CommitDecision]:
    """Commit 0/1/2 slots; redundant pairs keep the higher-gain slot."""
    if len(stats_list) > 2:
        raise ValueError("at most two candidate slots")
    decisions = []
    for stats in stats_list:
        reasons = []
        if stats.support_count < tau_support:
            reasons.append("support_below_tau_support")
        if stats.mean_gain < tau_gain:
            reasons.append("mean_gain_below_tau_gain")
        if stats.key_accuracy < tau_key:
            reasons.append("key_accuracy_below_tau_key")
        status = "rejected" if reasons else "provisional"
        decisions.append(
            CommitDecision(
                slot_id=stats.slot_id,
                status=status,
                reason=";".join(reasons) if reasons else "all_conditions_pass",
                stats=stats,
            )
        )
    # Redundancy: both provisional with high param cosine AND high overlap.
    provisional = [decision for decision in decisions if decision.status == "provisional"]
    if len(provisional) == 2 and config.slot_count == 2:
        left, right = provisional
        overlap = 0.0
        left_positive = set(left.stats.positive_sample_ids)
        right_positive = set(right.stats.positive_sample_ids)
        union = left_positive | right_positive
        if union:
            overlap = len(left_positive & right_positive) / len(union)
        param_cosine = left.stats.param_cosine
        if (
            param_cosine is not None
            and param_cosine >= tau_param_cosine
            and overlap >= tau_overlap
        ):
            keep, drop = sorted(
                (left, right), key=lambda decision: decision.stats.mean_gain, reverse=True
            )
            keep.reason = "kept_over_redundant_pair"
            drop.status = "redundant"
            drop.reason = "redundant_with_slot_{}_param_cosine={:.3f}_overlap={:.3f}".format(
                keep.slot_id, param_cosine, overlap
            )
    return decisions


def joint_effect(
    pool: CandidateExpertPool,
    records: Sequence[Dict[str, Any]],
    loss_fn: Callable[[Dict[str, Any], Tuple[int, ...]], float],
) -> Dict[str, float]:
    """Joint gain of both candidates together over the old teacher set."""
    if pool.config.slot_count != 2:
        return {"joint_gain": None, "joint_support": None}
    gains = []
    for record in records:
        old_set = tuple(sorted(int(value) for value in record["old_teacher_set"]))
        joint_set = tuple(sorted(set(old_set) | {0, 1}))
        gains.append(loss_fn(record, old_set) - loss_fn(record, joint_set))
    return {
        "joint_gain": float(statistics.mean(gains)) if gains else 0.0,
        "joint_support": sum(1 for gain in gains if gain > 0),
    }


def commit_candidates(
    transaction: CommitTransaction,
    registry,
    decisions: Sequence[CommitDecision],
    first_expert_id: int,
    creation_task: int,
    creation_task_name: str,
    created_seed: int,
    pool_version: int,
    config_hash: str,
    artifact_writers: Dict[int, Callable[[int, str], Dict[str, str]]],
    validation_reports: Dict[int, str],
    metadata_builder=None,
) -> List[CommitDecision]:
    """Transactional commit of the provisional decisions.

    For each provisional slot, ``artifact_writers[slot_id](expert_id,
    staging_dir)`` must write the LoRA checkpoint (+ key) and return
    ``{path: sha256}``. ``validation_reports[slot_id]`` is the path of the
    written validation report. Uses ``CommitTransaction`` (Stage E2):
    pending marker -> artifacts -> registry (provisional) -> pool_version
    bump. Idempotent under crash recovery.
    """
    committed = []
    for slot_index, decision in enumerate(decisions):
        if decision.status != "provisional":
            continue
        expert_id = int(first_expert_id) + slot_index
        if registry.contains(expert_id):
            raise ValueError("expert id {} already exists; refusing reuse".format(expert_id))
        staging = transaction.registry_dir / "expert_{:04d}".format(expert_id)
        staging.mkdir(parents=True, exist_ok=True)
        writer = artifact_writers[decision.slot_id]
        artifacts = writer(expert_id, str(staging))
        report = validation_reports.get(decision.slot_id)
        if report:
            artifacts[str(report)] = file_sha256(str(report))

        if metadata_builder is not None:
            metadata = metadata_builder(expert_id, decision, str(staging), artifacts)
        else:
            from compose.experts.metadata import ExpertLifecycleStatus, ExpertMetadata

            metadata = ExpertMetadata(
                expert_id=expert_id,
                adapter_name="expert_{:04d}".format(expert_id),
                rank=8,
                alpha=16.0,
                creation_task=int(creation_task),
                creation_task_name=str(creation_task_name),
                created_seed=int(created_seed),
                checkpoint_path=str(staging / "compose_experts.bin"),
                checkpoint_sha256=artifacts.get(
                    str(staging / "compose_experts.bin"), ""
                ),
                key_path=str(staging / "key.json"),
                key_sha256=artifacts.get(str(staging / "key.json"), ""),
                rms_stats_path=None,
                support_count=decision.stats.support_count,
                mean_conditional_gain=decision.stats.mean_gain,
                key_accuracy=decision.stats.key_accuracy,
                config_hash=str(config_hash),
                pool_version_created=int(pool_version),
                lifecycle_status=ExpertLifecycleStatus.CANDIDATE,
            )
        transaction.begin(expert_id, {"task_id": creation_task, "slot_id": decision.slot_id})
        result = transaction.complete(
            expert_id,
            artifacts=artifacts,
            condition_record={
                "task_id": creation_task,
                "support_count": decision.stats.support_count,
                "mean_conditional_gain": decision.stats.mean_gain,
                "key_accuracy": decision.stats.key_accuracy,
            },
            metadata=metadata,
        )
        decision.committed_expert_id = expert_id
        decision.reason = result["status"]
        committed.append(decision)
    return committed
