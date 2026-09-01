"""Validation-only usage, remove-and-reroute contribution and redundancy."""

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .config import V7PruningConfig
from .pool import V7ExpertKeyPool, pairwise_key_cosine
from .routing import GlobalTop2Router


@dataclass(frozen=True)
class CandidateValidation:
    expert_id: int
    train_usage: float
    val_usage: float
    selection_count: int
    key_norm: float
    key_to_task_center_cosine: float
    removal_gain_metric: float
    removal_gain_loss: float
    redundancy: Tuple[Mapping[str, object], ...]
    keep: bool
    reason: str


def route_usage(selected_ids: Tensor, expert_ids: Sequence[int]) -> Dict[int, float]:
    denominator = int(selected_ids.shape[0])
    if denominator == 0:
        raise ValueError("usage requires a non-empty split")
    return {
        int(expert_id): float(selected_ids.eq(int(expert_id)).any(dim=1).sum()) / denominator
        for expert_id in expert_ids
    }


class CandidatePruner:
    """The scorer consumes rerouted `[N,2]` ids and returns metric/loss."""

    def __init__(self, pool: V7ExpertKeyPool, config: V7PruningConfig) -> None:
        self.pool = pool
        self.router = GlobalTop2Router(pool)
        self.config = config

    def evaluate(
        self,
        train_queries: Tensor,
        val_queries: Tensor,
        task_center: Tensor,
        scorer: Callable[[Tensor], Mapping[str, float]],
    ) -> Tuple[Tuple[int, ...], Dict[int, Dict[str, object]], Dict[str, object]]:
        current_ids = self.pool.current_ids
        full_train = self.router(train_queries)
        full_val = self.router(val_queries)
        train_usage = route_usage(full_train.expert_ids, current_ids)
        val_usage = route_usage(full_val.expert_ids, current_ids)
        full_score = dict(scorer(full_val.expert_ids))
        if "metric" not in full_score or "loss" not in full_score:
            raise ValueError("pruning scorer must return metric and loss")
        removal: Dict[int, Dict[str, float]] = {}
        for expert_id in current_ids:
            rerouted = self.router(val_queries, excluded=(expert_id,))
            minus = dict(scorer(rerouted.expert_ids))
            removal[expert_id] = {
                "metric": float(minus["metric"]),
                "loss": float(minus["loss"]),
                "removal_gain_metric": float(full_score["metric"]) - float(minus["metric"]),
                "removal_gain_loss": float(minus["loss"]) - float(full_score["loss"]),
                "rerouted_expert_ids": rerouted.expert_ids.detach().cpu().tolist(),
            }

        current_keys = self.pool.normalized(current_ids)
        cosine = pairwise_key_cosine(current_keys)
        redundant: Dict[int, list] = {expert_id: [] for expert_id in current_ids}
        redundant_losers = set()
        minimum = float(self.config.candidate_min_removal_gain)
        for left in range(len(current_ids)):
            for right in range(left + 1, len(current_ids)):
                similarity = float(cosine[left, right])
                if similarity < self.config.candidate_redundancy_cosine:
                    continue
                left_id, right_id = current_ids[left], current_ids[right]
                left_contribution = max(
                    removal[left_id]["removal_gain_metric"],
                    removal[left_id]["removal_gain_loss"],
                )
                right_contribution = max(
                    removal[right_id]["removal_gain_metric"],
                    removal[right_id]["removal_gain_loss"],
                )
                # Redundancy only prunes when at least one removal is within
                # the configured near-zero contribution boundary.
                if min(left_contribution, right_contribution) <= minimum:
                    loser = right_id if left_contribution >= right_contribution else left_id
                    redundant_losers.add(loser)
                record = {
                    "other_expert_id": right_id,
                    "key_cosine": similarity,
                    "left_contribution": left_contribution,
                    "right_contribution": right_contribution,
                }
                redundant[left_id].append(record)
                redundant[right_id].append({**record, "other_expert_id": left_id})

        retained = []
        metrics: Dict[int, Dict[str, object]] = {}
        for index, expert_id in enumerate(current_ids):
            gain_metric = removal[expert_id]["removal_gain_metric"]
            gain_loss = removal[expert_id]["removal_gain_loss"]
            no_contribution = gain_metric <= minimum and gain_loss <= minimum
            if not self.config.candidate_prune_enabled:
                keep, reason = True, "pruning_disabled"
            elif expert_id in redundant_losers:
                keep, reason = False, "redundant_lower_removal_contribution"
            elif no_contribution:
                keep, reason = False, "no_validation_removal_contribution"
            else:
                keep, reason = True, "positive_validation_removal_contribution"
            # Usage is diagnostic only and never changes the keep decision.
            if val_usage[expert_id] <= self.config.candidate_min_usage:
                reason += ";dead_usage_diagnostic"
            if keep:
                retained.append(expert_id)
            record = CandidateValidation(
                expert_id=expert_id,
                train_usage=train_usage[expert_id],
                val_usage=val_usage[expert_id],
                selection_count=int(full_train.expert_ids.eq(expert_id).sum()),
                key_norm=float(self.pool.keys[str(expert_id)].detach().norm()),
                key_to_task_center_cosine=float(
                    torch.nn.functional.cosine_similarity(
                        self.pool.keys[str(expert_id)].detach(), task_center.detach(), dim=0
                    )
                ),
                removal_gain_metric=gain_metric,
                removal_gain_loss=gain_loss,
                redundancy=tuple(redundant[expert_id]),
                keep=keep,
                reason=reason,
            )
            metrics[expert_id] = {**asdict(record), **removal[expert_id]}
        audit = {
            "full": full_score,
            "candidate_key_cosine_matrix": cosine.cpu().tolist(),
            "train_routes": full_train.expert_ids.detach().cpu().tolist(),
            "val_routes": full_val.expert_ids.detach().cpu().tolist(),
            "thresholds": asdict(self.config),
        }
        return tuple(retained), metrics, audit

