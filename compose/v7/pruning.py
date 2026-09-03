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
    removal_gain_metric: Optional[float]
    removal_gain_loss: Optional[float]
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
        current_keys = self.pool.normalized(current_ids)
        cosine = pairwise_key_cosine(current_keys)
        minimum = float(self.config.candidate_min_removal_gain)
        excluded = set()
        trajectory = []
        latest: Dict[int, Dict[str, object]] = {}
        prune_reasons: Dict[int, str] = {}
        iteration = 0
        initial_full_score = None
        while True:
            surviving = tuple(value for value in current_ids if value not in excluded)
            full_routes = self.router(val_queries, excluded=excluded)
            full_score = dict(scorer(full_routes.expert_ids))
            if "metric" not in full_score or "loss" not in full_score:
                raise ValueError("pruning scorer must return metric and loss")
            if initial_full_score is None:
                initial_full_score = dict(full_score)
            removal = {}
            scored_entries = []
            for expert_id in surviving:
                pool_after = tuple(
                    value for value in self.pool.selectable_ids(excluded)
                    if value != expert_id
                )
                if len(pool_after) < 2:
                    removal[expert_id] = {
                        "protected": True,
                        "removal_gain_metric": None,
                        "removal_gain_loss": None,
                        "rerouted_expert_ids": [],
                        "full_scorer": dict(full_score),
                    }
                    continue
                rerouted = self.router(val_queries, excluded=excluded | {expert_id})
                # Two-phase iteration (spec 0903 S5): every removal hypothesis
                # of this iteration is submitted before any score is consumed,
                # so a parallel scorer (adaptive multi-GPU execution) can run
                # the full/minus jobs of one iteration on separate GPUs while
                # the serial remove-and-reroute trajectory stays exact.  A
                # synchronous scorer executes inline at submission, so legacy
                # behavior is bit-identical.
                scored_entries.append(
                    (expert_id, rerouted, dict(scorer(rerouted.expert_ids)))
                )
            for expert_id, rerouted, minus in scored_entries:
                removal[expert_id] = {
                    "protected": False,
                    "metric": float(minus["metric"]),
                    "loss": float(minus["loss"]),
                    "removal_gain_metric": float(full_score["metric"]) - float(minus["metric"]),
                    "removal_gain_loss": float(minus["loss"]) - float(full_score["loss"]),
                    "rerouted_expert_ids": rerouted.expert_ids.detach().cpu().tolist(),
                    "scorer": minus,
                    "full_scorer": dict(full_score),
                }

            redundant: Dict[int, list] = {expert_id: [] for expert_id in surviving}
            redundant_losers = set()
            for left_index, left_id in enumerate(surviving):
                for right_id in surviving[left_index + 1:]:
                    if removal[left_id]["protected"] or removal[right_id]["protected"]:
                        continue
                    left_slot, right_slot = current_ids.index(left_id), current_ids.index(right_id)
                    similarity = float(cosine[left_slot, right_slot])
                    if similarity < self.config.candidate_redundancy_cosine:
                        continue
                    left_contribution = max(
                        removal[left_id]["removal_gain_metric"],
                        removal[left_id]["removal_gain_loss"],
                    )
                    right_contribution = max(
                        removal[right_id]["removal_gain_metric"],
                        removal[right_id]["removal_gain_loss"],
                    )
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

            removable = []
            for expert_id in surviving:
                candidate = removal[expert_id]
                gain_metric = candidate["removal_gain_metric"]
                gain_loss = candidate["removal_gain_loss"]
                no_contribution = (
                    not candidate["protected"]
                    and gain_metric <= minimum and gain_loss <= minimum
                )
                redundant_loser = expert_id in redundant_losers
                safe = (
                    self.config.candidate_prune_enabled
                    and not candidate["protected"]
                    and (no_contribution or redundant_loser)
                )
                reason = (
                    "redundant_lower_removal_contribution"
                    if redundant_loser else "no_validation_removal_contribution"
                    if no_contribution else "positive_validation_removal_contribution"
                )
                if candidate["protected"]:
                    reason = "retained_for_global_top2_minimum_pool"
                latest[expert_id] = {
                    **candidate,
                    "redundancy": tuple(redundant[expert_id]),
                    "iteration": iteration,
                    "reason": reason,
                }
                if safe:
                    removable.append((max(gain_metric, gain_loss), expert_id, reason))

            selected_for_removal = min(removable) if removable else None
            for expert_id in surviving:
                candidate = removal[expert_id]
                remove_now = selected_for_removal is not None and expert_id == selected_for_removal[1]
                trajectory.append({
                    "iteration": iteration,
                    "pool_before": list(self.pool.selectable_ids(excluded)),
                    "candidate": expert_id,
                    "metric_full": float(full_score["metric"]),
                    "metric_minus_candidate": candidate.get("metric"),
                    "loss_full": float(full_score["loss"]),
                    "loss_minus_candidate": candidate.get("loss"),
                    "removal_gain_metric": candidate["removal_gain_metric"],
                    "removal_gain_loss": candidate["removal_gain_loss"],
                    "decision": "remove" if remove_now else latest[expert_id]["reason"],
                    "pool_after": list(
                        self.pool.selectable_ids(excluded | ({expert_id} if remove_now else set()))
                    ),
                })
            if selected_for_removal is None:
                break
            _, removed_id, removed_reason = selected_for_removal
            excluded.add(removed_id)
            prune_reasons[removed_id] = removed_reason
            iteration += 1

        retained = tuple(value for value in current_ids if value not in excluded)
        if len(self.pool.historical_ids) + len(retained) < 2:
            raise AssertionError("iterative pruning violated the Global Top-2 minimum pool")
        metrics: Dict[int, Dict[str, object]] = {}
        for expert_id in current_ids:
            keep = expert_id in retained
            detail = latest[expert_id]
            reason = detail["reason"] if keep else prune_reasons[expert_id]
            if val_usage[expert_id] <= self.config.candidate_min_usage:
                reason += ";dead_usage_diagnostic"
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
                removal_gain_metric=detail["removal_gain_metric"],
                removal_gain_loss=detail["removal_gain_loss"],
                redundancy=tuple(detail["redundancy"]),
                keep=keep,
                reason=reason,
            )
            full_scorer = detail.get("full_scorer") or {}
            minus_scorer = detail.get("scorer") or {}
            metrics[expert_id] = {
                **asdict(record),
                **detail,
                "official_metric_full": full_scorer.get("official_metric"),
                "official_metric_without_candidate": minus_scorer.get("official_metric"),
                "removal_gain_official_metric": detail["removal_gain_metric"],
                "answer_nll_full": full_scorer.get("answer_nll", full_scorer.get("loss")),
                "answer_nll_without_candidate": minus_scorer.get(
                    "answer_nll", minus_scorer.get("loss")
                ),
                "removal_gain_answer_nll": detail["removal_gain_loss"],
            }
        audit = {
            "full": initial_full_score,
            "candidate_key_cosine_matrix": cosine.cpu().tolist(),
            "train_routes": full_train.expert_ids.detach().cpu().tolist(),
            "val_routes": full_val.expert_ids.detach().cpu().tolist(),
            "thresholds": asdict(self.config),
            "pruning_trajectory": trajectory,
            "iterations": iteration + 1,
            "final_selectable_pool": list(self.pool.selectable_ids(excluded)),
        }
        return retained, metrics, audit

