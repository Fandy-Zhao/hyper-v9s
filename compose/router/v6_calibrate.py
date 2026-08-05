"""V6 global teacher re-generation and Router calibration (Stage E8).

After candidates commit, the old teacher cache is invalid (the cache key
already binds pool_version, so reads fail and the cache rebuilds). A new
calibration teacher is generated over the updated pool, converted to
multi-hot labels, and the Router is calibrated:

    L_router = L_BCE + lambda_rank*L_ranking + lambda_sparse*L_sparse
               + lambda_anchor*L_anchor

The Router only owns the query encoder, expert keys, per-expert bias and
temperature; backbone, vision encoder, multimodal projector and all LoRA
experts are frozen by construction (they are not parameters of the
Router). Historical anchors store frozen query features, teacher labels
and frozen Router logits. A Router failure is never worked around by
creating new experts.
"""

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .v6_router import V6Router


@dataclass(frozen=True)
class CalibrationConfig:
    lambda_bce: float = 1.0
    lambda_rank: float = 1.0
    lambda_sparse: float = 0.01
    lambda_anchor: float = 0.1
    rank_margin: float = 0.1
    epochs: int = 5
    learning_rate: float = 2.0e-4
    batch_size: int = 32
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        for name in ("lambda_bce", "lambda_rank", "lambda_sparse", "lambda_anchor", "rank_margin"):
            if getattr(self, name) < 0:
                raise ValueError("{} must be non-negative".format(name))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_multi_hot_labels(
    teacher_records: Sequence[Dict[str, Any]],
    pool_expert_ids: Sequence[int],
) -> Tuple[Tensor, Tuple[int, ...]]:
    """Multi-hot teacher labels over the whole pool: y_ik = 1 iff expert k
    is in sample i's teacher set."""
    pool_ids = tuple(sorted(int(value) for value in pool_expert_ids))
    if not pool_ids:
        raise ValueError("pool expert ids must not be empty")
    index = {expert_id: position for position, expert_id in enumerate(pool_ids)}
    labels = torch.zeros(len(teacher_records), len(pool_ids))
    for row, record in enumerate(teacher_records):
        for expert_id in record["teacher_set"]:
            if int(expert_id) not in index:
                raise ValueError(
                    "teacher expert {} not in pool {}".format(expert_id, pool_ids)
                )
            labels[row, index[int(expert_id)]] = 1.0
    return labels, pool_ids


@torch.no_grad()
def compute_anchor_logits(router: V6Router, queries: Tensor) -> Tensor:
    """Frozen Router logits for anchor regularization (detached)."""
    probabilities, ids = router._scores(queries, router.expert_ids)
    return probabilities.detach()


def _calibration_loss(
    probabilities: Tensor,
    multi_hot: Tensor,
    expert_ids: Tuple[int, ...],
    anchor_probabilities: Optional[Tensor],
    config: CalibrationConfig,
) -> Dict[str, Tensor]:
    if probabilities.shape != multi_hot.shape:
        raise ValueError("probabilities and multi-hot labels must align")
    bce = F.binary_cross_entropy(probabilities, multi_hot.float())
    rank_terms = []
    for row_probs, row_labels in zip(probabilities, multi_hot.bool()):
        positive, negative = row_probs[row_labels], row_probs[~row_labels]
        if positive.numel() and negative.numel():
            rank_terms.append(
                F.relu(config.rank_margin - positive[:, None] + negative[None, :]).mean()
            )
    ranking = torch.stack(rank_terms).mean() if rank_terms else probabilities.new_zeros(())
    sparse = (
        probabilities.sum(dim=-1) - multi_hot.sum(dim=-1).clamp_max(len(expert_ids))
    ).clamp_min(0).mean()
    anchor = (
        probabilities.new_zeros(())
        if anchor_probabilities is None
        else F.mse_loss(probabilities, anchor_probabilities)
    )
    total = (
        config.lambda_bce * bce
        + config.lambda_rank * ranking
        + config.lambda_sparse * sparse
        + config.lambda_anchor * anchor
    )
    return {
        "total": total, "bce": bce, "ranking": ranking,
        "sparse": sparse, "anchor": anchor,
    }


def calibrate_v6_router(
    router: V6Router,
    queries: Tensor,
    multi_hot: Tensor,
    config: CalibrationConfig,
    anchor_queries: Optional[Tensor] = None,
    anchor_logits: Optional[Tensor] = None,
) -> Dict[str, Any]:
    """Calibrate the Router on multi-hot teacher labels.

    Only the Router's own parameters are updated (query encoder, keys,
    bias, temperature); everything else is frozen by construction. When
    ``anchor_queries``/``anchor_logits`` are given, the anchor loss keeps
    the Router from drifting away from the historical decisions.
    """
    if queries.ndim != 2 or queries.shape[0] != multi_hot.shape[0]:
        raise ValueError("queries and labels must align")
    expert_ids = router.expert_ids
    if not expert_ids:
        raise ValueError("router has no experts; cannot calibrate")
    if multi_hot.shape[1] != len(expert_ids):
        raise ValueError(
            "multi-hot width {} does not match router experts {}".format(
                multi_hot.shape[1], len(expert_ids)
            )
        )
    if anchor_queries is not None and anchor_logits is None:
        anchor_logits = compute_anchor_logits(router, anchor_queries)

    optimizer = torch.optim.AdamW(router.parameters(), lr=config.learning_rate)
    history = []
    count = queries.shape[0]
    for epoch in range(config.epochs):
        generator = torch.Generator().manual_seed(config.seed + epoch)
        permutation = torch.randperm(count, generator=generator)
        epoch_loss = 0.0
        steps = 0
        for start in range(0, count, config.batch_size):
            indices = permutation[start: start + config.batch_size]
            batch_queries = queries[indices]
            batch_labels = multi_hot[indices]
            probabilities, _ = router._scores(batch_queries, expert_ids)
            anchor_batch = None
            if anchor_queries is not None:
                anchor_probs, _ = router._scores(anchor_queries, expert_ids)
                anchor_batch = anchor_probs
            losses = _calibration_loss(
                probabilities, batch_labels, expert_ids, anchor_batch, config
            )
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            epoch_loss += float(losses["total"].detach())
            steps += 1
        history.append(epoch_loss / max(steps, 1))
    return {
        "epochs": config.epochs,
        "loss_history": history,
        "final_loss": history[-1] if history else None,
        "config": config.to_dict(),
    }


def evaluate_v6_router(
    router: V6Router,
    queries: Tensor,
    teacher_records: Sequence[Dict[str, Any]],
    split: str = "validation",
    top_m: int = 8,
) -> Dict[str, Any]:
    """Set-level and retrieval-level metrics on a non-test split."""
    if "test" in str(split).lower():
        raise ValueError("test data cannot calibrate or evaluate router decisions")
    if queries.shape[0] != len(teacher_records):
        raise ValueError("queries and teacher records must align")
    expert_ids = router.expert_ids
    if not expert_ids:
        raise ValueError("router has no experts")
    probabilities, ids = router._scores(queries, expert_ids)
    selection = router.select(queries, expert_ids)
    counts = {
        "exact_set": 0, "empty_correct": 0, "empty_total": 0,
        "pair_recall_num": 0, "pair_total": 0,
        "recall1_num": 0, "recall2_num": 0, "teacher_nonempty": 0,
        "active_experts": 0,
    }
    per_expert = {expert_id: {"tp": 0, "pred": 0, "label": 0}
                  for expert_id in expert_ids}
    for row, (record, predicted) in enumerate(zip(teacher_records, selection.sets)):
        teacher = tuple(sorted(int(value) for value in record["teacher_set"]))
        counts["active_experts"] += len(predicted)
        if predicted == teacher:
            counts["exact_set"] += 1
        if not teacher:
            counts["empty_total"] += 1
            if not predicted:
                counts["empty_correct"] += 1
        else:
            counts["teacher_nonempty"] += 1
            top1 = tuple(predicted[:1])
            top2 = tuple(predicted[:2])
            if set(teacher).issubset(set(top1)):
                counts["recall1_num"] += 1
            if set(teacher).issubset(set(top2)):
                counts["recall2_num"] += 1
            if len(teacher) == 2:
                counts["pair_total"] += 1
                if set(teacher).issubset(set(top2)):
                    counts["pair_recall_num"] += 1
        predicted_set = set(predicted)
        teacher_set = set(teacher)
        for expert_id in expert_ids:
            stats = per_expert[expert_id]
            if expert_id in predicted_set:
                stats["pred"] += 1
            if expert_id in teacher_set:
                stats["label"] += 1
            if expert_id in predicted_set and expert_id in teacher_set:
                stats["tp"] += 1

    total = len(teacher_records)
    per_expert_metrics = {}
    for expert_id, stats in per_expert.items():
        precision = stats["tp"] / stats["pred"] if stats["pred"] else 0.0
        recall = stats["tp"] / stats["label"] if stats["label"] else 0.0
        per_expert_metrics[str(expert_id)] = {
            "precision": precision, "recall": recall, **stats
        }
    return {
        "samples": total,
        "SetExactAcc": counts["exact_set"] / total if total else 0.0,
        "EmptyAcc": (
            counts["empty_correct"] / counts["empty_total"]
            if counts["empty_total"] else None
        ),
        "ExpertRecall@1": (
            counts["recall1_num"] / counts["teacher_nonempty"]
            if counts["teacher_nonempty"] else None
        ),
        "ExpertRecall@2": (
            counts["recall2_num"] / counts["teacher_nonempty"]
            if counts["teacher_nonempty"] else None
        ),
        "PairRecall": (
            counts["pair_recall_num"] / counts["pair_total"]
            if counts["pair_total"] else None
        ),
        "average_active_experts": (
            counts["active_experts"] / total if total else 0.0
        ),
        "per_expert": per_expert_metrics,
        "top_m": top_m,
    }


def tune_v6_thresholds(
    router: V6Router,
    queries: Tensor,
    teacher_records: Sequence[Dict[str, Any]],
    candidates: Sequence[Tuple[float, float]],
    split: str = "validation",
) -> Tuple[Tuple[float, float], float]:
    """Grid-search (tau_none, tau_second) on validation; test is refused."""
    if "test" in str(split).lower():
        raise ValueError("test split cannot tune router thresholds")
    if not teacher_records:
        raise ValueError("validation rows are required")
    if not candidates:
        raise ValueError("threshold candidates are required")
    best = None
    for tau_none, tau_second in candidates:
        router.set_thresholds(tau_none=tau_none, tau_second=tau_second)
        metrics = evaluate_v6_router(router, queries, teacher_records, split=split)
        exact = metrics["SetExactAcc"]
        key = (exact, -tau_none, -tau_second)
        if best is None or key > best[0]:
            best = (key, (tau_none, tau_second))
    return best[1], best[0][0]
