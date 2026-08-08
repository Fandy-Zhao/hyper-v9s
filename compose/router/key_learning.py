"""Cluster-supervised learnable expert keys.

For residual sample i with frozen cluster label c_i = m:

- positive: the new key e_m;
- in-task negatives: the other new keys of the current task;
- hard negatives: the sample's historical Top-M keys (frozen).

Losses:

    L_cluster = CE(cos(q_i, new_keys) / tau_key, cluster_label)   # 0 when 1 cluster
    L_old_neg = mean relu(margin - cos(q_i, e_positive) + cos(q_i, e_old_neg))
    L_div      = mean_{m != n} relu(cos(e_m, e_n) - key_separation_margin)^2

    L_key = L_cluster + lambda_old * L_old_neg + lambda_div * L_div

The key optimizer only contains the current task's new keys. Old keys,
the query encoder and LoRA parameters are never updated here.

``key_mode == "prototype"`` skips training entirely: the key stays the
normalized cluster centroid.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

KEY_LEARNING_VERSION = 1


@dataclass(frozen=True)
class ComposeKeyLearningConfig:
    learning_rate: float = 3.0e-4
    epochs: int = 50
    temperature: float = 0.07
    margin: float = 0.3
    lambda_old: float = 0.5
    lambda_div: float = 0.1
    key_separation_margin: float = 0.1
    batch_size: int = 256
    seed: int = 42
    key_mode: str = "learnable"  # "prototype" | "learnable"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.margin < 0:
            raise ValueError("margin must be non-negative")
        if self.lambda_old < 0 or self.lambda_div < 0:
            raise ValueError("lambda weights must be non-negative")
        if self.key_mode not in ("prototype", "learnable"):
            raise ValueError("key_mode must be one of prototype, learnable")

    def to_dict(self) -> Dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "temperature": self.temperature,
            "margin": self.margin,
            "lambda_old": self.lambda_old,
            "lambda_div": self.lambda_div,
            "key_separation_margin": self.key_separation_margin,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "key_mode": self.key_mode,
        }


@dataclass
class KeyLearningResult:
    expert_id: int
    key_mode: str
    final_loss: float
    positive_similarity_before: float
    positive_similarity_after: float
    old_negative_similarity_before: float
    old_negative_similarity_after: float
    epochs_run: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "expert_id": self.expert_id,
            "key_mode": self.key_mode,
            "final_loss": self.final_loss,
            "positive_similarity_before": self.positive_similarity_before,
            "positive_similarity_after": self.positive_similarity_after,
            "old_negative_similarity_before": self.old_negative_similarity_before,
            "old_negative_similarity_after": self.old_negative_similarity_after,
            "epochs_run": self.epochs_run,
        }


def cluster_ce_loss(
    queries: Tensor,
    new_keys: Tensor,
    labels: Tensor,
    temperature: float,
) -> Tensor:
    """CrossEntropy over cosine similarities to the current task's keys.

    ``queries`` [N, D], ``new_keys`` [K, D] (normalized), ``labels`` [N]
    in [0, K). Returns 0 when K <= 1.
    """
    if new_keys.shape[0] <= 1:
        return queries.sum() * 0.0
    logits = queries @ new_keys.T / temperature
    return F.cross_entropy(logits, labels)


def old_negative_hinge_loss(
    queries: Tensor,
    positive_key: Tensor,
    old_negative_keys: Tensor,
    margin: float,
) -> Tensor:
    """relu(margin - cos(q, e_pos) + cos(q, e_neg)) over the sample's
    historical Top-M keys (hard negatives)."""
    if old_negative_keys.shape[0] == 0:
        return queries.sum() * 0.0
    positive_sim = (queries * positive_key.unsqueeze(0)).sum(dim=1)  # [N]
    negative_sims = queries @ old_negative_keys.T  # [N, H]
    margins = margin - positive_sim.unsqueeze(1) + negative_sims
    return F.relu(margins).mean()


def key_divergence_loss(new_keys: Tensor, separation_margin: float) -> Tensor:
    """Mean relu(cos(e_m, e_n) - margin)^2 over distinct new-key pairs."""
    if new_keys.shape[0] <= 1:
        return new_keys.sum() * 0.0
    similarities = new_keys @ new_keys.T
    mask = ~torch.eye(new_keys.shape[0], dtype=torch.bool, device=similarities.device)
    violations = F.relu(similarities[mask] - separation_margin)
    return (violations ** 2).mean()


def key_loss_total(
    queries: Tensor,
    labels: Tensor,
    new_keys: Tensor,
    old_negative_keys: Tensor,
    config: ComposeKeyLearningConfig,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Total key learning loss with per-term breakdown."""
    ce = cluster_ce_loss(queries, new_keys, labels, config.temperature)
    if old_negative_keys.shape[0] > 0:
        positive = new_keys[labels]  # [N, D] per-sample positive key
        positive_sim = (queries * positive).sum(dim=1)  # [N]
        negative_sims = queries @ old_negative_keys.T  # [N, H]
        margins = config.margin - positive_sim.unsqueeze(1) + negative_sims
        old_neg = F.relu(margins).mean()
    else:
        old_neg = queries.sum() * 0.0
    div = key_divergence_loss(new_keys, config.key_separation_margin)
    total = ce + config.lambda_old * old_neg + config.lambda_div * div
    terms = {
        "cluster_ce": ce.detach(),
        "old_negative_hinge": old_neg.detach(),
        "divergence": div.detach(),
        "total": total.detach(),
    }
    return total, terms


def learn_cluster_keys(
    queries: Tensor,
    cluster_labels: Tensor,
    old_negative_keys: Tensor,
    new_keys: nn.ParameterDict,
    new_expert_ids: Sequence[int],
    config: ComposeKeyLearningConfig,
    device: str = "cpu",
) -> Dict[int, KeyLearningResult]:
    """Train the current task's new keys; prototype mode skips training.

    ``queries`` [N, D] normalized functional queries of residual samples;
    ``cluster_labels`` [N] mapping to ``new_expert_ids`` (positions).
    ``old_negative_keys`` [H, D] normalized historical Top-M keys.
    ``new_keys`` is the router's key ParameterDict; only entries for
    ``new_expert_ids`` receive gradients.
    """
    new_expert_ids = [int(value) for value in new_expert_ids]
    if len(new_expert_ids) < 1:
        raise ValueError("at least one new expert key is required")
    queries = queries.detach().float().to(device)
    cluster_labels = cluster_labels.detach().long().to(device)
    old_negative_keys = old_negative_keys.detach().float().to(device)
    if queries.shape[0] != cluster_labels.shape[0]:
        raise ValueError("queries and cluster labels must align")

    key_tensors = [
        new_keys[str(expert_id)] for expert_id in new_expert_ids
    ]
    results = {}
    if config.key_mode == "prototype":
        for index, expert_id in enumerate(new_expert_ids):
            results[int(expert_id)] = KeyLearningResult(
                expert_id=int(expert_id),
                key_mode="prototype",
                final_loss=0.0,
                positive_similarity_before=0.0,
                positive_similarity_after=0.0,
                old_negative_similarity_before=0.0,
                old_negative_similarity_after=0.0,
                epochs_run=0,
            )
        return results

    optimizer = torch.optim.AdamW(key_tensors, lr=config.learning_rate)
    torch.manual_seed(config.seed)
    batches = torch.randperm(queries.shape[0])
    batch_size = max(1, min(config.batch_size, queries.shape[0]))

    def _stats():
        positive_sims = []
        old_neg_sims = []
        for batch_start in range(0, queries.shape[0], batch_size):
            batch = queries[batch_start : batch_start + batch_size]
            labels = cluster_labels[batch_start : batch_start + batch_size]
            # labels are POSITIONS into new_expert_ids, while new_keys is
            # keyed by expert-id strings; when the two differ (e.g. a single
            # new expert id 7 with a historical key "0" in the store) the
            # label string must be mapped through new_expert_ids, or the
            # recorded similarity is computed against the wrong key.
            positive = torch.stack(
                [
                    F.normalize(
                        new_keys[str(new_expert_ids[int(label)])], dim=0
                    )
                    for label in labels
                ]
            )
            positive_sims.append((batch * positive).sum(dim=1).mean().item())
            if old_negative_keys.shape[0] > 0:
                old_neg_sims.append((batch @ old_negative_keys.T).max(dim=1).values.mean().item())
        return (
            sum(positive_sims) / len(positive_sims) if positive_sims else 0.0,
            sum(old_neg_sims) / len(old_neg_sims) if old_neg_sims else 0.0,
        )

    pos_before, old_before = _stats()
    final_loss = 0.0
    epochs_run = 0
    for epoch in range(config.epochs):
        for batch_start in range(0, queries.shape[0], batch_size):
            indices = batches[batch_start : batch_start + batch_size]
            batch_queries = queries[indices]
            batch_labels = cluster_labels[indices]
            new_keys_stacked = torch.stack(
                [F.normalize(new_keys[str(expert_id)], dim=0) for expert_id in new_expert_ids]
            )
            loss, _ = key_loss_total(
                batch_queries,
                batch_labels,
                new_keys_stacked,
                old_negative_keys,
                config,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for key in key_tensors:
                with torch.no_grad():
                    key.data = F.normalize(key.data, dim=0)
            final_loss = float(loss.detach().item())
        epochs_run = epoch + 1
    pos_after, old_after = _stats()
    for index, expert_id in enumerate(new_expert_ids):
        results[int(expert_id)] = KeyLearningResult(
            expert_id=int(expert_id),
            key_mode="learnable",
            final_loss=final_loss,
            positive_similarity_before=pos_before,
            positive_similarity_after=pos_after,
            old_negative_similarity_before=old_before,
            old_negative_similarity_after=old_after,
            epochs_run=epochs_run,
        )
    return results


def initialize_key_from_centroid(
    key_parameter: nn.Parameter, centroid: Sequence[float]
) -> None:
    """Set a key from the normalized cluster centroid (initialization only)."""
    tensor = torch.tensor(list(centroid), dtype=key_parameter.dtype)
    with torch.no_grad():
        key_parameter.copy_(F.normalize(tensor.float(), dim=0))
