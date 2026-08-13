"""Contribution-supervised key refinement and lightweight set routing."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from .inference import predict_sets
from .set_losses import SetLossConfig, set_router_loss


def contribution_record(
    cluster_label: int,
    teacher_set: Sequence[int],
    new_expert_ids: Sequence[int],
    empty_loss: float,
    single_losses: Mapping[int, float],
    pair_losses: Mapping[Tuple[int, int], float],
) -> Dict[str, Any]:
    selected = tuple(sorted(int(value) for value in teacher_set))
    new_ids = set(int(value) for value in new_expert_ids)
    positives = tuple(value for value in selected if value in new_ids)
    best_single = min(single_losses, key=single_losses.get) if single_losses else None
    first_gain = (
        float(empty_loss) - float(single_losses[best_single])
        if best_single is not None else 0.0
    )
    second_gain = 0.0
    if len(selected) == 2 and selected in pair_losses and best_single is not None:
        second_gain = float(single_losses[best_single]) - float(pair_losses[selected])
    return {
        "cluster_label": int(cluster_label),
        "contribution_label": list(positives),
        "teacher_set": list(selected),
        "agreement": bool(int(cluster_label) in positives),
        "positive": bool(positives),
        "first_expert_gain": first_gain,
        "conditional_pair_gain": second_gain,
    }


def refine_new_keys_from_contribution(
    queries: Tensor,
    records: Sequence[Mapping[str, Any]],
    key_parameters,
    new_expert_ids: Sequence[int],
) -> Dict[str, Any]:
    """Move only new keys to centroids of teacher-positive train queries."""
    new_ids = [int(value) for value in new_expert_ids]
    before = {
        expert_id: key_parameters[str(expert_id)].detach().clone()
        for expert_id in new_ids
    }
    counts = {}
    for expert_id in new_ids:
        indices = [
            index for index, record in enumerate(records)
            if expert_id in set(map(int, record["contribution_label"]))
        ]
        counts[expert_id] = len(indices)
        if indices:
            centroid = F.normalize(queries[indices].float().mean(dim=0), dim=0)
            with torch.no_grad():
                key_parameters[str(expert_id)].copy_(centroid)
    return {
        "positive_count": sum(counts.values()),
        "negative_count": len(records) - sum(bool(record["contribution_label"]) for record in records),
        "per_expert_positive_count": {str(key): value for key, value in counts.items()},
        "historical_keys_touched": False,
        "new_key_change": {
            str(expert_id): float(
                (before[expert_id] - key_parameters[str(expert_id)].detach()).norm()
            ) for expert_id in new_ids
        },
    }


def train_contribution_set_router(
    set_router,
    queries: Tensor,
    keys: Tensor,
    expert_ids: Sequence[int],
    targets: Sequence[Sequence[int]],
    epochs: int = 40,
    learning_rate: float = 1.0e-3,
    anchor_queries: Tensor = None,
    anchor_targets: Sequence[Sequence[int]] = (),
) -> Dict[str, Any]:
    """Train only the small set router; queries, keys and backbone stay frozen."""
    queries = queries.detach().float()
    keys = keys.detach().float()
    previous_anchor_predictions = []
    if anchor_queries is not None and int(anchor_queries.shape[0]) > 0:
        anchor_queries = anchor_queries.detach().float()
        with torch.no_grad():
            anchor_visible = torch.ones(len(anchor_targets), len(expert_ids), dtype=torch.bool)
            previous_anchor_predictions = predict_sets(
                set_router(anchor_queries, keys, expert_ids, anchor_visible), 0.65
            )
        queries = torch.cat((queries, anchor_queries), dim=0)
        targets = list(targets) + [tuple(map(int, value)) for value in anchor_targets]
    visible = torch.ones(len(targets), len(expert_ids), dtype=torch.bool)
    optimizer = torch.optim.AdamW(set_router.parameters(), lr=float(learning_rate))
    final = None
    for _ in range(int(epochs)):
        output = set_router(queries, keys, expert_ids, visible)
        loss, parts = set_router_loss(output, targets, expert_ids, SetLossConfig())
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite contribution set-router loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final = (loss, parts)
    with torch.no_grad():
        output = set_router(queries, keys, expert_ids, visible)
        predicted = predict_sets(output, 0.65)
    exact = sum(tuple(sorted(map(int, target))) == tuple(sorted(prediction))
                for target, prediction in zip(targets, predicted))
    histogram = {size: sum(len(value) == size for value in predicted) for size in range(3)}
    anchor_drift = 0.0
    if previous_anchor_predictions:
        current_anchor_predictions = predicted[-len(previous_anchor_predictions):]
        anchor_drift = sum(
            tuple(before) != tuple(after)
            for before, after in zip(previous_anchor_predictions, current_anchor_predictions)
        ) / len(previous_anchor_predictions)
    return {
        "final_loss": float(final[0]),
        "loss_parts": {name: float(value) for name, value in final[1].items()},
        "SetExactAcc": exact / max(1, len(targets)),
        "empty_rate": histogram[0] / max(1, len(targets)),
        "single_rate": histogram[1] / max(1, len(targets)),
        "pair_rate": histogram[2] / max(1, len(targets)),
        "average_active_experts": sum(len(value) for value in predicted) / max(1, len(predicted)),
        "router_trainable_parameters": sum(value.numel() for value in set_router.parameters()),
        "historical_anchor_count": len(previous_anchor_predictions),
        "historical_route_decision_drift": anchor_drift,
    }


def update_route_anchors(
    previous: Sequence[Mapping[str, Any]],
    queries: Tensor,
    targets: Sequence[Sequence[int]],
    sample_ids: Sequence[str],
    max_per_cardinality: int = 32,
) -> Sequence[Dict[str, Any]]:
    """Keep deterministic, bounded train-only anchors for future router updates."""
    anchors = [dict(value) for value in previous]
    counts = {size: sum(len(value["target"]) == size for value in anchors) for size in range(3)}
    rows = sorted(
        zip(sample_ids, queries.detach().cpu(), targets), key=lambda value: str(value[0])
    )
    existing = {str(value["sample_id"]) for value in anchors}
    for sample_id, query, target in rows:
        size = len(target)
        if size not in counts or counts[size] >= int(max_per_cardinality) or str(sample_id) in existing:
            continue
        anchors.append({
            "sample_id": str(sample_id),
            "query": query.float().tolist(),
            "target": list(map(int, target)),
        })
        counts[size] += 1
        existing.add(str(sample_id))
    return anchors
