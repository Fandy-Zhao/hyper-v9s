"""Answer-free independent multi-label Query-Key router for Compose P3."""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RouterThresholds:
    tau_none: float
    tau_second: float

    def __post_init__(self):
        if not 0.0 <= self.tau_none <= 1.0 or not 0.0 <= self.tau_second <= 1.0:
            raise ValueError("router thresholds must lie in [0, 1]")


class MultiLabelQueryKeyRouter(nn.Module):
    def __init__(self, input_dim: int, expert_count: int, query_dim: int = 128):
        super().__init__()
        if input_dim <= 0 or expert_count <= 0 or query_dim <= 0:
            raise ValueError("router dimensions must be positive")
        self.query_projector = nn.Linear(input_dim, query_dim)
        self.expert_keys = nn.Parameter(F.normalize(torch.randn(expert_count, query_dim), dim=-1))
        self.expert_bias = nn.Parameter(torch.zeros(expert_count))
        self.log_temperature = nn.Parameter(torch.zeros(expert_count))

    def forward(self, features: Tensor, visible_mask: Tensor = None) -> Tensor:
        queries = F.normalize(self.query_projector(features), dim=-1)
        keys = F.normalize(self.expert_keys, dim=-1)
        temperature = self.log_temperature.exp().clamp(0.05, 20.0)
        logits = (queries @ keys.T) / temperature + self.expert_bias
        if visible_mask is not None:
            if visible_mask.shape != logits.shape:
                raise ValueError("visible_mask must match router logits")
            logits = logits.masked_fill(~visible_mask.bool(), -30.0)
        return logits

    def select(self, features: Tensor, expert_ids: Sequence[int], thresholds: RouterThresholds, visible_mask: Tensor = None):
        if len(expert_ids) != self.expert_keys.shape[0]:
            raise ValueError("expert_ids must match router outputs")
        probabilities = torch.sigmoid(self(features, visible_mask))
        selected = []
        for row in probabilities:
            values, indices = torch.topk(row, k=min(2, row.numel()), sorted=True)
            if float(values[0]) < thresholds.tau_none:
                selected.append(())
            elif len(values) == 1 or float(values[1]) < thresholds.tau_second:
                selected.append((int(expert_ids[int(indices[0])]),))
            else:
                selected.append((int(expert_ids[int(indices[0])]), int(expert_ids[int(indices[1])])))
        return tuple(selected), probabilities


def router_loss(
    logits: Tensor,
    targets: Tensor,
    anchor_logits: Tensor = None,
    lambda_rank: float = 1.0,
    lambda_sparse: float = 0.01,
    lambda_anchor: float = 0.1,
    rank_margin: float = 0.1,
) -> Dict[str, Tensor]:
    if logits.shape != targets.shape:
        raise ValueError("logits and multi-label targets must have equal shape")
    bce = F.binary_cross_entropy_with_logits(logits, targets.float())
    rank_terms = []
    for row_logits, row_targets in zip(logits, targets.bool()):
        positive, negative = row_logits[row_targets], row_logits[~row_targets]
        if positive.numel() and negative.numel():
            rank_terms.append(F.relu(rank_margin - positive[:, None] + negative[None, :]).mean())
    ranking = torch.stack(rank_terms).mean() if rank_terms else logits.new_zeros(())
    sparse = torch.sigmoid(logits).sum(dim=-1).sub(targets.sum(dim=-1).clamp_max(2)).clamp_min(0).mean()
    anchor = logits.new_zeros(()) if anchor_logits is None else F.mse_loss(logits, anchor_logits.detach())
    total = bce + lambda_rank * ranking + lambda_sparse * sparse + lambda_anchor * anchor
    return {"total": total, "bce": bce, "ranking": ranking, "sparse": sparse, "anchor": anchor}


def tune_thresholds(validation_rows, candidates: Sequence[RouterThresholds], split: str):
    if "test" in split.lower():
        raise ValueError("test split cannot tune router thresholds")
    if not validation_rows or not candidates:
        raise ValueError("validation rows and threshold candidates are required")
    best = None
    for thresholds in candidates:
        exact = 0
        for row in validation_rows:
            scores = list(map(float, row["probabilities"]))
            order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:2]
            if scores[order[0]] < thresholds.tau_none:
                predicted = ()
            elif len(order) == 1 or scores[order[1]] < thresholds.tau_second:
                predicted = (order[0],)
            else:
                predicted = tuple(order)
            exact += predicted == tuple(row["teacher_set"])
        score = exact / len(validation_rows)
        candidate = (score, -thresholds.tau_none, -thresholds.tau_second, thresholds)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    return best[3], best[0]
