"""Multi-positive retrieval, Empty margin, diversity, and anchor losses."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class RetrievalLossConfig:
    temperature: float = 0.07
    empty_margin: float = 0.20
    lambda_empty: float = 1.0
    lambda_diversity: float = 0.01
    lambda_anchor: float = 1.0


def multi_positive_retrieval_loss(similarities: Tensor, positive_mask: Tensor, visible_mask: Tensor, temperature: float = 0.07) -> Tensor:
    if similarities.shape != positive_mask.shape or similarities.shape != visible_mask.shape:
        raise ValueError("similarities, positive_mask, and visible_mask must share shape")
    valid_rows = positive_mask.any(dim=1)
    if not torch.any(valid_rows):
        return similarities.sum() * 0.0
    logits = similarities / float(temperature)
    logits = logits.masked_fill(~visible_mask.bool(), float("-inf"))
    numerator = torch.logsumexp(logits.masked_fill(~positive_mask.bool(), float("-inf")), dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return (denominator[valid_rows] - numerator[valid_rows]).mean()


def empty_margin_loss(similarities: Tensor, empty_mask: Tensor, visible_mask: Tensor, margin: float = 0.20) -> Tensor:
    rows = empty_mask.bool() & visible_mask.any(dim=1)
    if not torch.any(rows):
        return similarities.sum() * 0.0
    masked = similarities.masked_fill(~visible_mask.bool(), float("-inf"))
    return F.relu(masked.max(dim=1).values[rows] - float(margin)).mean()


def key_diversity_loss(keys: Tensor) -> Tensor:
    if keys.shape[0] < 2:
        return keys.sum() * 0.0
    normalized = F.normalize(keys, dim=-1)
    gram = normalized @ normalized.T
    off_diagonal = gram - torch.eye(len(keys), device=keys.device, dtype=keys.dtype)
    return off_diagonal.square().sum() / (len(keys) * (len(keys) - 1))


def retrieval_objective(similarities: Tensor, positive_mask: Tensor, visible_mask: Tensor, keys: Tensor,
                        config: RetrievalLossConfig = RetrievalLossConfig(), anchor_loss: Tensor = None):
    empty = ~positive_mask.any(dim=1)
    retrieval = multi_positive_retrieval_loss(similarities, positive_mask, visible_mask, config.temperature)
    empty_value = empty_margin_loss(similarities, empty, visible_mask, config.empty_margin)
    diversity = key_diversity_loss(keys)
    anchor = similarities.sum() * 0.0 if anchor_loss is None else anchor_loss
    total = retrieval + config.lambda_empty * empty_value + config.lambda_diversity * diversity + config.lambda_anchor * anchor
    return total, {"retrieval": retrieval, "empty": empty_value, "diversity": diversity, "anchor": anchor}
