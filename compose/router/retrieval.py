"""Deterministic temporally masked Query-Key retrieval."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class RetrievalResult:
    expert_ids: Tensor
    similarities: Tensor
    visible_expert_ids: Tuple[int, ...]


def retrieve_experts(queries: Tensor, keys: Tensor, expert_ids: Sequence[int], visible_mask: Tensor, top_k: int) -> RetrievalResult:
    if queries.ndim != 2 or keys.ndim != 2 or queries.shape[1] != keys.shape[1]:
        raise ValueError("queries and keys must be compatible matrices")
    if keys.shape[0] != len(expert_ids) or visible_mask.shape != (len(expert_ids),):
        raise ValueError("expert IDs and visible_mask must match key rows")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    visible_indices = torch.where(visible_mask.bool())[0]
    if visible_indices.numel() == 0:
        empty_ids = torch.empty(queries.shape[0], 0, dtype=torch.long, device=queries.device)
        return RetrievalResult(empty_ids, empty_ids.to(dtype=queries.dtype), ())
    visible_keys = keys.index_select(0, visible_indices)
    similarities = F.normalize(queries, dim=-1) @ F.normalize(visible_keys, dim=-1).T
    k = min(int(top_k), int(visible_indices.numel()))
    # Stable expert-ID epsilon gives deterministic tie-breaking without changing practical scores.
    tie = torch.arange(similarities.shape[1], device=similarities.device, dtype=similarities.dtype) * -1e-7
    values, local = torch.topk(similarities + tie, k=k, dim=-1, largest=True, sorted=True)
    visible_ids = torch.tensor([int(expert_ids[index]) for index in visible_indices.tolist()], device=queries.device)
    selected_ids = visible_ids[local]
    raw_values = similarities.gather(1, local)
    return RetrievalResult(selected_ids, raw_values, tuple(int(value) for value in visible_ids.tolist()))
