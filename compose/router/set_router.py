"""Top-M retrieval plus answer-free 0/1/2 expert set scoring."""

from dataclasses import dataclass
from itertools import combinations
from typing import List, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cardinality_head import CardinalityHead
from .pair_scorer import SymmetricPairScorer


@dataclass
class SetRouterOutput:
    cardinality_logits: Tensor
    candidate_expert_ids: List[Tuple[int, ...]]
    single_scores: List[Tensor]
    pair_ids: List[Tuple[Tuple[int, int], ...]]
    pair_scores: List[Tensor]
    top_similarities: Tensor


class ExpertSetRouter(nn.Module):
    def __init__(self, query_dim: int = 128, top_m: int = 8) -> None:
        super().__init__()
        self.query_dim, self.top_m = int(query_dim), int(top_m)
        self.cardinality = CardinalityHead(query_dim, top_m)
        self.pair_scorer = SymmetricPairScorer(query_dim)
        self.single_bias = nn.Parameter(torch.zeros(()))

    def forward(self, queries: Tensor, keys: Tensor, expert_ids: Sequence[int], visible_mask: Tensor) -> SetRouterOutput:
        if visible_mask.shape != (queries.shape[0], keys.shape[0]):
            raise ValueError("visible_mask must have shape [batch, experts]")
        similarities = F.normalize(queries, dim=-1) @ F.normalize(keys, dim=-1).T
        candidates, singles, pair_ids, pair_scores, top_rows = [], [], [], [], []
        for row in range(queries.shape[0]):
            visible = torch.where(visible_mask[row].bool())[0]
            k = min(self.top_m, int(visible.numel()))
            if k:
                scores, order = torch.topk(similarities[row, visible], k=k, sorted=True)
                indices = visible[order]
            else:
                scores = similarities.new_empty(0)
                indices = torch.empty(0, dtype=torch.long, device=queries.device)
            ids = tuple(int(expert_ids[index]) for index in indices.tolist())
            candidates.append(ids)
            singles.append(scores + self.single_bias)
            pairs = tuple(combinations(ids, 2))
            pair_ids.append(pairs)
            if pairs:
                local = {expert_id: offset for offset, expert_id in enumerate(ids)}
                first = torch.tensor([local[a] for a, _ in pairs], device=queries.device)
                second = torch.tensor([local[b] for _, b in pairs], device=queries.device)
                pair_scores.append(self.pair_scorer(
                    queries[row].expand(len(pairs), -1), keys[indices[first]], keys[indices[second]], scores[first], scores[second]
                ))
            else:
                pair_scores.append(similarities.new_empty(0))
            padded = similarities.new_full((self.top_m,), -1.0)
            padded[:len(scores)] = scores
            top_rows.append(padded)
        top = torch.stack(top_rows)
        counts = visible_mask.sum(dim=1)
        return SetRouterOutput(self.cardinality(queries, top, counts), candidates, singles, pair_ids, pair_scores, top)
