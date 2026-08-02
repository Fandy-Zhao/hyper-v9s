"""Predict Empty/Single/Pair cardinality from answer-free retrieval statistics."""

import torch
from torch import Tensor, nn


class CardinalityHead(nn.Module):
    def __init__(self, query_dim: int = 128, top_m: int = 8, hidden_dim: int = 128) -> None:
        super().__init__()
        self.query_dim, self.top_m = int(query_dim), int(top_m)
        self.network = nn.Sequential(
            nn.Linear(self.query_dim + self.top_m + 3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3)
        )

    def forward(self, query: Tensor, top_similarities: Tensor, visible_counts: Tensor) -> Tensor:
        if query.ndim != 2 or query.shape[1] != self.query_dim:
            raise ValueError("query must have shape [batch, query_dim]")
        if top_similarities.ndim != 2 or top_similarities.shape[0] != query.shape[0]:
            raise ValueError("top similarities must have shape [batch, candidates]")
        values = torch.zeros(query.shape[0], self.top_m, device=query.device, dtype=query.dtype)
        width = min(self.top_m, top_similarities.shape[1])
        values[:, :width] = top_similarities[:, :width]
        probabilities = torch.softmax(top_similarities, dim=-1) if top_similarities.shape[1] else top_similarities
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1, keepdim=True) if top_similarities.shape[1] else torch.zeros(query.shape[0], 1, device=query.device)
        margin = (values[:, :1] - values[:, 1:2]) if width > 1 else values[:, :1]
        count = visible_counts.to(device=query.device, dtype=query.dtype).view(-1, 1) / max(1, self.top_m)
        return self.network(torch.cat((query, values, margin, entropy, count), dim=-1))
