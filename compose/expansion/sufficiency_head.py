"""Deployment-side sufficiency head with no answer/teacher/task-ID inputs."""

import torch
from torch import Tensor, nn


class SufficiencyHead(nn.Module):
    def __init__(self, query_dim: int = 128, top_m: int = 8, hidden_dim: int = 128) -> None:
        super().__init__()
        self.query_dim, self.top_m = int(query_dim), int(top_m)
        # query + cardinality(3) + top-M + margin + entropy + predicted score + visible count
        self.network = nn.Sequential(
            nn.Linear(query_dim + 3 + top_m + 4, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1)
        )

    def forward(self, query: Tensor, cardinality_logits: Tensor, top_similarities: Tensor,
                similarity_margin: Tensor, entropy: Tensor, predicted_set_score: Tensor, visible_count: Tensor) -> Tensor:
        if query.ndim != 2 or query.shape[1] != self.query_dim or cardinality_logits.shape != (query.shape[0], 3):
            raise ValueError("invalid sufficiency input shapes")
        top = torch.zeros(query.shape[0], self.top_m, device=query.device, dtype=query.dtype)
        width = min(self.top_m, top_similarities.shape[1])
        top[:, :width] = top_similarities[:, :width]
        scalars = [value.to(query).reshape(-1, 1) for value in (similarity_margin, entropy, predicted_set_score, visible_count)]
        return self.network(torch.cat((query, cardinality_logits.to(query), top, *scalars), dim=-1)).squeeze(-1)
