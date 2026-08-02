"""Order-invariant pair compatibility scorer."""

import torch
from torch import Tensor, nn


class SymmetricPairScorer(nn.Module):
    def __init__(self, query_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.query_dim = int(query_dim)
        self.network = nn.Sequential(nn.Linear(query_dim * 4 + 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, query: Tensor, key_i: Tensor, key_j: Tensor, sim_i: Tensor, sim_j: Tensor) -> Tensor:
        if query.shape != key_i.shape or query.shape != key_j.shape or query.shape[-1] != self.query_dim:
            raise ValueError("query and pair keys must share [batch, query_dim]")
        similarities = torch.stack((sim_i, sim_j), dim=-1)
        symmetric = torch.cat((
            query, key_i + key_j, torch.abs(key_i - key_j), key_i * key_j,
            similarities.max(dim=-1).values.unsqueeze(-1), similarities.min(dim=-1).values.unsqueeze(-1),
        ), dim=-1)
        return self.network(symmetric).squeeze(-1)
