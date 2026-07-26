import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceModalityRouter(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 32,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, route_feat: torch.Tensor, prior_alpha: torch.Tensor):
        eps = 1e-6
        prior_alpha = prior_alpha.clamp(eps, 1.0 - eps)
        prior_logits = torch.log(prior_alpha)
        delta_logits = self.net(route_feat)
        logits = prior_logits + self.residual_scale * delta_logits
        return F.softmax(logits, dim=-1)
