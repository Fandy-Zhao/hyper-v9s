"""The V7 query coordinate system has no parameters and never changes."""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FixedQueryProvenance:
    module_hash: str = "v7_fixed_layernorm_concat_l2_v1"

    def to_dict(self):
        return {
            "kind": "v7_fixed_multimodal_query",
            "visual_dim": 768,
            "text_dim": 768,
            "query_dim": 1536,
            "trainable_parameter_count": 0,
            "module_hash": self.module_hash,
        }


class FixedMultimodalQuery(nn.Module):
    """q = L2Norm(concat(LayerNorm(z_visual), LayerNorm(z_text)))."""

    def __init__(self, visual_dim: int = 768, text_dim: int = 768) -> None:
        super().__init__()
        if visual_dim != 768 or text_dim != 768:
            raise ValueError("V7 requires 768-D visual and 768-D text features")
        self.visual_dim = int(visual_dim)
        self.text_dim = int(text_dim)
        self.query_dim = self.visual_dim + self.text_dim

    @staticmethod
    def _fixed_layer_norm(value: Tensor) -> Tensor:
        return F.layer_norm(value, (value.shape[-1],), weight=None, bias=None)

    def provenance(self) -> FixedQueryProvenance:
        return FixedQueryProvenance()

    def forward(self, z_visual: Tensor, z_text: Tensor) -> Tensor:
        if z_visual.ndim != 2 or z_text.ndim != 2:
            raise ValueError("features must be rank-2 [batch, dim]")
        if z_visual.shape != (z_text.shape[0], self.visual_dim):
            raise ValueError("visual features must have shape [B, 768]")
        if z_text.shape[1] != self.text_dim:
            raise ValueError("text features must have shape [B, 768]")
        fused = torch.cat(
            [self._fixed_layer_norm(z_visual), self._fixed_layer_norm(z_text)], dim=-1
        )
        # Explicit detach makes the no-learnable-query contract auditable even
        # when callers accidentally pass tensors from a trainable extractor.
        return F.normalize(fused, dim=-1).detach()


def full_train_task_center(
    queries: Tensor, num_train_samples: int
) -> Tuple[Tensor, dict]:
    if queries.ndim != 2 or queries.shape[1] != 1536:
        raise ValueError("V7 task-center queries must have shape [N, 1536]")
    if queries.shape[0] != int(num_train_samples):
        raise ValueError(
            "full-data center mismatch: num_train_samples={}, queries={}".format(
                num_train_samples, queries.shape[0]
            )
        )
    if not queries.shape[0]:
        raise ValueError("cannot compute a task center from an empty train split")
    center = F.normalize(queries.detach().float().mean(dim=0), dim=0)
    return center, {
        "num_train_samples": int(num_train_samples),
        "num_queries_used_for_center": int(queries.shape[0]),
    }
