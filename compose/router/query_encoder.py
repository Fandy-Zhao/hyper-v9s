"""Multimodal Query encoder whose public input schema excludes answer-side data."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class QueryInputs:
    """Frozen inference-time features only; answers and task IDs are absent by design."""

    image_features: Optional[Tensor]
    text_features: Optional[Tensor]
    image_available: Tensor
    text_available: Tensor

    def validate(self) -> int:
        if self.image_features is None and self.text_features is None:
            raise ValueError("at least one modality feature tensor is required")
        batch = (self.image_features if self.image_features is not None else self.text_features).shape[0]
        for name, mask in (("image_available", self.image_available), ("text_available", self.text_available)):
            if mask.ndim != 1 or mask.shape[0] != batch:
                raise ValueError(f"{name} must have shape [batch]")
        for name, value in (("image_features", self.image_features), ("text_features", self.text_features)):
            if value is not None and (value.ndim != 2 or value.shape[0] != batch):
                raise ValueError(f"{name} must have shape [batch, feature_dim]")
        if torch.any((self.image_available <= 0) & (self.text_available <= 0)):
            raise ValueError("each sample must expose at least one inference-time modality")
        return int(batch)


class MultimodalQueryEncoder(nn.Module):
    """Projected image/text fusion followed by LayerNorm and L2 normalization."""

    def __init__(self, image_dim: int, text_dim: int, query_dim: int = 128) -> None:
        super().__init__()
        if query_dim != 128:
            raise ValueError("query_dim is frozen at 128")
        self.image_dim, self.text_dim, self.query_dim = int(image_dim), int(text_dim), int(query_dim)
        self.image_projection = nn.Linear(self.image_dim, self.query_dim)
        self.text_projection = nn.Linear(self.text_dim, self.query_dim)
        self.gate = nn.Sequential(nn.Linear(self.query_dim * 2 + 2, self.query_dim), nn.Sigmoid())
        self.layer_norm = nn.LayerNorm(self.query_dim)

    def forward(self, inputs: QueryInputs) -> Tensor:
        batch = inputs.validate()
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        image = torch.zeros(batch, self.image_dim, device=device, dtype=dtype)
        text = torch.zeros(batch, self.text_dim, device=device, dtype=dtype)
        if inputs.image_features is not None:
            image = inputs.image_features.to(device=device, dtype=dtype)
        if inputs.text_features is not None:
            text = inputs.text_features.to(device=device, dtype=dtype)
        image_mask = inputs.image_available.to(device=device, dtype=dtype).view(batch, 1)
        text_mask = inputs.text_available.to(device=device, dtype=dtype).view(batch, 1)
        image_hidden = self.image_projection(image) * image_mask
        text_hidden = self.text_projection(text) * text_mask
        gate = self.gate(torch.cat((image_hidden, text_hidden, image_mask, text_mask), dim=-1))
        both = image_mask * text_mask
        fused = both * (gate * image_hidden + (1.0 - gate) * text_hidden)
        fused = fused + (1.0 - both) * (image_hidden + text_hidden)
        return F.normalize(self.layer_norm(fused), dim=-1)


def prompt_only_mask(input_ids: Tensor, attention_mask: Tensor, answer_start: Tensor) -> Tensor:
    """Return a token mask truncated before the first answer token for every sample."""
    if input_ids.shape != attention_mask.shape or input_ids.ndim != 2:
        raise ValueError("input_ids and attention_mask must share [batch, sequence] shape")
    if answer_start.shape != (input_ids.shape[0],):
        raise ValueError("answer_start must have shape [batch]")
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    result = attention_mask.bool() & positions.lt(answer_start.to(input_ids.device).unsqueeze(1))
    if torch.any(answer_start < 0) or torch.any(answer_start > input_ids.shape[1]):
        raise ValueError("answer_start lies outside the token sequence")
    return result
