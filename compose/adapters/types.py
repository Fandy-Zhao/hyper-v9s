from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ComposeSelection:
    """Sparse per-sample expert selection.

    Both tensors have shape ``[batch_size, top_k]``. The foundation supports
    top-1 and top-2 execution; gates are normalized per sample on construction.
    """

    expert_ids: torch.LongTensor
    gates: torch.FloatTensor

    def __post_init__(self) -> None:
        if self.expert_ids.ndim != 2 or self.gates.ndim != 2:
            raise ValueError("expert_ids and gates must be rank-2 tensors")
        if self.expert_ids.shape != self.gates.shape:
            raise ValueError("expert_ids and gates must have identical shapes")
        if self.expert_ids.shape[1] not in (1, 2):
            raise ValueError("Compose foundation supports top_k equal to 1 or 2")
        if self.expert_ids.dtype != torch.long:
            raise TypeError("expert_ids must use torch.long")
        if not self.gates.is_floating_point():
            raise TypeError("gates must be floating point")
        if torch.any(self.expert_ids < 0):
            raise ValueError("expert_ids must be non-negative")
        if not torch.isfinite(self.gates).all():
            raise ValueError("gates must be finite")
        if torch.any(self.gates < 0):
            raise ValueError("gates must be non-negative")
        gate_sum = self.gates.sum(dim=1, keepdim=True)
        if torch.any(gate_sum <= 0):
            raise ValueError("each sample must have a positive gate sum")
        object.__setattr__(self, "gates", self.gates / gate_sum)

    @property
    def batch_size(self) -> int:
        return self.expert_ids.shape[0]

    @property
    def top_k(self) -> int:
        return self.expert_ids.shape[1]

    def to(self, device: torch.device) -> "ComposeSelection":
        return ComposeSelection(
            expert_ids=self.expert_ids.to(device=device),
            gates=self.gates.to(device=device),
        )
