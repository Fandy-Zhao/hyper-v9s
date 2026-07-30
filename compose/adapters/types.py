from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ComposeSelection:
    """Sparse per-sample expert selection.

    Both tensors have shape ``[batch_size, top_k]``. The foundation supports
    top-1 and top-2 execution. Gate normalization is always explicit.
    """

    expert_ids: torch.LongTensor
    gates: torch.FloatTensor
    normalization: str = "none"

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
        if self.expert_ids.shape[1] == 2 and torch.any(
            self.expert_ids[:, 0] == self.expert_ids[:, 1]
        ):
            raise ValueError("duplicate expert IDs per sample are not allowed")
        if self.normalization not in ("none", "l1", "l2"):
            raise ValueError(
                "normalization must be one of none, l1, or l2; got {!r}".format(
                    self.normalization
                )
            )
        positive_count = (self.gates > 0).sum(dim=1)
        if torch.any(positive_count == 0):
            raise ValueError("each sample must have at least one positive gate")
        if self.normalization == "l1":
            denominator = self.gates.sum(dim=1, keepdim=True)
            object.__setattr__(self, "gates", self.gates / denominator)
        elif self.normalization == "l2":
            denominator = torch.linalg.vector_norm(self.gates, ord=2, dim=1, keepdim=True)
            object.__setattr__(self, "gates", self.gates / denominator)

    @property
    def batch_size(self) -> int:
        return self.expert_ids.shape[0]

    @property
    def top_k(self) -> int:
        return self.expert_ids.shape[1]

    def to(self, device: torch.device) -> "ComposeSelection":
        if self.expert_ids.device == device and self.gates.device == device:
            return self
        return ComposeSelection(
            expert_ids=self.expert_ids.to(device=device),
            gates=self.gates.to(device=device),
            normalization=self.normalization,
        )
