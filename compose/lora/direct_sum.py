"""Parameter-free direct addition of at most two LoRA deltas."""

from typing import Sequence, Tuple

import torch


def direct_sum(base: torch.Tensor, deltas: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, Tuple[float, ...]]:
    """Return ``base + sum(deltas)`` with explicit unit coefficients."""
    if len(deltas) > 2:
        raise ValueError("direct_sum supports at most two expert deltas")
    # Match the frozen controlled implementation's bf16 accumulation topology:
    # accumulate each expert into a zero delta buffer, then add base exactly once.
    combined = torch.zeros_like(base)
    if base.ndim:
        rows = torch.arange(base.shape[0], device=base.device)
        for delta in deltas:
            combined.index_add_(0, rows, delta.to(dtype=base.dtype))
    else:
        for delta in deltas:
            combined.add_(delta.to(dtype=base.dtype))
    return base + combined, tuple(1.0 for _ in deltas)
