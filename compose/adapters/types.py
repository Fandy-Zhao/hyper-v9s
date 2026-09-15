"""Unified per-sample expert selection (empty / single / pair).

Inference selections are ``[batch_size, 4]``: four expert slots are the
maximum active experts (``max_active_experts = 4``). Empty slots are marked
with the pad id ``-1`` and a zero gate:

  empty  -> expert_ids row [-1, -1], gates row [0.0, 0.0]   (backbone-only)
  single -> expert_ids row [x, -1],  gates row [w, 0.0]
  pair   -> expert_ids row [x, y],   gates row [w_x, w_y]

Cluster-wise conditional-residual training widens the selection to three
slots: the per-sample old-teacher set (up to a pair) plus the new cluster
expert. The composition rule generalizes the pair rule (1/sqrt(2)) to
1/sqrt(3) so activations keep unit variance (see ComposeLinear.forward).

All cardinalities share one forward path, one batch-grouping rule (expert
dedup via ``torch.unique``), one cache-key representation and one
snapshot-restore representation.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

PAD_EXPERT_ID = -1
MAX_ACTIVE_EXPERTS = 4
MAX_INFERENCE_EXPERTS = 2


@dataclass(frozen=True)
class ComposeSelection:
    """Sparse per-sample expert selection.

    Tensors have shape ``[batch_size, MAX_ACTIVE_EXPERTS]``. A row of
    ``PAD_EXPERT_ID`` slots is the empty (backbone-only) selection. Gate
    normalization is explicit and only applies to active slots (pads stay
    zero).
    """

    expert_ids: torch.LongTensor
    gates: torch.FloatTensor
    normalization: str = "none"
    #: Slot count of this selection.  V8/V7 always build ``MAX_ACTIVE_EXPERTS``
    #: (4) slots, which is what the default preserves byte for byte.  V9 widens
    #: it to ``historical_retrieval.top_c + num_candidates`` because every row of a V9
    #: micro-batch carries the *same number* of active experts (only the expert
    #: ids differ per row), so the dense row is still a valid sparse selection.
    max_slots: int = MAX_ACTIVE_EXPERTS
    #: Opt-in relaxation for a straight-through hard selection: a real expert
    #: slot may carry a gate of exactly zero, which the composition masks out
    #: (``active_mask = ids.ne(PAD) & gates.gt(0)``) so the forward result is
    #: the deployed Top-2 while the slot stays in the backward graph.  V8 never
    #: sets it, so V8's validation is unchanged.
    allow_zero_gates: bool = False

    def __post_init__(self) -> None:
        if self.expert_ids.ndim != 2 or self.gates.ndim != 2:
            raise ValueError("expert_ids and gates must be rank-2 tensors")
        if self.expert_ids.shape != self.gates.shape:
            raise ValueError("expert_ids and gates must have identical shapes")
        if self.max_slots < 1:
            raise ValueError("max_slots must be positive")
        if self.expert_ids.shape[1] != self.max_slots:
            raise ValueError(
                "the unified ComposeSelection uses exactly {} slots; got {}; "
                "use {} to mark an empty slot".format(
                    self.max_slots, self.expert_ids.shape[1], PAD_EXPERT_ID
                )
            )
        if self.expert_ids.dtype != torch.long:
            raise TypeError("expert_ids must use torch.long")
        if not self.gates.is_floating_point():
            raise TypeError("gates must be floating point")
        if not torch.isfinite(self.gates).all():
            raise ValueError("gates must be finite")
        if self.normalization not in ("none", "l1", "l2"):
            raise ValueError(
                "normalization must be one of none, l1, or l2; got {!r}".format(
                    self.normalization
                )
            )
        active = self.expert_ids != PAD_EXPERT_ID
        if torch.any(self.expert_ids[active] < 0):
            raise ValueError("expert ids must be non-negative or the {} pad".format(PAD_EXPERT_ID))
        if torch.any((~active) & (self.gates != 0)):
            raise ValueError("padded slots must have zero gates")
        if torch.any(self.gates < 0):
            raise ValueError("gates must be non-negative")
        if not self.allow_zero_gates and torch.any(active & (self.gates <= 0)):
            raise ValueError("active slots must have positive gates")
        for left in range(self.max_slots):
            for right in range(left + 1, self.max_slots):
                if torch.any(
                    (self.expert_ids[:, left] == self.expert_ids[:, right])
                    & (self.expert_ids[:, left] != PAD_EXPERT_ID)
                ):
                    raise ValueError("duplicate expert IDs per sample are not allowed")
        if self.normalization == "l1":
            denominator = self.gates.sum(dim=1, keepdim=True)
            denominator = denominator.clamp_min(torch.finfo(self.gates.dtype).tiny)
            object.__setattr__(self, "gates", self.gates / denominator)
        elif self.normalization == "l2":
            denominator = torch.linalg.vector_norm(
                self.gates, ord=2, dim=1, keepdim=True
            ).clamp_min(torch.finfo(self.gates.dtype).tiny)
            object.__setattr__(self, "gates", self.gates / denominator)

    @property
    def batch_size(self) -> int:
        return self.expert_ids.shape[0]

    @property
    def top_k(self) -> int:
        """Backward-compatible alias: the unified slot count (pads included)."""
        return self.max_slots

    def is_empty_row(self, index: int) -> bool:
        return bool(torch.all(self.expert_ids[index] == PAD_EXPERT_ID))

    def per_sample_sets(self) -> List[Dict[str, object]]:
        """Record the actual per-sample set and weights (pads removed).

        Used for logs, cache keys and snapshot manifests: every sample's real
        selection and weights are recoverable from the returned list.
        """
        records: List[Dict[str, object]] = []
        for index in range(self.batch_size):
            ids = []
            gates = []
            for slot_id, slot_gate in zip(self.expert_ids[index], self.gates[index]):
                if int(slot_id) != PAD_EXPERT_ID:
                    ids.append(int(slot_id))
                    gates.append(float(slot_gate))
            records.append({"expert_ids": ids, "gates": gates})
        return records

    def canonical_sets(self) -> List[Tuple[Tuple[int, ...], Tuple[float, ...]]]:
        """Hashable per-sample representation for cache keys."""
        return [
            (tuple(record["expert_ids"]), tuple(record["gates"]))
            for record in self.per_sample_sets()
        ]

    def to(self, device: torch.device) -> "ComposeSelection":
        if self.expert_ids.device == device and self.gates.device == device:
            return self
        return ComposeSelection(
            expert_ids=self.expert_ids.to(device=device),
            gates=self.gates.to(device=device),
            normalization=self.normalization,
            max_slots=self.max_slots,
            allow_zero_gates=self.allow_zero_gates,
        )


def pad_selection(
    expert_ids: Tuple[int, ...],
    gates: Tuple[float, ...],
    slots: int = MAX_ACTIVE_EXPERTS,
) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    """Pad a (possibly empty) selection to exactly ``slots`` slots."""
    if len(expert_ids) > slots:
        raise ValueError(
            "at most {} experts per sample; got {}".format(slots, expert_ids)
        )
    if len(expert_ids) != len(gates):
        raise ValueError("gates must match expert_ids")
    padded_ids = tuple(expert_ids) + (PAD_EXPERT_ID,) * (slots - len(expert_ids))
    padded_gates = tuple(gates) + (0.0,) * (slots - len(gates))
    return padded_ids, padded_gates
