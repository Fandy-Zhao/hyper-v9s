import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .runtime import get_current_selection
from .types import MAX_ACTIVE_EXPERTS, PAD_EXPERT_ID, ComposeSelection, pad_selection


#: ``ComposeLinear`` selection-plan sharing (V8-Exact-Accelerated, S5).
#:
#: Enabled, the per-selection decomposition (``torch.unique``, active masks,
#: cardinality scale) is computed once per micro-step instead of once per layer,
#: which removes ~3 device synchronisations per layer call.  The arithmetic --
#: expert order, gate sums, kappa, ``index_add_`` order -- is unchanged, so the
#: forward result is bit-identical; ``tests/compose`` covers both paths.
#:
#: Disabled (the default), the original per-layer code path runs verbatim and
#: the repository reproduces the frozen baseline byte for byte.
_FAST_SELECTION = os.environ.get("COMPOSE_SELECTION_PLAN", "0") == "1"


def set_fast_selection(enabled: bool) -> None:
    """Enable or disable shared selection plans for this process."""
    global _FAST_SELECTION
    _FAST_SELECTION = bool(enabled)


def fast_selection_enabled() -> bool:
    return _FAST_SELECTION


class _SelectionPlan:
    """Layer-independent decomposition of one ``ComposeSelection``.

    Every ``ComposeLinear`` in the model sees the *same* selection within a
    micro-step, so ``torch.unique``, the per-expert active masks and the
    cardinality scale are identical for all 224 layers.  Recomputing them per
    layer costs one device synchronisation per distinct expert id per layer
    (``int(expert_id_tensor.item())``), which is what used to dominate the
    step.  They are computed once here and shared.

    ``entries`` preserves the original evaluation order (ascending expert id,
    as returned by ``torch.unique``), so ``delta.index_add_`` accumulates in
    exactly the previous order and the result is bit-identical.
    """

    __slots__ = ("expert_ids", "gates", "key", "entries")

    def __init__(
        self,
        expert_ids: torch.Tensor,
        gates: torch.Tensor,
        key: Tuple,
        entries: List[Tuple[int, torch.Tensor, torch.Tensor]],
    ) -> None:
        self.expert_ids = expert_ids
        self.gates = gates
        self.key = key
        self.entries = entries

    def matches(self, selection: ComposeSelection, key: Tuple) -> bool:
        return (
            self.key == key
            and self.expert_ids is selection.expert_ids
            and self.gates is selection.gates
        )


def _build_selection_plan(
    selection: ComposeSelection, key: Tuple
) -> _SelectionPlan:
    expert_ids = selection.expert_ids
    gates = selection.gates
    device = expert_ids.device
    active_mask = expert_ids.ne(PAD_EXPERT_ID) & gates.gt(0)
    per_sample_active_count = active_mask.sum(dim=1)  # [batch]
    active_counts = per_sample_active_count.to(key[1])
    cardinality_scale = torch.rsqrt(active_counts.clamp_min(1.0))
    per_sample_scale = torch.where(
        active_counts.eq(2),
        torch.full_like(active_counts, float(key[3])),
        cardinality_scale,
    )
    entries: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
    for expert_id_tensor in torch.unique(expert_ids):
        expert_id = int(expert_id_tensor.item())
        if expert_id == PAD_EXPERT_ID:
            continue
        active_slots = expert_ids.eq(expert_id_tensor) & gates.gt(0)
        sample_indices = torch.where(active_slots.any(dim=1))[0]
        if sample_indices.numel() == 0:
            continue
        sample_gates = (gates * active_slots.to(gates.dtype)).sum(dim=1).index_select(
            0, sample_indices
        )
        entries.append(
            (expert_id, sample_indices, sample_gates, per_sample_scale.index_select(0, sample_indices))
        )
    return _SelectionPlan(expert_ids, gates, key, entries)


def selection_plan(
    selection: ComposeSelection, device: torch.device, dtype: torch.dtype, pair_scale: float
) -> _SelectionPlan:
    """Return the cached plan for this selection, building it at most once."""
    key = (device, dtype, selection.batch_size, float(pair_scale))
    cache = getattr(selection, "_compose_plan_cache", None)
    if cache is None:
        cache = {}
        object.__setattr__(selection, "_compose_plan_cache", cache)
    plan = cache.get(key)
    if plan is not None and plan.matches(selection, key):
        return plan
    plan = _build_selection_plan(selection, key)
    cache[key] = plan
    return plan


class LoRAExpert(nn.Module):
    """One self-contained LoRA expert."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        expert_dtype = self.lora_A.weight.dtype
        delta = self.lora_B(self.lora_A(self.dropout(inputs.to(expert_dtype))))
        return delta * self.scaling


DEFAULT_PAIR_SCALE = 1.0 / (2.0 ** 0.5)


class ComposeLinear(nn.Module):
    """A frozen base linear layer plus independently stored LoRA experts."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer must be torch.nn.Linear")
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.experts = nn.ModuleDict()
        self._default_expert_ids = None  # type: Optional[Sequence[int]]
        self._default_gates = None  # type: Optional[Sequence[float]]
        self._default_normalization = "none"
        # RMS calibration: expert_id -> kappa_k_l (per-layer coefficient).
        # Absent ids behave as kappa = 1.0. Pair selections additionally
        # scale every expert contribution by ``pair_scale`` (1/sqrt(2)).
        self._expert_calibration = {}  # type: Dict[int, float]
        self._pair_scale = float(DEFAULT_PAIR_SCALE)

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    @property
    def weight(self) -> nn.Parameter:
        return self.base_layer.weight

    @property
    def bias(self) -> Optional[nn.Parameter]:
        return self.base_layer.bias

    def add_expert(self, expert_id: int) -> LoRAExpert:
        key = str(int(expert_id))
        if int(expert_id) < 0:
            raise ValueError("expert_id must be non-negative")
        if key in self.experts:
            raise ValueError("expert {} already exists".format(expert_id))
        expert = LoRAExpert(
            self.in_features,
            self.out_features,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
        )
        expert.to(device=self.weight.device, dtype=self.weight.dtype)
        self.experts[key] = expert
        return expert

    def set_default_selection(
        self,
        expert_ids: Sequence[int],
        gates: Optional[Sequence[float]] = None,
        normalization: str = "none",
    ) -> None:
        if len(expert_ids) > MAX_ACTIVE_EXPERTS:
            raise ValueError(
                "default selection supports zero through {} experts".format(
                    MAX_ACTIVE_EXPERTS
                )
            )
        if gates is None:
            gates = [1.0] * len(expert_ids)
        if len(gates) != len(expert_ids):
            raise ValueError("gates must match expert_ids")
        padded_ids, padded_gates = pad_selection(
            tuple(int(value) for value in expert_ids),
            tuple(float(value) for value in gates),
        )
        self._default_expert_ids = padded_ids
        self._default_gates = padded_gates
        self._default_normalization = normalization

    def clear_default_selection(self) -> None:
        self._default_expert_ids = None
        self._default_gates = None
        self._default_normalization = "none"

    # ------------------------------------------------------------------
    # RMS calibration (runtime kappa)
    # ------------------------------------------------------------------

    def set_expert_calibration(
        self, kappa_map: Dict[int, float], pair_scale: float = DEFAULT_PAIR_SCALE
    ) -> None:
        """Apply per-expert RMS kappa coefficients for this layer.

        ``kappa_map`` maps expert_id -> kappa_k_l; experts not listed keep
        kappa 1.0. ``pair_scale`` is the 1/sqrt(2) pair composition factor
        applied to every expert when a sample selects a pair. Both are
        used inside :meth:`forward`, not just reported.
        """
        if pair_scale <= 0:
            raise ValueError("pair_scale must be positive")
        self._expert_calibration = {
            int(expert_id): float(kappa)
            for expert_id, kappa in kappa_map.items()
            if float(kappa) > 0
        }
        self._pair_scale = float(pair_scale)

    def clear_expert_calibration(self) -> None:
        """Drop all kappa coefficients and reset the pair scale."""
        self._expert_calibration = {}
        self._pair_scale = float(DEFAULT_PAIR_SCALE)

    def expert_calibration(self) -> Dict[str, object]:
        """Serializable per-expert kappa for this layer (persistence)."""
        return {
            "kappa": dict(self._expert_calibration),
            "pair_scale": self._pair_scale,
        }

    def _selection_for(self, inputs: torch.Tensor) -> Optional[ComposeSelection]:
        selection = get_current_selection()
        if selection is not None:
            return selection.to(inputs.device)
        if self._default_expert_ids is None:
            return None
        batch_size = inputs.shape[0]
        ids = torch.tensor(
            self._default_expert_ids, device=inputs.device, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1)
        gates = torch.tensor(
            self._default_gates, device=inputs.device, dtype=inputs.dtype
        ).unsqueeze(0).expand(batch_size, -1)
        return ComposeSelection(ids, gates, normalization=self._default_normalization)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(inputs)
        selection = self._selection_for(inputs)
        if selection is None:
            return result
        if inputs.ndim < 2:
            raise ValueError("ComposeLinear expects a batch dimension")
        if selection.batch_size != inputs.shape[0]:
            raise ValueError(
                "selection batch size {} does not match input batch size {}".format(
                    selection.batch_size, inputs.shape[0]
                )
            )

        if _FAST_SELECTION:
            # Everything that depends only on the selection -- the cardinality
            # scale, the active sample indices and the summed gates -- is shared
            # across layers by ``selection_plan``.  Only ``kappa`` and
            # ``pair_scale`` are per layer, and ``pair_scale`` is part of the
            # plan key, so the arithmetic below is unchanged.
            plan = selection_plan(
                selection, result.device, result.dtype, self._pair_scale
            )
            delta = torch.zeros_like(result)
            for expert_id, sample_indices, sample_gates, scale in plan.entries:
                key = str(expert_id)
                if key not in self.experts:
                    raise KeyError("expert {} is not registered".format(expert_id))
                selected_inputs = inputs.index_select(0, sample_indices)
                expert_delta = self.experts[key](selected_inputs).to(result.dtype)
                kappa = self._expert_calibration.get(expert_id, 1.0)
                effective = sample_gates * kappa * scale
                gate_shape = [sample_indices.shape[0]] + [1] * (result.ndim - 1)
                weighted_delta = expert_delta * effective.to(result.dtype).reshape(
                    gate_shape
                )
                delta.index_add_(0, sample_indices, weighted_delta)
            return result + delta

        return self._forward_per_layer_selection(inputs, result, selection)

    def _forward_per_layer_selection(
        self, inputs: torch.Tensor, result: torch.Tensor, selection: ComposeSelection
    ) -> torch.Tensor:
        """Frozen baseline path: selection decomposition recomputed per layer.

        Kept verbatim so ``COMPOSE_SELECTION_PLAN=0`` reproduces the pre-V8
        execution exactly; ``tests/compose`` asserts the two paths agree.
        """
        # RMS calibration: per-sample per-expert effective scale =
        # gate * kappa_k_l, times the composition rule for the sample's
        # selection cardinality. This is the formal composition rule
        # shared by teacher scoring, cluster training and test inference:
        #   single -> 1.0
        #   pair   -> pair_scale (default 1/sqrt(2))
        #   N experts -> 1/sqrt(N)
        # 1/sqrt(count) keeps activation variance constant.
        active_mask = selection.expert_ids.ne(PAD_EXPERT_ID) & selection.gates.gt(0)
        per_sample_active_count = active_mask.sum(dim=1)  # [batch]
        active_counts = per_sample_active_count.to(result.dtype)
        cardinality_scale = torch.rsqrt(active_counts.clamp_min(1.0))
        per_sample_scale = torch.where(
            active_counts.eq(2),
            torch.full_like(active_counts, self._pair_scale),
            cardinality_scale,
        )

        delta = torch.zeros_like(result)
        for expert_id_tensor in torch.unique(selection.expert_ids):
            expert_id = int(expert_id_tensor.item())
            if expert_id == PAD_EXPERT_ID:
                # Empty slot (empty selection); contributes nothing.
                continue
            key = str(expert_id)
            if key not in self.experts:
                raise KeyError("expert {} is not registered".format(expert_id))
            active_slots = selection.expert_ids.eq(expert_id_tensor) & selection.gates.gt(0)
            sample_indices = torch.where(active_slots.any(dim=1))[0]
            if sample_indices.numel() == 0:
                continue
            selected_inputs = inputs.index_select(0, sample_indices)
            expert_delta = self.experts[key](selected_inputs).to(result.dtype)
            sample_gates = (
                selection.gates * active_slots.to(selection.gates.dtype)
            ).sum(dim=1).index_select(0, sample_indices)
            kappa = self._expert_calibration.get(expert_id, 1.0)
            scale = per_sample_scale.index_select(0, sample_indices)
            effective = sample_gates * kappa * scale
            gate_shape = [sample_indices.shape[0]] + [1] * (result.ndim - 1)
            weighted_delta = expert_delta * effective.to(result.dtype).reshape(gate_shape)
            delta.index_add_(0, sample_indices, weighted_delta)
        return result + delta

    def expert_ids(self) -> Iterable[int]:
        return (int(key) for key in self.experts.keys())

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, experts={}".format(
            self.in_features, self.out_features, sorted(self.expert_ids())
        )
