from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn

from .runtime import get_current_selection
from .types import MAX_ACTIVE_EXPERTS, PAD_EXPERT_ID, ComposeSelection, pad_selection


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
