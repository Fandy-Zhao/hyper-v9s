from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn

from .runtime import get_current_selection
from .types import ComposeSelection


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
        self, expert_ids: Sequence[int], gates: Optional[Sequence[float]] = None
    ) -> None:
        if len(expert_ids) not in (1, 2):
            raise ValueError("default selection supports one or two experts")
        if gates is None:
            gates = [1.0 / len(expert_ids)] * len(expert_ids)
        if len(gates) != len(expert_ids):
            raise ValueError("gates must match expert_ids")
        self._default_expert_ids = tuple(int(value) for value in expert_ids)
        self._default_gates = tuple(float(value) for value in gates)

    def clear_default_selection(self) -> None:
        self._default_expert_ids = None
        self._default_gates = None

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
        return ComposeSelection(ids, gates)

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

        delta = torch.zeros_like(result)
        gate_shape = [selection.batch_size] + [1] * (result.ndim - 1)
        for expert_id_tensor in torch.unique(selection.expert_ids):
            expert_id = int(expert_id_tensor.item())
            key = str(expert_id)
            if key not in self.experts:
                raise KeyError("expert {} is not registered".format(expert_id))
            sample_gates = torch.where(
                selection.expert_ids == expert_id_tensor,
                selection.gates,
                torch.zeros_like(selection.gates),
            ).sum(dim=1)
            if not torch.any(sample_gates > 0):
                continue
            expert_delta = self.experts[key](inputs).to(result.dtype)
            delta = delta + expert_delta * sample_gates.to(result.dtype).reshape(gate_shape)
        return result + delta

    def expert_ids(self) -> Iterable[int]:
        return (int(key) for key in self.experts.keys())

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, experts={}".format(
            self.in_features, self.out_features, sorted(self.expert_ids())
        )
