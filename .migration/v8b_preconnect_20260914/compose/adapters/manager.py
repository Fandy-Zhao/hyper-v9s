from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from .lora import ComposeLinear
from .runtime import use_selection
from .types import MAX_ACTIVE_EXPERTS, PAD_EXPERT_ID, ComposeSelection, pad_selection


class ExpertManager:
    """Coordinates expert lifecycle and fixed selections across injected layers."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.layers = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, ComposeLinear)
        }  # type: Dict[str, ComposeLinear]
        if not self.layers:
            raise ValueError("model has no ComposeLinear layers")

    def add_expert(self, expert_id: int) -> None:
        for layer in self.layers.values():
            layer.add_expert(expert_id)

    def expert_ids(self) -> List[int]:
        common = None
        for layer in self.layers.values():
            layer_ids = set(layer.expert_ids())
            common = layer_ids if common is None else common.intersection(layer_ids)
        return sorted(common or [])

    def set_default_selection(
        self,
        expert_ids: Sequence[int],
        gates: Optional[Sequence[float]] = None,
        normalization: str = "none",
    ) -> None:
        self._require_experts(expert_ids)
        for layer in self.layers.values():
            layer.set_default_selection(expert_ids, gates, normalization)

    def clear_default_selection(self) -> None:
        for layer in self.layers.values():
            layer.clear_default_selection()

    def make_selection(
        self,
        expert_ids: Sequence[int],
        batch_size: int,
        gates: Optional[Sequence[float]] = None,
        device: Optional[torch.device] = None,
        normalization: str = "none",
    ) -> ComposeSelection:
        """Build a fixed selection expanded over ``batch_size`` rows.

        Empty (``expert_ids=[]``) selects backbone only; one and two experts
        use the same padded ``[batch, 2]`` slot structure.
        """
        self._require_experts(expert_ids)
        if len(expert_ids) > MAX_ACTIVE_EXPERTS:
            raise ValueError(
                "fixed selection supports zero through {} experts".format(
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
        ids_tensor = torch.tensor(padded_ids, dtype=torch.long, device=device)
        gate_tensor = torch.tensor(padded_gates, dtype=torch.float32, device=device)
        return ComposeSelection(
            ids_tensor.unsqueeze(0).expand(batch_size, -1),
            gate_tensor.unsqueeze(0).expand(batch_size, -1),
            normalization=normalization,
        )

    def selection_context(self, selection: ComposeSelection):
        self._require_experts(
            [int(value) for value in torch.unique(selection.expert_ids) if int(value) != PAD_EXPERT_ID]
        )
        return use_selection(selection)

    def freeze_base(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for layer in self.layers.values():
            for expert in layer.experts.values():
                expert.requires_grad_(True)

    def train_only(self, expert_ids: Iterable[int]) -> None:
        selected = set(int(value) for value in expert_ids)
        self._require_experts(selected)
        self.freeze_base()
        for layer in self.layers.values():
            for key, expert in layer.experts.items():
                is_selected = int(key) in selected
                expert.requires_grad_(is_selected)
                if not is_selected:
                    for parameter in expert.parameters():
                        parameter.grad = None

    def _require_experts(self, expert_ids: Iterable[int]) -> None:
        available = set(self.expert_ids())
        missing = sorted(
            set(int(value) for value in expert_ids if int(value) != PAD_EXPERT_ID)
            - available
        )
        if missing:
            raise KeyError("unregistered experts: {}".format(missing))
