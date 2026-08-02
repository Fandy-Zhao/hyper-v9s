"""Unified 0/1/2-expert LoRA composition API."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

from .direct_sum import direct_sum
from .rms_composition import RMSCompositionConfig, rms_compose


MODES = ("base_only", "single", "direct_sum", "rms_calibrated")


def _rms(tensor: torch.Tensor) -> float:
    value = tensor.detach().to(torch.float32)
    return float(torch.sqrt(torch.mean(value * value)).item()) if value.numel() else 0.0


@dataclass
class CompositionResult:
    output: torch.Tensor
    active_expert_ids: Tuple[int, ...]
    coefficients: Dict[int, float]
    delta_rms: Dict[int, float]
    pair_output_rms: float
    delta_cosine: Optional[float]
    diagnostics: Dict[str, Any]


class ExpertComposer:
    def __init__(self, bridge, statistics=None, rms_config: Optional[RMSCompositionConfig] = None) -> None:
        self.bridge = bridge
        self.statistics = statistics
        self.rms_config = rms_config or RMSCompositionConfig()

    def validate(self, active_expert_ids: Iterable[int], mode: str) -> Tuple[int, ...]:
        values = tuple(int(value) for value in active_expert_ids)
        if len(values) != len(set(values)):
            raise ValueError("duplicate active expert IDs are forbidden")
        if len(values) > 2:
            raise ValueError("at most two active experts are supported")
        if mode not in MODES:
            raise ValueError("unknown composition mode: {}".format(mode))
        expected = {"base_only": 0, "single": 1, "direct_sum": 2, "rms_calibrated": 2}[mode]
        if len(values) != expected:
            raise ValueError("mode {} requires {} active experts".format(mode, expected))
        missing = sorted(set(values) - set(self.bridge.expert_ids))
        if missing:
            raise KeyError("unregistered experts: {}".format(missing))
        return tuple(sorted(values))

    def forward(self, module, hidden_states: torch.Tensor, active_expert_ids, mode: str, runtime_context: Dict[str, Any]) -> CompositionResult:
        ids = self.validate(active_expert_ids, mode)
        base = runtime_context["base_output"]
        layer_name = runtime_context["layer_name"]
        deltas = tuple(self.bridge.compute_expert_delta(module, expert_id, hidden_states).to(base.dtype) for expert_id in ids)
        diagnostics = {"mode": mode, "layer_name": layer_name}
        if mode == "base_only":
            output, coefficients = base, ()
        elif mode == "single":
            output, coefficients = direct_sum(base, deltas)
        elif mode == "direct_sum":
            output, coefficients = direct_sum(base, deltas)
        else:
            if self.statistics is None:
                raise RuntimeError("rms_calibrated requires frozen calibration statistics")
            calibrated = tuple(self.statistics.delta_rms(expert_id, layer_name) for expert_id in ids)
            output, coefficients, rms_audit = rms_compose(base, deltas, calibrated, self.rms_config)
            diagnostics.update(rms_audit)
        cosine = None
        if len(deltas) == 2:
            left, right = (item.detach().to(torch.float32).reshape(-1) for item in deltas)
            denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
            cosine = float(torch.dot(left, right).div(denominator).item()) if denominator.item() > 0 else 0.0
        return CompositionResult(
            output=output,
            active_expert_ids=ids,
            coefficients={expert_id: float(value) for expert_id, value in zip(ids, coefficients)},
            delta_rms={expert_id: _rms(delta) for expert_id, delta in zip(ids, deltas)},
            pair_output_rms=_rms(output - base),
            delta_cosine=cosine,
            diagnostics=diagnostics,
        )
