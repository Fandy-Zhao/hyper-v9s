"""Frozen arithmetic-mean RMS calibration for two LoRA experts."""

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch


@dataclass(frozen=True)
class RMSCompositionConfig:
    pair_scale: float = 1.0 / math.sqrt(2.0)
    epsilon: float = 1e-8
    coefficient_min: float = 0.25
    coefficient_max: float = 4.0
    expert_scalars: Tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        if self.pair_scale <= 0 or self.epsilon <= 0:
            raise ValueError("pair_scale and epsilon must be positive")
        if not 0 < self.coefficient_min <= self.coefficient_max:
            raise ValueError("invalid coefficient clip interval")
        if len(self.expert_scalars) != 2 or any(
            not 0.0 <= float(value) <= 2.0 for value in self.expert_scalars
        ):
            raise ValueError("expert_scalars must contain two values in [0, 2]")


def frozen_coefficients(rms_i: float, rms_j: float, config: RMSCompositionConfig) -> Tuple[Tuple[float, float], Dict[str, object]]:
    """Equalize each expert to the same-layer arithmetic mean RMS.

    This is the formal Compose rule: ``kappa_k = mean_R / (R_k + eps)``.
    The earlier geometric-pair rule is intentionally not used here.
    """
    if rms_i < 0 or rms_j < 0:
        raise ValueError("RMS values must be non-negative")
    reference = 0.5 * (rms_i + rms_j)
    raw = (reference / (rms_i + config.epsilon), reference / (rms_j + config.epsilon))
    clipped = tuple(min(max(value, config.coefficient_min), config.coefficient_max) for value in raw)
    guarded = rms_i <= config.epsilon or rms_j <= config.epsilon
    return (float(clipped[0]), float(clipped[1])), {
        "epsilon_guarded": guarded,
        "epsilon_fallback": guarded,  # historical diagnostics compatibility
        "reference_rms": reference,
        "raw_coefficients": list(raw),
        "clipped": clipped != raw,
    }


def rms_compose(base: torch.Tensor, deltas: Sequence[torch.Tensor], calibrated_rms: Sequence[float], config: RMSCompositionConfig):
    if len(deltas) != 2 or len(calibrated_rms) != 2:
        raise ValueError("rms_calibrated requires exactly two experts")
    coefficients, audit = frozen_coefficients(float(calibrated_rms[0]), float(calibrated_rms[1]), config)
    effective = tuple(
        coefficient * float(scalar)
        for coefficient, scalar in zip(coefficients, config.expert_scalars)
    )
    output = base
    for coefficient, delta in zip(effective, deltas):
        output = output + delta.to(base.dtype) * (config.pair_scale * coefficient)
    if not torch.isfinite(output).all():
        raise FloatingPointError("rms_calibrated produced NaN or Inf")
    audit.update({
        "pair_scale": config.pair_scale,
        "calibrated_rms": list(map(float, calibrated_rms)),
        "expert_scalars": list(map(float, config.expert_scalars)),
        "rms_coefficients": list(coefficients),
        "effective_coefficients": list(effective),
    })
    return output, effective, audit
