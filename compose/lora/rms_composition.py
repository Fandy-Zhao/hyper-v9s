"""Frozen symmetric RMS calibration for two LoRA experts."""

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

    def __post_init__(self) -> None:
        if self.pair_scale <= 0 or self.epsilon <= 0:
            raise ValueError("pair_scale and epsilon must be positive")
        if not 0 < self.coefficient_min <= self.coefficient_max:
            raise ValueError("invalid coefficient clip interval")


def frozen_coefficients(rms_i: float, rms_j: float, config: RMSCompositionConfig) -> Tuple[Tuple[float, float], Dict[str, object]]:
    """Equalize calibrated pair RMS to their geometric mean.

    c_i=sqrt(rms_j/rms_i), c_j=sqrt(rms_i/rms_j). If either source RMS is
    at/below epsilon, fixed unit coefficients are used and audited.
    """
    fallback = rms_i <= config.epsilon or rms_j <= config.epsilon
    if fallback:
        raw = (1.0, 1.0)
    else:
        raw = (math.sqrt(rms_j / rms_i), math.sqrt(rms_i / rms_j))
    clipped = tuple(min(max(value, config.coefficient_min), config.coefficient_max) for value in raw)
    return (float(clipped[0]), float(clipped[1])), {"epsilon_fallback": fallback, "raw_coefficients": list(raw), "clipped": clipped != raw}


def rms_compose(base: torch.Tensor, deltas: Sequence[torch.Tensor], calibrated_rms: Sequence[float], config: RMSCompositionConfig):
    if len(deltas) != 2 or len(calibrated_rms) != 2:
        raise ValueError("rms_calibrated requires exactly two experts")
    coefficients, audit = frozen_coefficients(float(calibrated_rms[0]), float(calibrated_rms[1]), config)
    output = base
    for coefficient, delta in zip(coefficients, deltas):
        output = output + delta.to(base.dtype) * (config.pair_scale * coefficient)
    if not torch.isfinite(output).all():
        raise FloatingPointError("rms_calibrated produced NaN or Inf")
    audit.update({"pair_scale": config.pair_scale, "calibrated_rms": list(map(float, calibrated_rms))})
    return output, coefficients, audit
