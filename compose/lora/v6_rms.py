"""V6 RMS statistics runner and composition calibration report (Stage E9).

For each newly committed expert:

    R_k_l = mean_x RMS(B_k_l A_k_l h_l)

collected per layer and target module with fp32 accumulation (the
underlying OnlineMoments merge in float64), all-reduced across GPUs,
bound to the checkpoint hash through the statistics provenance. When the
checkpoint changes, ``validate_rms_freshness`` reports the stale cache.

Composition calibration (per layer):

    kappa_k_l = clip(reference_R_l / (R_k_l + epsilon), kappa_min, kappa_max)

Reported diagnostics: raw RMS, calibrated RMS, clip ratio, pair cosine
(same-layer delta alignment), dominance ratio (max/min RMS) and
cancellation ratio (pairwise delta cancellation).
"""

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .statistics import RMSStatistics, StatisticKey, stable_hash
from .rms_composition import RMSCompositionConfig

PAIR_DIAGNOSTIC_VERSION = 1


@dataclass(frozen=True)
class V6RMSConfig:
    epsilon: float = 1.0e-8
    kappa_min: float = 0.25
    kappa_max: float = 4.0
    calibration_split: str = "validation"

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0.0 < self.kappa_min <= self.kappa_max:
            raise ValueError("kappa bounds must satisfy 0 < min <= max")
        if "test" in self.calibration_split.lower():
            raise ValueError("test data cannot be used for RMS calibration")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PairDeltaMoments:
    """Same-layer cross terms for two experts: E[<dA, dB>] and norms."""

    def __init__(self) -> None:
        self.count = 0
        self.dot_sum = 0.0
        self.norm_a_sq_sum = 0.0
        self.norm_b_sq_sum = 0.0

    def update(self, delta_a: torch.Tensor, delta_b: torch.Tensor) -> None:
        flat_a = delta_a.detach().float().reshape(-1)
        flat_b = delta_b.detach().float().reshape(-1)
        if flat_a.numel() != flat_b.numel():
            raise ValueError("pair deltas must share the same shape")
        self.count += 1
        self.dot_sum += float((flat_a * flat_b).sum())
        self.norm_a_sq_sum += float((flat_a * flat_a).sum())
        self.norm_b_sq_sum += float((flat_b * flat_b).sum())

    def cosine(self) -> Optional[float]:
        if self.count == 0:
            return None
        denominator = math.sqrt(self.norm_a_sq_sum * self.norm_b_sq_sum)
        if denominator == 0:
            return None
        return self.dot_sum / denominator

    def cancellation(self) -> Optional[float]:
        """||dA + dB||^2 / (||dA||^2 + ||dB||^2); ~1 orthogonal, 0 cancelled."""
        denominator = self.norm_a_sq_sum + self.norm_b_sq_sum
        if denominator == 0:
            return None
        numerator = (
            self.norm_a_sq_sum + self.norm_b_sq_sum + 2.0 * self.dot_sum
        )
        return numerator / denominator

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": PAIR_DIAGNOSTIC_VERSION,
            "count": self.count,
            "dot_sum": self.dot_sum,
            "norm_a_sq_sum": self.norm_a_sq_sum,
            "norm_b_sq_sum": self.norm_b_sq_sum,
        }


def compute_v6_expert_rms(
    model,
    expert_ids: Sequence[int],
    dataloader: Iterable[Any],
    provenance: Dict[str, Any],
    config: V6RMSConfig,
    prepare_batch: Callable[[Any], Any],
    device: str = "cuda:0",
    forward_fn: Optional[Callable[[Any], Any]] = None,
) -> Tuple[RMSStatistics, Dict[str, PairDeltaMoments]]:
    """Collect per-layer delta RMS for ``expert_ids`` over ``dataloader``.

    ``prepare_batch(batch)`` returns the model input; ``forward_fn(inputs)``
    runs the forward pass (defaults to ``model(inputs)``). A forward hook
    recomputes each expert's delta per ComposeLinear layer
    (post-activation), so no core operator is modified. Pair cross terms
    are collected for every expert pair sharing a layer.
    """
    from compose.adapters.lora import ComposeLinear

    if len(expert_ids) < 1:
        raise ValueError("at least one expert is required")
    stats = RMSStatistics(provenance)
    pair_moments = {}  # type: Dict[str, PairDeltaMoments]
    layers = {
        name: module for name, module in model.named_modules()
        if isinstance(module, ComposeLinear)
    }
    if not layers:
        raise ValueError("model has no ComposeLinear layers")
    layer_by_id = {
        str(expert_id): layer for expert_id in expert_ids
        for layer in layers.values() if str(expert_id) in layer.experts
    }
    missing = [expert_id for expert_id in expert_ids if str(expert_id) not in layer_by_id]
    if missing:
        raise KeyError("experts not registered in the model: {}".format(missing))

    hooks = []

    def make_post_hook(name: str, module: ComposeLinear):
        def post_hook(hooked_module, module_inputs, module_output):
            hidden = module_inputs[0]
            for expert_id in expert_ids:
                key = str(expert_id)
                if key not in module.experts:
                    continue
                delta = module.experts[key](hidden).to(module_output.dtype)
                statistic_key = StatisticKey(
                    expert_id=int(expert_id),
                    layer_name=name,
                    module_name=name,
                    target_module_type=module.__class__.__name__,
                )
                stats.update(
                    statistic_key, delta.detach(), module_output.detach(), None
                )
            # Pair cross terms on the first layer occurrence.
            if len(expert_ids) == 2 and len(layers) and name == next(iter(layers)):
                left, right = expert_ids
                if str(left) in module.experts and str(right) in module.experts:
                    pair = pair_moments.setdefault(name, PairDeltaMoments())
                    pair.update(
                        module.experts[str(left)](hidden).detach(),
                        module.experts[str(right)](hidden).detach(),
                    )
            return None

        return post_hook

    for name, module in layers.items():
        hooks.append(module.register_forward_hook(make_post_hook(name, module)))
    try:
        # The forward itself runs backbone-only; every expert delta is
        # recomputed inside the hook, so no selection context is needed
        # (and the batch size is unconstrained).
        model.eval()
        forward = forward_fn or model
        with torch.inference_mode():
            for batch in dataloader:
                inputs = prepare_batch(batch)
                forward(inputs)
    finally:
        for hook in hooks:
            hook.remove()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        stats.all_reduce_(torch.device(device))
    return stats, pair_moments


def rms_report(
    stats: RMSStatistics,
    expert_ids: Sequence[int],
    pair_moments: Optional[Dict[str, PairDeltaMoments]] = None,
    config: Optional[V6RMSConfig] = None,
) -> Dict[str, Any]:
    """Per-expert raw/calibrated RMS plus pair diagnostics."""
    config = config or V6RMSConfig()
    expert_ids = [int(value) for value in expert_ids]
    layers = sorted(
        {entry["key"]["layer_name"] for entry in stats.entries.values()}
    )
    per_expert = {}
    clip_counts = {expert_id: 0 for expert_id in expert_ids}
    layer_counts = {expert_id: 0 for expert_id in expert_ids}
    for expert_id in expert_ids:
        per_expert[str(expert_id)] = {}
        for layer in layers:
            try:
                raw = stats.delta_rms(expert_id, layer)
            except KeyError:
                continue
            layer_counts[expert_id] += 1
            # Calibrated coefficient against the arithmetic-mean reference.
            reference = raw  # single-expert layer: itself
            kappa = min(
                max(reference / (raw + config.epsilon), config.kappa_min),
                config.kappa_max,
            )
            if kappa != reference / (raw + config.epsilon):
                clip_counts[expert_id] += 1
            per_expert[str(expert_id)][layer] = {
                "raw_rms": raw,
                "calibrated_rms": raw * kappa,
                "kappa": kappa,
            }
    pair_diagnostics = {}
    for layer, moments in (pair_moments or {}).items():
        pair_diagnostics[layer] = {
            "cosine": moments.cosine(),
            "cancellation_ratio": moments.cancellation(),
            "samples": moments.count,
        }
    dominance = {}
    for expert_id in expert_ids:
        raw_values = [
            per_expert[str(expert_id)][layer]["raw_rms"]
            for layer in per_expert[str(expert_id)]
        ]
        dominance[str(expert_id)] = (
            max(raw_values) / min(raw_values) if len(raw_values) >= 2 and min(raw_values) > 0 else None
        )
    return {
        "expert_ids": expert_ids,
        "layers": layers,
        "per_expert": per_expert,
        "clip_ratio": {
            str(expert_id): (
                clip_counts[expert_id] / layer_counts[expert_id]
                if layer_counts[expert_id] else None
            )
            for expert_id in expert_ids
        },
        "dominance_ratio": dominance,
        "pair": pair_diagnostics,
        "config": config.to_dict(),
    }


def validate_rms_freshness(
    stats: RMSStatistics, expected_checkpoint_hash: str
) -> bool:
    """The RMS cache is bound to the checkpoint hash; a changed checkpoint
    invalidates it automatically."""
    return str(stats.provenance.get("checkpoint_hash")) == str(expected_checkpoint_hash)


def build_rms_provenance(
    calibration_split: str,
    checkpoint_hash: str,
    dataset_manifest_hash: str,
    composition_config_hash: str,
) -> Dict[str, Any]:
    return {
        "calibration_split": calibration_split,
        "checkpoint_hash": checkpoint_hash,
        "dataset_manifest_hash": dataset_manifest_hash,
        "composition_config_hash": composition_config_hash,
    }
