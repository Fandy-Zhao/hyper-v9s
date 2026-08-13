"""Compose RMS statistics and runtime kappa calibration.

For each newly committed expert:

    R_k_l = mean_x RMS(B_k_l A_k_l h_l)

collected per layer and target module with fp32 accumulation (the
underlying OnlineMoments merge in float64), all-reduced across GPUs,
bound to the checkpoint hash through the statistics provenance.

Composition calibration (per layer):

    kappa_k_l = clip(R_bar_l / (R_k_l + epsilon), kappa_min, kappa_max)
    R_bar_l   = mean over ALL active experts in layer l   # never the
                 expert's own RMS alone

The kappa map is persisted (``save_calibration``) and applied at runtime
through ``ComposeLinear.set_expert_calibration``: every single/pair
forward multiplies each expert's delta by its kappa. Calibration is not a
report-only diagnostic.

Reported diagnostics: raw RMS, calibrated RMS, clip ratio, pair cosine
(same-layer delta alignment), dominance ratio (max/min RMS) and
cancellation ratio (pairwise delta cancellation).
"""

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from .rms_composition import RMSCompositionConfig
from .statistics import RMSStatistics, StatisticKey, stable_hash

PAIR_DIAGNOSTIC_VERSION = 1
CALIBRATION_VERSION = 1


@dataclass(frozen=True)
class ComposeRMSConfig:
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


def compute_expert_rms(
    model,
    expert_ids: Sequence[int],
    dataloader: Iterable[Any],
    provenance: Dict[str, Any],
    config: ComposeRMSConfig,
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
        # The hook path materializes every expert delta.  Large expert pools
        # can leave most of the device in PyTorch's caching allocator even
        # after the final forward has released its live tensors.  NCCL still
        # needs a small device buffer for the exact fp64 scalar reductions;
        # release only unused cached blocks before allocating that buffer.
        # This does not alter samples, moments, reduction order, or kappa.
        target = torch.device(device)
        if target.type == "cuda":
            torch.cuda.synchronize(target)
            torch.cuda.empty_cache()
        stats.all_reduce_(target)
    return stats, pair_moments


def build_kappa_calibration(
    stats: RMSStatistics,
    expert_ids: Sequence[int],
    config: Optional[ComposeRMSConfig] = None,
) -> Dict[str, Dict[str, float]]:
    """Runtime kappa map ``{layer_name: {expert_id: kappa_k_l}}``.

    The reference is the arithmetic mean over ALL active experts in the
    layer (never an expert's own RMS alone):

        R_bar_l = mean_k R_k_l
        kappa_k_l = clip(R_bar_l / (R_k_l + epsilon), kappa_min, kappa_max)

    A layer with a single measured expert calibrates to kappa 1.0 (its
    own RMS is the mean). Layers without statistics for an expert are
    omitted, so those experts keep kappa 1.0 at runtime.
    """
    config = config or ComposeRMSConfig()
    expert_ids = [int(value) for value in expert_ids]
    layers = sorted(
        {entry["key"]["layer_name"] for entry in stats.entries.values()}
    )
    calibration = {}  # type: Dict[str, Dict[str, float]]
    for layer in layers:
        per_expert = {}
        raw_values = []
        for expert_id in expert_ids:
            try:
                raw = stats.delta_rms(expert_id, layer)
            except KeyError:
                continue
            per_expert[expert_id] = raw
            raw_values.append(raw)
        if not raw_values:
            continue
        reference = sum(raw_values) / len(raw_values)  # mean over active experts
        layer_map = {}
        for expert_id, raw in per_expert.items():
            kappa = min(
                max(reference / (raw + config.epsilon), config.kappa_min),
                config.kappa_max,
            )
            layer_map[str(expert_id)] = float(kappa)
        calibration[layer] = layer_map
    return calibration


def merge_commit_frozen_calibration(
    frozen: Mapping[str, Mapping[str, float]],
    current: Mapping[str, Mapping[str, float]],
    new_expert_ids: Sequence[int],
) -> Dict[str, Dict[str, float]]:
    """Preserve every committed kappa and add values only for new experts.

    ``current`` may have been measured over the expanded pool, but historical
    entries are deliberately ignored. This makes an expert's effective LoRA
    scale immutable after its commit boundary.
    """
    new_ids = {str(int(value)) for value in new_expert_ids}
    merged = {
        str(layer): {str(expert_id): float(value) for expert_id, value in values.items()}
        for layer, values in frozen.items()
    }
    for layer, values in current.items():
        target = merged.setdefault(str(layer), {})
        for expert_id, value in values.items():
            if str(expert_id) in new_ids:
                target[str(expert_id)] = float(value)
    return {layer: dict(sorted(values.items(), key=lambda item: int(item[0])))
            for layer, values in sorted(merged.items())}


def apply_kappa_calibration(model, calibration: Mapping[str, Mapping[str, float]]) -> None:
    """Apply the persisted kappa map to every ComposeLinear layer at runtime.

    Each layer gets ``set_expert_calibration({expert_id: kappa_k_l})`` so
    the coefficients participate in every single/pair forward, not just
    reports. Layers absent from the calibration keep kappa 1.0.
    """
    from compose.adapters.lora import ComposeLinear

    applied = 0
    for name, module in model.named_modules():
        if not isinstance(module, ComposeLinear):
            continue
        layer_map = calibration.get(name)
        if layer_map:
            module.set_expert_calibration(
                {int(expert_id): float(kappa) for expert_id, kappa in layer_map.items()}
            )
            applied += 1
    if applied != len(calibration):
        missing = sorted(set(calibration) - {name for name, _ in model.named_modules()})
        raise ValueError(
            "calibration references layers that are not ComposeLinear modules: {}".format(
                missing
            )
        )


def rms_report(
    stats: RMSStatistics,
    expert_ids: Sequence[int],
    pair_moments: Optional[Dict[str, PairDeltaMoments]] = None,
    config: Optional[ComposeRMSConfig] = None,
) -> Dict[str, Any]:
    """Per-expert raw/calibrated RMS plus pair diagnostics."""
    config = config or ComposeRMSConfig()
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
            calibration = build_kappa_calibration(stats, expert_ids, config)
            kappa = calibration.get(layer, {}).get(str(expert_id), 1.0)
            if kappa != 1.0:
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


def save_calibration(
    calibration: Mapping[str, Mapping[str, float]],
    path,
    provenance: Mapping[str, Any],
    config: Optional[ComposeRMSConfig] = None,
) -> str:
    """Persist the runtime kappa map atomically; returns its sha256.

    The payload is bound to the checkpoint/dataset/config hashes in
    ``provenance``; a changed checkpoint invalidates it automatically.
    """
    payload = {
        "schema_version": CALIBRATION_VERSION,
        "provenance": dict(provenance),
        "config": (config or ComposeRMSConfig()).to_dict(),
        "calibration": {
            layer: {str(expert_id): float(kappa) for expert_id, kappa in layer_map.items()}
            for layer, layer_map in sorted(calibration.items())
        },
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_calibration(path, expected_provenance: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Load a persisted kappa map; validates schema and provenance."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError("RMS calibration does not exist: {}".format(target))
    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("schema_version", -1)) != CALIBRATION_VERSION:
        raise ValueError(
            "unsupported RMS calibration schema_version: {}".format(
                payload.get("schema_version")
            )
        )
    if expected_provenance is not None and dict(payload.get("provenance", {})) != dict(expected_provenance):
        raise ValueError("RMS calibration invalidated by provenance/hash change")
    return payload


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
