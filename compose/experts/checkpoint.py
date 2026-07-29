import json
import os
import re
from typing import Dict, Iterable, Optional

import torch

from compose.adapters.lora import ComposeLinear

from .pool import ExpertPool


MANIFEST_NAME = "compose_experts.json"
WEIGHTS_NAME = "compose_experts.bin"
_DECODER_LAYER_RE = re.compile(
    r"^model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
_EXPERT_PARAMETER_NAMES = ("lora_A.weight", "lora_B.weight")


def _validated_layers(pool: ExpertPool):
    invalid = sorted(
        name for name in pool.manager.layers if _DECODER_LAYER_RE.fullmatch(name) is None
    )
    if invalid:
        raise ValueError(
            "Compose checkpoints may contain decoder projections only: {}".format(invalid)
        )
    return sorted(pool.manager.layers)


def _expected_keys(pool: ExpertPool, expert_ids: Iterable[int]):
    return {
        "{}.experts.{}.{}".format(layer_name, int(expert_id), parameter_name)
        for layer_name in _validated_layers(pool)
        for expert_id in expert_ids
        for parameter_name in _EXPERT_PARAMETER_NAMES
    }


def _validate_state_keys(state, expected_keys, context: str) -> None:
    actual_keys = set(state)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "{} expert tensor keys do not match exactly; expected={}, actual={}, "
            "missing={}, unexpected={}".format(
                context, len(expected_keys), len(actual_keys), missing, unexpected
            )
        )


def _gather_parameter(parameter: torch.nn.Parameter, name: str) -> torch.Tensor:
    if hasattr(parameter, "ds_id"):
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

        if parameter.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            with zero.GatheredParameters([parameter]):
                return parameter.detach().cpu().clone()
    return parameter.detach().cpu().clone()


def expert_state_dict(
    pool: ExpertPool, expert_ids: Optional[Iterable[int]] = None
) -> Dict[str, torch.Tensor]:
    selected = set(pool.expert_ids() if expert_ids is None else expert_ids)
    state = {}
    for layer_name, layer in pool.manager.layers.items():
        for expert_key, expert in layer.experts.items():
            if int(expert_key) not in selected:
                continue
            for parameter_name, parameter in expert.named_parameters():
                key = "{}.experts.{}.{}".format(
                    layer_name, expert_key, parameter_name
                )
                state[key] = _gather_parameter(parameter, key)
    return state


def save_expert_checkpoint(
    pool: ExpertPool,
    output_dir: str,
    expert_ids: Optional[Iterable[int]] = None,
) -> None:
    selected = sorted(
        int(value) for value in (pool.expert_ids() if expert_ids is None else set(expert_ids))
    )
    for expert_id in selected:
        pool.get(expert_id)
    os.makedirs(output_dir, exist_ok=True)
    state = expert_state_dict(pool, selected)
    expected_keys = _expected_keys(pool, selected)
    _validate_state_keys(state, expected_keys, "saving")
    weights_path = os.path.join(output_dir, WEIGHTS_NAME)
    torch.save(state, weights_path)

    manifest = pool.to_dict()
    manifest["experts"] = [pool.get(value).to_dict() for value in selected]
    first_layer = next(iter(pool.manager.layers.values()))
    manifest["adapter"] = {
        "rank": first_layer.rank,
        "alpha": first_layer.alpha,
        "dropout": first_layer.dropout,
        "layers": _validated_layers(pool),
    }
    manifest["metrics"] = {
        "adapter_tensor_count": len(state),
        "adapter_parameter_count": sum(tensor.numel() for tensor in state.values()),
        "checkpoint_bytes": os.path.getsize(weights_path),
    }
    with open(os.path.join(output_dir, MANIFEST_NAME), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_expert_checkpoint(
    pool: ExpertPool,
    checkpoint_dir: str,
    map_location: str = "cpu",
) -> Dict[str, object]:
    manifest_path = os.path.join(checkpoint_dir, MANIFEST_NAME)
    weights_path = os.path.join(checkpoint_dir, WEIGHTS_NAME)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported Compose checkpoint format")
    expected_layers = set(manifest.get("adapter", {}).get("layers", []))
    actual_layers = set(_validated_layers(pool))
    if expected_layers != actual_layers:
        raise ValueError(
            "checkpoint target layers do not match the current model; missing={}, "
            "unexpected={}".format(
                sorted(actual_layers - expected_layers),
                sorted(expected_layers - actual_layers),
            )
        )
    pool.restore_metadata(manifest["experts"])
    state = torch.load(weights_path, map_location=map_location)
    if not isinstance(state, dict):
        raise TypeError("Compose checkpoint weights must be a tensor dictionary")
    expected_keys = _expected_keys(pool, pool.expert_ids())
    _validate_state_keys(state, expected_keys, "loading")
    metrics = manifest.get("metrics", {})
    declared_count = metrics.get("adapter_tensor_count")
    if declared_count != len(expected_keys):
        raise ValueError(
            "checkpoint manifest tensor count does not match; declared={}, expected={}".format(
                declared_count, len(expected_keys)
            )
        )
    for layer_name, layer in pool.manager.layers.items():
        for expert_key, expert in layer.experts.items():
            prefix = "{}.experts.{}.".format(layer_name, expert_key)
            expert_state = {
                key[len(prefix):]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            expert.load_state_dict(expert_state, strict=True)
    manifest["load_summary"] = {
        "expected_tensor_count": len(expected_keys),
        "loaded_tensor_count": len(state),
        "missing_tensor_count": 0,
        "unexpected_tensor_count": 0,
    }
    return manifest
