import json
import os
from typing import Dict, Iterable, Optional

import torch

from compose.adapters.lora import ComposeLinear

from .pool import ExpertPool


MANIFEST_NAME = "compose_experts.json"
WEIGHTS_NAME = "compose_experts.bin"


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
    selected = pool.expert_ids() if expert_ids is None else sorted(set(expert_ids))
    for expert_id in selected:
        pool.get(expert_id)
    os.makedirs(output_dir, exist_ok=True)
    manifest = pool.to_dict()
    manifest["experts"] = [pool.get(value).to_dict() for value in selected]
    first_layer = next(iter(pool.manager.layers.values()))
    manifest["adapter"] = {
        "rank": first_layer.rank,
        "alpha": first_layer.alpha,
        "dropout": first_layer.dropout,
        "layers": sorted(pool.manager.layers.keys()),
    }
    with open(os.path.join(output_dir, MANIFEST_NAME), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    torch.save(expert_state_dict(pool, selected), os.path.join(output_dir, WEIGHTS_NAME))


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
    actual_layers = set(pool.manager.layers.keys())
    if expected_layers and expected_layers != actual_layers:
        raise ValueError("checkpoint target layers do not match the current model")
    pool.restore_metadata(manifest["experts"])
    state = torch.load(weights_path, map_location=map_location)
    missing = []
    for layer_name, layer in pool.manager.layers.items():
        for expert_key, expert in layer.experts.items():
            prefix = "{}.experts.{}.".format(layer_name, expert_key)
            expert_state = {
                key[len(prefix):]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            if expert_state:
                expert.load_state_dict(expert_state, strict=True)
            elif int(expert_key) in pool.expert_ids():
                missing.append(prefix)
    if missing:
        raise ValueError("checkpoint is missing expert weights: {}".format(missing))
    return manifest
