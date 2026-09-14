import json
import os
import re
import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

from compose.adapters.lora import ComposeLinear

from .pool import ExpertPool
from .registry import ExpertRegistry


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
    rms_calibration: Optional[Dict[str, Dict[str, float]]] = None,
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
    # Runtime RMS kappa persistence: per-layer {expert_id: kappa_k_l}.
    # Additive key (format stays version 1); the loader returns it in the
    # manifest so inference can apply kappa through
    # ComposeLinear.set_expert_calibration without re-collecting statistics.
    if rms_calibration:
        manifest["rms_calibration"] = {
            layer: {str(expert_id): float(kappa) for expert_id, kappa in layer_map.items()}
            for layer, layer_map in sorted(rms_calibration.items())
        }
    with open(os.path.join(output_dir, MANIFEST_NAME), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_expert_checkpoint(
    pool: ExpertPool,
    checkpoint_dir: str,
    map_location: str = "cpu",
    keep_ids: Optional[Iterable[int]] = None,
) -> Dict[str, object]:
    """Load a Compose checkpoint, optionally instantiating only ``keep_ids``.

    ``keep_ids`` is a memory lever for inference-only runs, not a change to
    what the checkpoint contains: the manifest is still validated in full and
    the tensor-count check still counts every expert.  Only the modules for
    the listed ids are built, so an unlisted expert stays absent from
    ``layer.experts`` and can never be selected.  Callers must obtain
    ``keep_ids`` from the routing manifest (the experts the run actually
    selects); guessing a smaller set would silently drop a selected expert
    and fail loudly at selection time rather than silently score wrong.
    """
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
    pool.restore_metadata(manifest["experts"], keep_ids=keep_ids)
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
        "instantiated_experts": sorted(pool.manager.expert_ids()),
        "declared_experts": len(pool.expert_ids()),
    }
    return manifest


def _registry_checkpoint_payload(
    registry: ExpertRegistry,
    source_git_commit: str,
    source_model_identifier: str,
    source_adapter_config_hash: str,
    timestamp: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, object]:
    if not source_git_commit:
        raise ValueError("source_git_commit must not be empty")
    if not source_model_identifier:
        raise ValueError("source_model_identifier must not be empty")
    if not source_adapter_config_hash:
        raise ValueError("source_adapter_config_hash must not be empty")
    payload = registry.state_dict()
    payload["provenance"] = {
        "source_git_commit": source_git_commit,
        "source_model_identifier": source_model_identifier,
        "source_adapter_config_hash": source_adapter_config_hash,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
    }
    return payload


def save_registry_checkpoint(
    registry: ExpertRegistry,
    path,
    source_git_commit: str,
    source_model_identifier: str,
    source_adapter_config_hash: str,
    timestamp: Optional[str] = None,
    run_id: Optional[str] = None,
) -> str:
    """Atomically save metadata/provenance only and return its SHA-256.

    Existing paths are intentionally rejected.  Registry checkpoints never
    contain tensors; adapter weights remain owned by the existing Hyper loader.
    """

    target = Path(path)
    if target.exists():
        raise FileExistsError("registry checkpoint already exists: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _registry_checkpoint_payload(
        registry,
        source_git_commit,
        source_model_identifier,
        source_adapter_config_hash,
        timestamp=timestamp,
        run_id=run_id,
    )
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(encoded).hexdigest()


def load_registry_checkpoint(path):
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError("registry checkpoint does not exist: {}".format(target))
    encoded = target.read_bytes()
    payload = json.loads(encoded.decode("utf-8"))
    registry = ExpertRegistry()
    registry.load_state_dict(payload)
    return registry, payload.get("provenance", {}), hashlib.sha256(encoded).hexdigest()
