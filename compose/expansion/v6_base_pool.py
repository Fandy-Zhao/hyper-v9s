"""Frozen-backbone-only checkpoint pool (empty-registry fix, Stage R4).

When the active expert registry is empty, the current system IS the frozen
backbone: teacher scoring, residual construction and task-boundary
evaluation run against a checkpoint directory that contains an adapter
manifest (so the Compose loader can inject) but **zero experts** and an
empty weight state dict.  ``selected_experts = []`` then genuinely means
backbone-only; no rejected or temporary candidate checkpoint is ever
loaded in its place.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import torch

MANIFEST_NAME = "compose_experts.json"
WEIGHTS_NAME = "compose_experts.bin"

# llava-v1.5-7b decoder (vicuna-7b): 32 layers x the 7 injected projections.
DEFAULT_LAYER_COUNT = 32
INJECTED_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def decoder_layer_names(layer_count: int = DEFAULT_LAYER_COUNT) -> List[str]:
    """Deterministic decoder layer list for the Compose injection manifest."""
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    return sorted(
        "model.layers.{}.{}".format(layer, module)
        for layer in range(int(layer_count))
        for module in INJECTED_MODULES
    )


def write_base_only_checkpoint(
    output_dir,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    layer_names: Optional[List[str]] = None,
) -> Path:
    """Write an empty-pool checkpoint directory (manifest + empty weights).

    ``load_expert_checkpoint`` accepts it: ``experts=[]`` and an empty state
    dict validate against empty expected-key sets, and the per-sample
    selection is always backbone-only.
    """
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    weights_path = target / WEIGHTS_NAME
    torch.save({}, str(weights_path))
    manifest = {
        "format_version": 1,
        "kind": "base_only_pool",
        "experts": [],
        "adapter": {
            "rank": int(rank),
            "alpha": float(alpha),
            "dropout": float(dropout),
            "layers": layer_names if layer_names is not None else decoder_layer_names(),
        },
        "metrics": {
            "adapter_tensor_count": 0,
            "adapter_parameter_count": 0,
            "checkpoint_bytes": weights_path.stat().st_size,
        },
        "note": "frozen backbone only: no active experts in the registry; "
        "never contains rejected or temporary candidate weights",
    }
    manifest_path = target / MANIFEST_NAME
    descriptor, temporary = tempfile.mkstemp(
        prefix=manifest_path.name + ".", suffix=".tmp", dir=str(target)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target
