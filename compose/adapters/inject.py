from typing import Dict, Iterable, List, Sequence, Tuple

import torch.nn as nn

from compose.config import ComposeAdapterConfig

from .lora import ComposeLinear


DECODER_PROJECTION_PATHS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)  # type: Tuple[str, ...]
DECODER_PROJECTION_NAMES = frozenset(path.rsplit(".", 1)[-1] for path in DECODER_PROJECTION_PATHS)


def _decoder_layers(model: nn.Module) -> Tuple[str, Sequence[nn.Module]]:
    if not hasattr(model, "get_model") or not callable(model.get_model):
        raise TypeError("Compose injection requires a model with get_model().layers")
    decoder = model.get_model()
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise TypeError("Compose injection requires LLaMA decoder layers")
    decoder_names = [name for name, module in model.named_modules() if module is decoder]
    if len(decoder_names) != 1:
        raise ValueError("could not determine the unique decoder module path")
    return decoder_names[0], layers


def _resolve_parent(root: nn.Module, path: str) -> Tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _expected_targets(model: nn.Module) -> List[Tuple[str, nn.Module, str, nn.Module]]:
    decoder_name, layers = _decoder_layers(model)
    targets = []
    for layer_index, layer in enumerate(layers):
        for relative_path in DECODER_PROJECTION_PATHS:
            parent, child_name = _resolve_parent(layer, relative_path)
            module = getattr(parent, child_name)
            prefix = decoder_name + "." if decoder_name else ""
            full_name = "{}layers.{}.{}".format(prefix, layer_index, relative_path)
            targets.append((full_name, parent, child_name, module))
    if not targets:
        raise ValueError("Compose injection found no LLaMA decoder layers")
    return targets


def decoder_projection_names(model: nn.Module) -> List[str]:
    """Return the exact decoder projection boundary used by Compose and PEFT."""

    return [name for name, _, _, _ in _expected_targets(model)]


def validate_compose_injection(
    model: nn.Module, injected_names: Iterable[str] = None
) -> Dict[str, object]:
    """Validate that every and only decoder projection is Compose-enabled."""

    expected = {entry[0] for entry in _expected_targets(model)}
    actual = {
        name for name, module in model.named_modules() if isinstance(module, ComposeLinear)
    }
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if injected_names is not None:
        reported = set(injected_names)
        if reported != expected:
            raise ValueError(
                "reported Compose injection set does not match decoder boundary; "
                "missing={}, unexpected={}".format(
                    sorted(expected - reported), sorted(reported - expected)
                )
            )
    if missing or unexpected:
        raise ValueError(
            "invalid Compose injection boundary; missing={}, unexpected={}".format(
                missing, unexpected
            )
        )
    excluded = {
        "vision_tower": sum("vision_tower" in name for name in actual),
        "mm_projector": sum("mm_projector" in name for name in actual),
        "lm_head": sum(name == "lm_head" or name.startswith("lm_head.") for name in actual),
    }
    return {
        "decoder_layers": len(_decoder_layers(model)[1]),
        "injected_layers": len(actual),
        "expected_layers": len(expected),
        "vision_tower_injected": excluded["vision_tower"],
        "mm_projector_injected": excluded["mm_projector"],
        "lm_head_injected": excluded["lm_head"],
    }


def inject_compose_adapters(
    model: nn.Module,
    config: ComposeAdapterConfig,
    target_modules: Iterable[str] = None,
) -> List[str]:
    """Replace exactly the seven projections in each LLaMA decoder layer."""

    configured = set(target_modules or config.target_modules)
    if configured != DECODER_PROJECTION_NAMES:
        raise ValueError(
            "Compose Foundation requires exactly the decoder targets {}; got {}".format(
                sorted(DECODER_PROJECTION_NAMES), sorted(configured)
            )
        )
    targets = _expected_targets(model)
    duplicates = [name for name, _, _, module in targets if isinstance(module, ComposeLinear)]
    invalid = [
        "{} ({})".format(name, type(module).__name__)
        for name, _, _, module in targets
        if not isinstance(module, (nn.Linear, ComposeLinear))
    ]
    if duplicates:
        raise ValueError("Compose adapters are already injected: {}".format(duplicates))
    if invalid:
        raise TypeError("decoder projection targets must be nn.Linear: {}".format(invalid))

    matches = []
    for name, parent, child_name, module in targets:
        setattr(
            parent,
            child_name,
            ComposeLinear(
                module,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            ),
        )
        matches.append(name)
    validate_compose_injection(model, matches)
    return matches
