from typing import Iterable, List, Tuple

import torch.nn as nn

from compose.config import ComposeAdapterConfig

from .lora import ComposeLinear


def _parent_and_child(root: nn.Module, path: str) -> Tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_compose_adapters(
    model: nn.Module,
    config: ComposeAdapterConfig,
    target_modules: Iterable[str] = None,
) -> List[str]:
    """Replace selected nn.Linear leaves and return their full module names."""

    targets = set(target_modules or config.target_modules)
    matches = []
    for name, module in list(model.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue
        if name.split(".")[-1] not in targets:
            continue
        parent, child_name = _parent_and_child(model, name)
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
    if not matches:
        raise ValueError("no target linear modules matched {}".format(sorted(targets)))
    return matches
