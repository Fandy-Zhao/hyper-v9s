"""Atomic Sufficiency Head checkpoint helpers."""

import os
import tempfile
from pathlib import Path

import torch


def save_sufficiency_checkpoint(path, model, optimizer=None, extra=None):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    os.close(descriptor)
    try:
        torch.save({"version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None, "extra": dict(extra or {})}, temporary)
        os.replace(temporary, target)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def load_sufficiency_checkpoint(path, model, optimizer=None, map_location="cpu"):
    value = torch.load(path, map_location=map_location)
    if value.get("version") != 1:
        raise ValueError("unsupported sufficiency checkpoint")
    model.load_state_dict(value["model"])
    if optimizer is not None and value["optimizer"] is not None:
        optimizer.load_state_dict(value["optimizer"])
    return value["extra"]
