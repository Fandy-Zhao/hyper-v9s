"""Atomic, versioned Query/Key checkpoints with exact metadata restoration."""

import os
import tempfile
from pathlib import Path

import torch


CHECKPOINT_VERSION = 1


def save_router_checkpoint(path, query_encoder, key_store, optimizer=None, extra=None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "query_encoder": query_encoder.state_dict(),
        "key_store": key_store.state_dict(),
        "key_metadata": key_store.metadata_state(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "extra": dict(extra or {}),
    }
    descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_router_checkpoint(path, query_encoder, key_store, optimizer=None, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    if int(payload.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError("unsupported Query/Key checkpoint version")
    if payload["key_metadata"] != key_store.metadata_state():
        raise ValueError("checkpoint expert key metadata mismatch")
    query_encoder.load_state_dict(payload["query_encoder"])
    key_store.load_state_dict(payload["key_store"])
    if optimizer is not None and payload["optimizer"] is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload.get("extra", {})


def state_fingerprint(module) -> str:
    import hashlib
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_set_router_checkpoint(path, router, extra=None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    os.close(descriptor)
    try:
        torch.save({"checkpoint_version": 1, "set_router": router.state_dict(), "extra": dict(extra or {})}, temporary)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_set_router_checkpoint(path, router, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    if payload.get("checkpoint_version") != 1:
        raise ValueError("unsupported Set Router checkpoint")
    router.load_state_dict(payload["set_router"])
    return payload.get("extra", {})
