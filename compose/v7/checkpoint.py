"""Atomic V7 checkpoint/resume including optimizer, RNG and lifecycle state."""

import os
import random
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch

from .config import V7Config
from .pool import V7ExpertKeyPool


V7_CHECKPOINT_VERSION = 1


def capture_rng_state() -> Dict[str, object]:
    state = {
        "python": random.getstate(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    torch.random.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_v7_checkpoint(
    path: str,
    *,
    task_index: int,
    training_step: int,
    key_pool: V7ExpertKeyPool,
    candidate_lora_state: Mapping[str, object],
    optimizer,
    scheduler,
    usage_counters: Mapping[str, object],
    config: V7Config,
    rms_state: Optional[Mapping[str, object]] = None,
) -> None:
    payload = {
        "checkpoint_version": V7_CHECKPOINT_VERSION,
        "method": config.method,
        "task_index": int(task_index),
        "training_step": int(training_step),
        "global_expert_registry": key_pool.export_state(),
        "candidate_lora_state": dict(candidate_lora_state),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": capture_rng_state(),
        "rms_state": dict(rms_state or {}),
        "candidate_usage_counters": dict(usage_counters),
        "config": config.to_dict(),
        "pool_version": key_pool.pool_version,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_v7_checkpoint(
    path: str, optimizer=None, scheduler=None, restore_rng: bool = True
) -> Tuple[Dict[str, object], V7ExpertKeyPool, V7Config]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("checkpoint_version", -1)) != V7_CHECKPOINT_VERSION:
        raise ValueError("unsupported V7 checkpoint version")
    if payload.get("method") != "v7_global_coevolution":
        raise ValueError("checkpoint is not Hyper-LLaVA V7")
    config = V7Config.from_dict(payload["config"])
    pool = V7ExpertKeyPool.from_state(payload["global_expert_registry"])
    # load_state_dict creates no lifecycle guarantees; reassert them here.
    pool.freeze_historical()
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload, pool, config

