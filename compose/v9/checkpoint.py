"""Atomic V9 checkpoint/resume (spec §27).

Everything a resumed run needs to be indistinguishable from an uninterrupted
one is written here: the expert pool (base keys, current-task keys,
candidate keys, RMS state), the candidate LoRA weights, the router bias, the
optimizer and scheduler, the RNG streams, the training stage and step, and the
task-end audit accumulators.

The write is atomic: a temporary file is fsynced and then ``os.replace``d, so a
crash mid-save can never leave a half-written payload where a checkpoint should
be.
"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from compose.v7.checkpoint import capture_rng_state, restore_rng_state

from .config import V9Config
from .keys import V9KeyPool


V9_CHECKPOINT_VERSION = 1


def save_v9_checkpoint(
    path: str,
    *,
    task_index: int,
    training_step: int,
    stage: str,
    stage_progress: float,
    key_pool: V9KeyPool,
    router_state: Mapping[str, Any],
    candidate_lora_state: Mapping[str, object],
    optimizer,
    scheduler,
    usage_counters: Mapping[str, object],
    config: V9Config,
    rms_state: Optional[Mapping[str, object]] = None,
    dataset_fingerprint: Optional[str] = None,
    query_fingerprint: Optional[str] = None,
    retrieval_fingerprint: Optional[str] = None,
) -> None:
    payload = {
        "checkpoint_version": V9_CHECKPOINT_VERSION,
        "method": config.method,
        "task_index": int(task_index),
        "training_step": int(training_step),
        "stage": str(stage),
        "stage_progress": float(stage_progress),
        "global_expert_registry": key_pool.export_state(),
        # The router owns the routing geometry the pool does not carry: the
        # expert ordering the bias vector is indexed by.
        "router_state": dict(router_state),
        "candidate_lora_state": dict(candidate_lora_state),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": capture_rng_state(),
        "rms_state": dict(rms_state or {}),
        "candidate_usage_counters": dict(usage_counters),
        "config": config.to_dict(),
        "pool_version": len(key_pool.key_records),
        "fingerprints": {
            "dataset": dataset_fingerprint,
            "query": query_fingerprint,
            "retrieval": retrieval_fingerprint,
        },
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


def load_v9_checkpoint(
    path: str, current_task: Optional[int] = None
) -> Tuple[Dict[str, Any], V9KeyPool, V9Config]:
    """Load a V9 checkpoint, restoring the pool's trainability registry.

    ``load_state_dict`` on a key store creates no lifecycle guarantees, so the
    frozen set is re-asserted here rather than trusted from the file: a
    corrupt or hand-edited checkpoint must not be able to make a historical
    base key trainable.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("checkpoint_version", -1)) != V9_CHECKPOINT_VERSION:
        raise ValueError("unsupported V9 checkpoint version")
    if payload.get("method") != config_method():
        raise ValueError("checkpoint is not Hyper-LLaVA V9")
    config = V9Config.from_dict(payload["config"])
    pool = V9KeyPool.from_state(
        payload["global_expert_registry"], current_task=current_task
    )
    task_index = int(payload["task_index"])
    # Anything on a task other than the one being resumed is frozen outright,
    # so a checkpoint cannot widen the trainable set by being loaded.
    pool.freeze_historical(
        current_task=task_index if current_task is None else int(current_task)
    )
    return payload, pool, config


def config_method() -> str:
    from .config import V9_METHOD_NAME

    return V9_METHOD_NAME


__all__ = [
    "V9_CHECKPOINT_VERSION",
    "capture_rng_state",
    "load_v9_checkpoint",
    "restore_rng_state",
    "save_v9_checkpoint",
]
