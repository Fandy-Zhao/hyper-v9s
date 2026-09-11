"""V8 checkpoint and resume.

A resume that silently re-maps expert ids or loses an alias key would corrupt
the whole method: the pool would look fine and route to the wrong expert.  So a
V8 checkpoint carries the **full id mapping** -- every expert record and every
key record -- and :func:`verify_resume_identity` re-checks it after loading.

Checkpoints are written atomically (``tmp`` + ``os.replace``) and record the
frozen ledger digests captured at training start, so a resume can prove that
nothing historical drifted between the crash and the restart.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from compose.v8.config import V8Config
from compose.v8.gating import FrozenLedger
from compose.v8.pool import MultiKeyExpertPool, MultiKeyPoolError


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be resumed without changing identity."""


SCHEMA_VERSION = 1


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def save_checkpoint(
    path: str | Path,
    pool: MultiKeyExpertPool,
    config: V8Config,
    current_task: int,
    global_step: int = 0,
    optimizer_state: Optional[Mapping[str, Any]] = None,
    scheduler_state: Optional[Mapping[str, Any]] = None,
    frozen_ledger: Optional[FrozenLedger] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist the pool, the id mapping and the run contract."""
    path = Path(path)
    pool_state = pool.export_state()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pool_kind": "v8_multi_key",
        "current_task": int(current_task),
        "global_step": int(global_step),
        "num_experts": len(pool.expert_records),
        "num_keys": len(pool.key_records),
        "expert_ids": pool.expert_ids(),
        "key_ids": pool.key_ids(),
        "origin_key_by_expert": {
            str(expert_id): pool.origin_key_id(expert_id)
            for expert_id in pool.expert_ids()
        },
        "alias_keys": {
            key_id: {
                "expert_id": pool.key_records[key_id]["expert_id"],
                "task_id": pool.key_records[key_id]["task_id"],
                "support_count": pool.key_records[key_id]["support_count"],
            }
            for key_id in pool.key_ids(key_type="task_alias")
        },
        "config": config.to_dict(),
        "config_seed": int(config.seed),
        "extra": dict(extra or {}),
    }
    _atomic_torch_save(
        {
            "schema_version": SCHEMA_VERSION,
            "pool_state": pool_state,
            "current_task": int(current_task),
            "global_step": int(global_step),
            "optimizer_state": dict(optimizer_state) if optimizer_state else None,
            "scheduler_state": dict(scheduler_state) if scheduler_state else None,
            "frozen_ledger": frozen_ledger.to_dict() if frozen_ledger else None,
        },
        path,
    )
    _atomic_json(manifest, path.with_name(path.name + ".manifest.json"))
    return manifest


def load_checkpoint(
    path: str | Path,
    current_task: Optional[int] = None,
) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"no checkpoint at {path}")
    payload = torch.load(path, map_location="cpu")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointError(
            f"unsupported V8 checkpoint schema {payload.get('schema_version')}"
        )
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task = int(payload["current_task"] if current_task is None else current_task)
    pool = MultiKeyExpertPool.from_state(payload["pool_state"], current_task=task)
    ledger = None
    if payload.get("frozen_ledger"):
        raw = payload["frozen_ledger"]
        ledger = FrozenLedger(
            key_checksums=dict(raw.get("historical_key_checksums", {})),
            lora_checksums={int(k): v for k, v in
                            raw.get("historical_lora_checksums", {}).items()},
            edge_checksums=dict(raw.get("candidate_edges", {})),
        )
    return {
        "pool": pool,
        "manifest": manifest,
        "current_task": int(payload["current_task"]),
        "global_step": int(payload.get("global_step", 0)),
        "optimizer_state": payload.get("optimizer_state"),
        "scheduler_state": payload.get("scheduler_state"),
        "frozen_ledger": ledger,
    }


def verify_resume_identity(
    loaded: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """The resumed pool must be the same pool, id for id.

    Compares expert ids, key ids and the origin-key mapping.  A mismatch is a
    hard error: continuing from a re-mapped pool would train keys against the
    wrong experts while every downstream file still looked plausible.
    """
    pool: MultiKeyExpertPool = loaded["pool"]
    for field, actual in (
        ("expert_ids", pool.expert_ids()),
        ("key_ids", pool.key_ids()),
    ):
        expected = [int(v) for v in expected_manifest.get(field, [])] if field == "expert_ids" \
            else [str(v) for v in expected_manifest.get(field, [])]
        if expected != actual:
            raise CheckpointError(
                f"resume changed {field}: expected {expected}, resumed {actual}"
            )
    expected_origin = {
        str(k): str(v) for k, v in expected_manifest.get("origin_key_by_expert", {}).items()
    }
    actual_origin = {
        str(expert_id): pool.origin_key_id(expert_id) for expert_id in pool.expert_ids()
    }
    if expected_origin and expected_origin != actual_origin:
        raise CheckpointError(
            "resume changed the expert -> origin-key mapping: "
            f"{expected_origin} != {actual_origin}"
        )
    expected_aliases = {
        str(k): int(v["expert_id"]) for k, v in expected_manifest.get("alias_keys", {}).items()
    }
    actual_aliases = {
        key_id: int(pool.key_records[key_id]["expert_id"])
        for key_id in pool.key_ids(key_type="task_alias")
    }
    if expected_aliases != actual_aliases:
        raise CheckpointError(
            f"resume changed alias keys: {expected_aliases} != {actual_aliases}"
        )
    return {
        "status": "IDENTICAL",
        "experts": len(pool.expert_records),
        "keys": len(pool.key_records),
        "alias_keys": len(actual_aliases),
        "origin_key_by_expert": actual_origin,
    }


def verify_config_compatibility(
    expected: Mapping[str, Any],
    resumed: Mapping[str, Any],
    keys: Sequence[str] = ("method", "seed"),
) -> Dict[str, Any]:
    """A resume must not silently change the method or the seed."""
    mismatched = {
        key: (expected.get(key), resumed.get(key))
        for key in keys
        if expected.get(key) != resumed.get(key)
    }
    if mismatched:
        raise CheckpointError(f"resume changed the run contract: {mismatched}")
    return {"status": "COMPATIBLE", "checked": list(keys)}


def prune_resume_state(optimizer_state: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Report optimizer state size so a resume can be validated cheaply."""
    if not optimizer_state:
        return {"param_groups": 0, "state_entries": 0}
    state = optimizer_state.get("state", {})
    return {
        "param_groups": len(optimizer_state.get("param_groups", [])),
        "state_entries": len(state),
    }


__all__ = [
    "CheckpointError",
    "SCHEMA_VERSION",
    "load_checkpoint",
    "prune_resume_state",
    "save_checkpoint",
    "verify_config_compatibility",
    "verify_resume_identity",
]
