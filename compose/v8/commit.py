"""Task commit: turn a finished task's candidates into committed pool state.

At the end of a task, three things become permanent:

* the current task's **alias keys** (one per historical expert the teacher
  actually selected, already created lazily and trained by the key loss),
* the current task's **candidate experts** (the LoRA that learned the residual
  capability the historical pool lacked),
* the current task's **candidate keys** (the origin keys of those new experts,
  which is what makes them reachable in later tasks).

Committing is the point where the freeze contract flips: what was trainable
becomes frozen history, and the V7-compatible artifacts must be written so the
next task -- and the existing V7 evaluator -- can read the pool.  Because a
half-finished commit would leave a pool that trains against the wrong experts,
the validation below runs *before* anything is written, and every commit
records the checksums it committed to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from compose.v8.config import KEY_TYPE_ORIGIN, KEY_TYPE_TASK_ALIAS
from compose.v8.pool import (
    LIFECYCLE_CANDIDATE,
    LIFECYCLE_HISTORICAL,
    LIFECYCLE_PRUNED,
    MultiKeyExpertPool,
    MultiKeyPoolError,
    tensor_checksum,
)


class CommitError(RuntimeError):
    """Raised when a task cannot be committed without corrupting the pool."""


@dataclass
class CommitReport:
    task_id: int
    committed_experts: List[int] = field(default_factory=list)
    committed_alias_keys: List[str] = field(default_factory=list)
    committed_origin_keys: List[str] = field(default_factory=list)
    frozen_keys: List[str] = field(default_factory=list)
    key_checksums: Dict[str, str] = field(default_factory=dict)
    num_experts_after: int = 0
    num_keys_after: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": int(self.task_id),
            "committed_experts": list(self.committed_experts),
            "committed_alias_keys": list(self.committed_alias_keys),
            "committed_origin_keys": list(self.committed_origin_keys),
            "frozen_keys": list(self.frozen_keys),
            "key_checksums": dict(self.key_checksums),
            "num_experts_after": int(self.num_experts_after),
            "num_keys_after": int(self.num_keys_after),
        }


def validate_precommit(
    pool: MultiKeyExpertPool,
    task_id: int,
    candidate_expert_ids: Iterable[int],
) -> Dict[str, Any]:
    """Everything that must hold before a commit mutates the pool.

    The failure this prevents: committing a candidate expert that has no origin
    key, which would leave an expert present in the pool but unreachable by the
    router -- silently invisible in every later task.
    """
    candidates = sorted({int(value) for value in candidate_expert_ids})
    problems: List[str] = []
    for expert_id in candidates:
        if expert_id not in pool.expert_records:
            problems.append(f"candidate expert {expert_id} is not in the pool")
            continue
        origin_id = pool.origin_key_id(expert_id)
        if origin_id not in pool.key_records:
            problems.append(
                f"candidate expert {expert_id} has no origin key ({origin_id})"
            )
            continue
        if pool.key_records[origin_id]["task_id"] != int(task_id):
            problems.append(
                f"candidate expert {expert_id} origin key is on task "
                f"{pool.key_records[origin_id]['task_id']}, expected {task_id}"
            )
    for key_id, record in pool.key_records.items():
        if record["task_id"] != int(task_id):
            continue
        if record["key_type"] == KEY_TYPE_TASK_ALIAS:
            if int(record["expert_id"]) in candidates:
                problems.append(
                    f"{key_id} is an alias key on a candidate expert; the origin "
                    "key already covers this task"
                )
        elif record["key_type"] != KEY_TYPE_ORIGIN:
            problems.append(f"{key_id} has an unknown key type")
    if problems:
        raise CommitError(
            f"task {task_id} cannot be committed: " + "; ".join(problems[:8])
        )
    return {
        "task_id": int(task_id),
        "candidates": candidates,
        "checks": ["candidate_origin_keys_present", "no_alias_on_candidate"],
        "ok": True,
    }


def commit_task(
    pool: MultiKeyExpertPool,
    task_id: int,
    candidate_expert_ids: Iterable[int] = (),
    extra: Optional[Mapping[str, Any]] = None,
) -> CommitReport:
    """Freeze this task's new state and make it part of the pool's history."""
    task_id = int(task_id)
    candidates = sorted({int(value) for value in candidate_expert_ids})
    validate_precommit(pool, task_id, candidates)

    report = CommitReport(task_id=task_id)
    for key_id, record in pool.key_records.items():
        if record["lifecycle"] == LIFECYCLE_PRUNED:
            continue
        was_trainable = bool(record["trainable"]) or \
            record["lifecycle"] == LIFECYCLE_CANDIDATE
        if int(record["task_id"]) == task_id and was_trainable:
            # This is state the task produced; it becomes frozen history now.
            record["lifecycle"] = LIFECYCLE_HISTORICAL
            record["trainable"] = False
            pool.keys[key_id].requires_grad = False
            if record["key_type"] == KEY_TYPE_TASK_ALIAS:
                report.committed_alias_keys.append(key_id)
            else:
                report.committed_origin_keys.append(key_id)
        else:
            # Everything else was already frozen and stays exactly as it was.
            report.frozen_keys.append(key_id)

    for expert_id in candidates:
        record = pool.expert_records[expert_id]
        if record["lifecycle"] == LIFECYCLE_PRUNED:
            raise CommitError(f"candidate expert {expert_id} is already pruned")
        record["lifecycle"] = LIFECYCLE_HISTORICAL
        record["creation_task"] = task_id
        report.committed_experts.append(expert_id)

    pool.validate()
    report.committed_alias_keys.sort()
    report.committed_origin_keys.sort()
    report.frozen_keys.sort()
    report.key_checksums = pool.key_checksums()
    report.num_experts_after = len(pool.expert_records)
    report.num_keys_after = len(pool.key_records)
    return report


def write_v7_compatible_keys(
    pool: MultiKeyExpertPool,
    path: str | Path,
    pool_version: Optional[int] = None,
) -> Dict[str, Any]:
    """Write the V7-shaped key store so the existing evaluator can read the pool.

    V8 keeps every key it owns; this file exposes only the **origin** keys, which
    are exactly what a V7 store contains.  Writing the alias keys here would make
    the V7 evaluator -- which assumes one key per expert -- route to an expert
    through a key that belongs to a later task.
    """
    path = Path(path)
    origin_keys: Dict[str, torch.Tensor] = {}
    metadata: Dict[str, Dict[str, Any]] = {}
    for expert_id in sorted(pool.expert_ids()):
        record = pool.expert_records[expert_id]
        if record["lifecycle"] == LIFECYCLE_PRUNED:
            continue
        origin_id = pool.origin_key_id(expert_id)
        origin_keys[str(expert_id)] = pool.keys[origin_id].detach().clone()
        metadata[str(expert_id)] = {
            "expert_id": int(expert_id),
            "origin_task": int(record["origin_task"]),
            "lifecycle": str(record["lifecycle"]),
            "rms_state": record.get("rms_state"),
        }
    payload = {
        "schema_version": 1,
        "query_dim": int(pool.query_dim),
        "pool_version": int(pool_version if pool_version is not None else len(origin_keys)),
        "keys": origin_keys,
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return {
        "path": str(path),
        "experts": len(origin_keys),
        "query_dim": int(pool.query_dim),
        "sha256": tensor_checksum(payload["keys"][sorted(origin_keys)[0]]) if origin_keys else None,
        "alias_keys_written": 0,
    }


def write_v8_state(
    pool: MultiKeyExpertPool,
    path: str | Path,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Write the full V8 pool state (every key, alias keys included)."""
    path = Path(path)
    payload = {
        "schema_version": 1,
        "pool_kind": "v8_multi_key",
        "query_dim": int(pool.query_dim),
        "pool_state": pool.export_state(),
        "audit": pool.audit(),
        "extra": dict(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    manifest = {
        "schema_version": 1,
        "num_experts": len(pool.expert_records),
        "num_keys": len(pool.key_records),
        "key_lifecycle_counts": pool.audit()["key_lifecycle_counts"],
        "key_checksums": pool.key_checksums(),
        "extra": dict(extra or {}),
    }
    manifest_path = path.with_name(path.name + ".manifest.json")
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(manifest_path)
    return manifest


def assert_commit_immutable(
    report: CommitReport,
    pool: MultiKeyExpertPool,
) -> Dict[str, Any]:
    """After a commit, every committed tensor must still match its checksum."""
    changed = [
        key_id for key_id, expected in report.key_checksums.items()
        if key_id not in pool.key_records
        or tensor_checksum(pool.keys[key_id]) != expected
    ]
    if changed:
        raise CommitError(f"committed keys changed after commit: {changed[:8]}")
    return {
        "status": "IMMUTABLE",
        "checked": len(report.key_checksums),
        "changed": changed,
    }


__all__ = [
    "CommitError",
    "CommitReport",
    "assert_commit_immutable",
    "commit_task",
    "validate_precommit",
    "write_v7_compatible_keys",
    "write_v8_state",
]
