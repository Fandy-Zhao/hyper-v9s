"""Rejected candidate artifacts and registry registration (Stage R2).

A candidate that fails validation (below_tau) keeps its training artifacts
for offline diagnosis but never participates in any formal behavior:

  allowed:      LoRA checkpoint, key, validation metrics, rejection reason,
                training logs, offline diagnostics
  forbidden:    active expert registry view, Router retrieval, teacher search
                candidate set, selected_experts, pair composition, formal RMS,
                default/fallback adapter, later-task evaluation, later-task
                residual baseline

Artifact layout under the task root::

    rejected_candidates/
        task_00/
            candidate_00/
                adapter/            (copied candidate pool checkpoint dir)
                key.pt              (only when the pool saved a key)
                validation.json
                rejection.json
                manifest.json

The rejection is also recorded in the ExpertRegistry with lifecycle
status REJECTED (terminal): snapshots can then distinguish active experts
from rejected candidates, and a rejected expert id can never be committed
again (the registry rejects duplicate ids).
"""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from compose.experts.metadata import ExpertLifecycleStatus, ExpertMetadata

REJECTED_ROOT_NAME = "rejected_candidates"
REJECTION_REASON_BELOW_TAU = "below_tau"
REJECTION_STATUS = "rejected"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def rejected_candidate_dir(root, task_id: int, candidate_id: int) -> Path:
    """Artifact root for one rejected candidate (never part of the pool)."""
    return (
        Path(root)
        / REJECTED_ROOT_NAME
        / "task_{:02d}".format(int(task_id))
        / "candidate_{:02d}".format(int(candidate_id))
    )


def write_rejected_candidate(
    root,
    *,
    task_id: int,
    task_name: str,
    candidate_id: int,
    reason: str,
    mean_gain: float,
    support: int,
    validation_stats: Dict[str, Any],
    commit_thresholds: Dict[str, Any],
    adapter_dir,
    seed: Optional[int] = None,
    pool_version: Optional[int] = None,
    config_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist a rejected candidate's artifacts and return the rejection record.

    ``adapter_dir`` is the candidate's training-time pool checkpoint
    (``candidate/cold_start`` for task 0, ``candidate/train`` for later
    tasks); the full pool is copied so the raw candidate state dict, LoRA
    weights and manifest are all preserved for diagnostics.
    """
    candidate_id = int(candidate_id)
    task_id = int(task_id)
    source = Path(adapter_dir)
    if not (source / "compose_experts.json").is_file():
        raise FileNotFoundError(
            "rejected candidate adapter dir lacks compose_experts.json: {}".format(source)
        )
    dest = rejected_candidate_dir(root, task_id, candidate_id)
    adapter_dest = dest / "adapter"
    if adapter_dest.exists():
        shutil.rmtree(str(adapter_dest))
    shutil.copytree(str(source), str(adapter_dest))

    checkpoint_path = adapter_dest / "compose_experts.bin"
    if checkpoint_path.is_file():
        checkpoint_sha256 = _sha256_file(str(checkpoint_path))
    else:
        checkpoint_sha256 = ""

    rejection = {
        "task_id": task_id,
        "candidate_id": candidate_id,
        "status": REJECTION_STATUS,
        "reason": reason,
        "mean_gain": float(mean_gain),
        "support": int(support),
        "commit_thresholds": dict(commit_thresholds),
        "checkpoint_sha256": checkpoint_sha256,
        "excluded_from_active_pool": True,
        "checkpoint_path": str(checkpoint_path),
    }
    _write_json(dest / "rejection.json", rejection)
    _write_json(dest / "validation.json", dict(validation_stats))
    _write_json(
        dest / "manifest.json",
        {
            "task_id": task_id,
            "task_name": str(task_name),
            "candidate_id": candidate_id,
            "seed": seed,
            "pool_version": pool_version,
            "config_hash": config_hash,
            "schema": "v6_rejected_candidate_v1",
            "note": "diagnostics only; never loaded for formal inference, "
            "teacher search, Router or evaluation",
        },
    )
    return rejection


def register_rejected_candidate(
    registry,
    *,
    expert_id: int,
    task_id: int,
    task_name: str,
    seed: Optional[int],
    reason: str,
    mean_gain: float,
    support: int,
    checkpoint_path,
    checkpoint_sha256: str,
    commit_thresholds: Dict[str, Any],
) -> None:
    """Record the rejection in the registry (terminal REJECTED status).

    The expert id is never reused (the registry refuses duplicate ids), the
    rejection never enters the active pool, never bumps pool_version and is
    excluded from ``get_active_experts()``.
    """
    expert_id = int(expert_id)
    if registry.contains(expert_id):
        raise ValueError(
            "expert {} already exists in the registry; a rejected candidate "
            "cannot overwrite it".format(expert_id)
        )
    metadata = ExpertMetadata(
        expert_id=expert_id,
        adapter_name="expert_{:04d}".format(expert_id),
        rank=8,
        alpha=16.0,
        creation_task=int(task_id),
        creation_task_name=str(task_name),
        created_seed=int(seed) if seed is not None else None,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=str(checkpoint_sha256),
        lifecycle_status=ExpertLifecycleStatus.CANDIDATE,
        support_count=int(support),
        mean_conditional_gain=float(mean_gain),
    )
    registry.register(metadata)
    registry.mark_rejected(
        expert_id,
        {
            "task_id": int(task_id),
            "reason": str(reason),
            "mean_gain": float(mean_gain),
            "support": int(support),
            "commit_thresholds": dict(commit_thresholds),
        },
    )
