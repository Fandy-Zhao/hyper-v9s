"""Two-phase commit transactions for expert submission (V6 Stage E2).

Commit protocol (mirrors the task book):
  1. mark pending (intent marker, atomic)
  2. write LoRA checkpoint
  3. write expert key
  4. write validation report
  5. compute hashes
  6. atomically update the registry (provisional)
  7. bump pool_version
  8. remove the pending marker

Crash semantics:
- crash between 1 and 6: the pending marker exists but the registry has no
  entry for the expert id -> the commit never landed; roll back the marker
  and the half-written artifacts (no half-committed expert).
- crash after 6: the registry already contains the expert; re-running
  ``complete`` detects the registry entry and finishes idempotently without
  bumping pool_version again.
- expert IDs are never reused: the registry rejects duplicate ids, and the
  pending marker is keyed by expert id, so a crash cannot produce two
  experts with the same id.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .metadata import ExpertLifecycleStatus, ExpertMetadata
from .registry import ExpertRegistry


PENDING_PREFIX = "pending_expert_"
PENDING_SUFFIX = ".json"
REGISTRY_FILENAME = "expert_registry.json"
TRANSACTION_VERSION = 1


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(target: Path, payload: Dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class CommitTransaction:
    """Transaction coordinator around one expert commit."""

    def __init__(self, registry_dir, registry: ExpertRegistry) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry = registry

    # ------------------------------------------------------------------
    # Artifact paths
    # ------------------------------------------------------------------

    def pending_path(self, expert_id: int) -> Path:
        return self.registry_dir / (
            PENDING_PREFIX + str(int(expert_id)) + PENDING_SUFFIX
        )

    def registry_path(self) -> Path:
        return self.registry_dir / REGISTRY_FILENAME

    # ------------------------------------------------------------------
    # Phase 1: intent
    # ------------------------------------------------------------------

    def begin(self, expert_id: int, intent: Dict[str, Any]) -> None:
        """Write the pending intent marker atomically.

        A pending marker for the same expert id without a registry entry
        means a previous commit attempt never landed; overwriting the marker
        is allowed (the artifacts are keyed by expert id and rewritten by
        the caller), but the expert id must not yet exist in the registry.
        """
        expert_id = int(expert_id)
        if self.registry.contains(expert_id):
            raise ValueError(
                "expert {} already exists in the registry; refusing to reuse"
                " its id".format(expert_id)
            )
        payload = {
            "transaction_version": TRANSACTION_VERSION,
            "expert_id": expert_id,
            "phase": "pending",
            "intent": intent,
        }
        _atomic_write_json(self.pending_path(expert_id), payload)

    # ------------------------------------------------------------------
    # Phase 2: verify artifacts, land the commit
    # ------------------------------------------------------------------

    def complete(
        self,
        expert_id: int,
        artifacts: Dict[str, str],
        condition_record: Dict[str, Any],
        metadata: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Land the commit.

        ``artifacts`` maps artifact paths (LoRA checkpoint, key, validation
        report, ...) to their expected sha256. All must exist and match.

        The expert enters the registry only inside ``complete``: if it is
        not registered yet (the usual flow, since ``begin`` refuses ids that
        already exist), ``metadata`` is used to register it. It is then
        marked provisional, pool_version is bumped exactly once, the
        registry is persisted atomically and the pending marker is removed.

        Idempotent: if the expert is already provisional in the registry
        (crash after the registry update), returns without bumping
        pool_version again. An existing non-provisional entry is a state
        conflict and raises.
        """
        expert_id = int(expert_id)
        if self.registry.contains(expert_id):
            existing = self.registry.get(expert_id)
            if existing.lifecycle_status is ExpertLifecycleStatus.PROVISIONAL:
                # Crash happened after the registry update: the commit
                # already landed. Finish idempotently (no version bump).
                marker = self.pending_path(expert_id)
                if marker.exists():
                    marker.unlink()
                return {"status": "already_committed", "expert_id": expert_id}
            raise ValueError(
                "expert {} exists with lifecycle {}; refusing to commit over it".format(
                    expert_id, existing.lifecycle_status.value
                )
            )

        verified = {}
        for path, expected_hash in artifacts.items():
            artifact_path = Path(path)
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    "commit artifact missing: {}".format(artifact_path)
                )
            actual_hash = file_sha256(artifact_path)
            if actual_hash != expected_hash:
                raise ValueError(
                    "commit artifact hash mismatch for {}: expected {}, got {}".format(
                        artifact_path, expected_hash, actual_hash
                    )
                )
            verified[str(artifact_path)] = actual_hash

        if metadata is None:
            raise ValueError(
                "metadata is required to register expert {} at commit time".format(
                    expert_id
                )
            )
        self.registry.register(metadata)

        # Atomic registry update: provisional + pool_version bump, then
        # persist, then clear the pending marker (last step => a crash at
        # any earlier point leaves the marker for rollback).
        self.registry.mark_provisional(expert_id, condition_record)
        self.registry.increment_pool_version()
        self.registry.save_atomic(self.registry_path(), allow_overwrite=True)
        marker = self.pending_path(expert_id)
        if marker.exists():
            marker.unlink()
        return {
            "status": "committed",
            "expert_id": expert_id,
            "pool_version": self.registry.pool_version,
            "verified_artifacts": verified,
        }

    def abort(self, expert_id: int) -> None:
        """Remove the pending marker; half-written artifacts are left for
        the caller to clean (their hashes never enter the registry)."""
        marker = self.pending_path(expert_id)
        if marker.exists():
            marker.unlink()

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    @classmethod
    def pending_intents(cls, registry_dir) -> Dict[int, Dict[str, Any]]:
        """Return all pending markers keyed by expert id."""
        directory = Path(registry_dir)
        intents = {}
        if not directory.is_dir():
            return intents
        for path in sorted(directory.glob(PENDING_PREFIX + "*" + PENDING_SUFFIX)):
            expert_id = int(
                path.name[len(PENDING_PREFIX): -len(PENDING_SUFFIX)]
            )
            with path.open("r", encoding="utf-8") as handle:
                intents[expert_id] = json.load(handle)
        return intents

    @classmethod
    def resume(cls, registry_dir, registry: Optional[ExpertRegistry] = None):
        """Recover from a crash.

        Returns ``(registry, incomplete, completed)`` where:
        - ``incomplete``: pending markers whose expert is missing from the
          registry (commit never landed; must roll back),
        - ``completed``: pending markers whose expert is already in the
          registry (commit landed; markers are cleaned here).
        """
        registry_path = Path(registry_dir) / REGISTRY_FILENAME
        if registry is None:
            registry = (
                ExpertRegistry.load_json(registry_path)
                if registry_path.is_file()
                else ExpertRegistry()
            )
        intents = cls.pending_intents(registry_dir)
        incomplete = {}
        completed = {}
        for expert_id, intent in intents.items():
            if registry.contains(expert_id):
                completed[expert_id] = intent
                marker = Path(registry_dir) / (
                    PENDING_PREFIX + str(expert_id) + PENDING_SUFFIX
                )
                marker.unlink(missing_ok=True)
            else:
                incomplete[expert_id] = intent
        return registry, incomplete, completed
