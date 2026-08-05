"""Ordered metadata registry for V6 experts.

The registry intentionally contains no model tensors and does not infer expert
selection from task IDs.  Execution is delegated to the Compose-to-Hyper bridge.
"""

import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .metadata import (
    ExpertLifecycleStatus,
    ExpertMetadata,
    ExpertStatus,
    METADATA_VERSION,
)


REGISTRY_VERSION = 1
POOL_VERSION_INITIAL = 1


def _ordered_unique(values: Iterable[int]) -> Tuple[int, ...]:
    return tuple(dict.fromkeys(int(value) for value in values))


class ExpertRegistry:
    def __init__(self) -> None:
        self._experts = OrderedDict()  # type: OrderedDict[int, ExpertMetadata]
        self._active_ids = ()  # type: Tuple[int, ...]
        self._trainable_ids = ()  # type: Tuple[int, ...]
        self._pool_version = POOL_VERSION_INITIAL

    @property
    def pool_version(self) -> int:
        """Monotonic pool version; incremented only by commit transactions."""
        return self._pool_version

    def increment_pool_version(self) -> int:
        """Advance the pool version. Called exclusively by commit transactions
        (``CommitTransaction.complete``); recovery must never bump it."""
        self._pool_version += 1
        return self._pool_version

    @property
    def active_expert_ids(self) -> Tuple[int, ...]:
        return self._active_ids

    @property
    def trainable_expert_ids(self) -> Tuple[int, ...]:
        return self._trainable_ids

    def register(self, metadata: ExpertMetadata) -> ExpertMetadata:
        if not isinstance(metadata, ExpertMetadata):
            raise TypeError("metadata must be ExpertMetadata")
        if metadata.expert_id in self._experts:
            raise ValueError("expert {} is already registered".format(metadata.expert_id))
        self._experts[metadata.expert_id] = metadata
        if metadata.active:
            self.set_active_ids(self._active_ids + (metadata.expert_id,))
        if metadata.trainable:
            self.set_trainable_ids(self._trainable_ids + (metadata.expert_id,))
        self.validate()
        return metadata

    def unregister(self, expert_id: int) -> ExpertMetadata:
        metadata = self.get(expert_id)
        if metadata.checkpoint_path or metadata.source_checkpoint:
            raise ValueError("cannot unregister an expert bound to a checkpoint")
        self._active_ids = tuple(value for value in self._active_ids if value != metadata.expert_id)
        self._trainable_ids = tuple(value for value in self._trainable_ids if value != metadata.expert_id)
        return self._experts.pop(metadata.expert_id)

    def get(self, expert_id: int) -> ExpertMetadata:
        expert_id = int(expert_id)
        try:
            return self._experts[expert_id]
        except KeyError:
            raise KeyError("expert {} is not registered".format(expert_id))

    def contains(self, expert_id: int) -> bool:
        return int(expert_id) in self._experts

    def list_all(self) -> List[ExpertMetadata]:
        return list(self._experts.values())

    def list_active(self) -> List[ExpertMetadata]:
        return [self._experts[value] for value in self._active_ids]

    def list_trainable(self) -> List[ExpertMetadata]:
        return [self._experts[value] for value in self._trainable_ids]

    def list_archived(self) -> List[ExpertMetadata]:
        return [item for item in self._experts.values() if item.status is ExpertStatus.ARCHIVED]

    def _require_selectable(self, values: Iterable[int], role: str) -> Tuple[int, ...]:
        ordered = _ordered_unique(values)
        for expert_id in ordered:
            metadata = self.get(expert_id)
            if metadata.status is ExpertStatus.ARCHIVED:
                raise ValueError("archived expert {} cannot be {}".format(expert_id, role))
        return ordered

    def set_active_ids(self, values: Iterable[int]) -> None:
        ordered = self._require_selectable(values, "active")
        self._active_ids = ordered
        selected = set(ordered)
        for expert_id, metadata in self._experts.items():
            metadata.active = expert_id in selected

    def set_trainable_ids(self, values: Iterable[int]) -> None:
        ordered = self._require_selectable(values, "trainable")
        self._trainable_ids = ordered
        selected = set(ordered)
        for expert_id, metadata in self._experts.items():
            metadata.trainable = expert_id in selected
            if metadata.status is not ExpertStatus.ARCHIVED:
                metadata.status = ExpertStatus.TRAINABLE if metadata.trainable else (
                    ExpertStatus.FROZEN if metadata.status is ExpertStatus.TRAINABLE else metadata.status
                )

    def freeze(self, expert_id: int) -> None:
        metadata = self.get(expert_id)
        if metadata.status is ExpertStatus.ARCHIVED:
            raise ValueError("cannot freeze archived expert {}".format(expert_id))
        metadata.status = ExpertStatus.FROZEN
        self.set_trainable_ids(value for value in self._trainable_ids if value != metadata.expert_id)

    def archive(self, expert_id: int) -> None:
        metadata = self.get(expert_id)
        metadata.status = ExpertStatus.ARCHIVED
        self.set_active_ids(value for value in self._active_ids if value != metadata.expert_id)
        self.set_trainable_ids(value for value in self._trainable_ids if value != metadata.expert_id)
        metadata.active = False
        metadata.trainable = False

    # ------------------------------------------------------------------
    # V6 Stage E2 lifecycle transitions (candidate -> provisional ->
    # formal -> archived). Transitions mutate in memory; the caller
    # persists the registry atomically (save_atomic).
    # ------------------------------------------------------------------

    _LIFECYCLE_TRANSITIONS = {
        ExpertLifecycleStatus.CANDIDATE: {ExpertLifecycleStatus.PROVISIONAL},
        ExpertLifecycleStatus.PROVISIONAL: {ExpertLifecycleStatus.FORMAL, ExpertLifecycleStatus.ARCHIVED},
        ExpertLifecycleStatus.FORMAL: {ExpertLifecycleStatus.ARCHIVED},
        ExpertLifecycleStatus.REJECTED: set(),
        ExpertLifecycleStatus.ARCHIVED: set(),
    }

    def _set_lifecycle(self, expert_id: int, target: ExpertLifecycleStatus,
                       condition_record: Dict[str, Any]) -> ExpertLifecycleStatus:
        metadata = self.get(expert_id)
        current = metadata.lifecycle_status
        allowed = self._LIFECYCLE_TRANSITIONS.get(current, set())
        if target not in allowed:
            raise ValueError(
                "illegal lifecycle transition {} -> {} for expert {}".format(
                    current.value if current else None, target.value, expert_id
                )
            )
        metadata.lifecycle_status = target
        # Record the promotion/archival condition for auditability.
        conditions = dict(metadata.extra.get("lifecycle_conditions") or {})
        conditions[target.value] = condition_record
        metadata.extra["lifecycle_conditions"] = conditions
        return target

    def mark_provisional(self, expert_id: int, condition_record: Dict[str, Any]) -> None:
        """Commit-time transition: a validated candidate becomes provisional.

        ``condition_record`` must capture the validation metrics (support
        count, mean gain, key accuracy) that justified the commit.
        """
        self._set_lifecycle(expert_id, ExpertLifecycleStatus.PROVISIONAL, condition_record)

    def mark_formal(self, expert_id: int, condition_record: Dict[str, Any]) -> None:
        """Promotion interface: provisional -> formal under recorded conditions."""
        self._set_lifecycle(expert_id, ExpertLifecycleStatus.FORMAL, condition_record)

    def mark_rejected(self, expert_id: int, condition_record: Dict[str, Any]) -> None:
        """Mark a candidate that failed validation as rejected (terminal)."""
        metadata = self.get(expert_id)
        if metadata.lifecycle_status is not ExpertLifecycleStatus.CANDIDATE:
            raise ValueError(
                "only candidates can be rejected; expert {} is {}".format(
                    expert_id, metadata.lifecycle_status.value
                )
            )
        metadata.lifecycle_status = ExpertLifecycleStatus.REJECTED
        conditions = dict(metadata.extra.get("lifecycle_conditions") or {})
        conditions["rejected"] = condition_record
        metadata.extra["lifecycle_conditions"] = conditions

    def validate(self) -> None:
        if len(self._experts) != len(set(self._experts)):
            raise ValueError("expert IDs must be unique")
        active = set(self._active_ids)
        trainable = set(self._trainable_ids)
        missing = sorted((active | trainable) - set(self._experts))
        if missing:
            raise ValueError("registry selections contain unregistered experts: {}".format(missing))
        for expert_id, metadata in self._experts.items():
            if metadata.expert_id != expert_id:
                raise ValueError("registry key and metadata expert_id disagree")
            if metadata.active != (expert_id in active):
                raise ValueError("active flag mismatch for expert {}".format(expert_id))
            if metadata.trainable != (expert_id in trainable):
                raise ValueError("trainable flag mismatch for expert {}".format(expert_id))
            if metadata.status is ExpertStatus.ARCHIVED and (metadata.active or metadata.trainable):
                raise ValueError("archived expert {} is selected".format(expert_id))

    def state_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "metadata_version": METADATA_VERSION,
            "registry_version": REGISTRY_VERSION,
            "pool_version": self._pool_version,
            "experts": [metadata.to_dict() for metadata in self._experts.values()],
            "active_expert_ids": list(self._active_ids),
            "trainable_expert_ids": list(self._trainable_ids),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("registry state must be a dictionary")
        if int(state.get("registry_version", -1)) != REGISTRY_VERSION:
            raise ValueError("unsupported registry_version: {}".format(state.get("registry_version")))
        entries = state.get("experts")
        if not isinstance(entries, list):
            raise ValueError("registry state requires an experts list")
        replacement = OrderedDict()
        for entry in entries:
            metadata = ExpertMetadata.from_dict(entry)
            if metadata.expert_id in replacement:
                raise ValueError("duplicate expert_id {} in registry state".format(metadata.expert_id))
            metadata.active = False
            metadata.trainable = False
            replacement[metadata.expert_id] = metadata
        self._experts = replacement
        self._active_ids = ()
        self._trainable_ids = ()
        # Pool version is restored as-is; recovery never re-increments it.
        restored_pool_version = int(state.get("pool_version", POOL_VERSION_INITIAL))
        if restored_pool_version < POOL_VERSION_INITIAL:
            raise ValueError("pool_version must be at least {}".format(POOL_VERSION_INITIAL))
        self._pool_version = restored_pool_version
        self.set_active_ids(state.get("active_expert_ids", []))
        self.set_trainable_ids(state.get("trainable_expert_ids", []))
        self.validate()

    def save_json(self, path) -> None:
        self.save_atomic(path, allow_overwrite=False)

    def save_atomic(self, path, allow_overwrite: bool = False) -> None:
        """Atomically persist the registry (mkstemp + fsync + os.replace).

        ``allow_overwrite=False`` keeps the original first-write guard
        (registry checkpoint files are normally write-once); commit
        transactions pass ``allow_overwrite=True`` for the final registry
        update of a task.
        """
        target = Path(path)
        if target.exists() and not allow_overwrite:
            raise FileExistsError("registry checkpoint already exists: {}".format(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self.state_dict(), handle, indent=2, sort_keys=True)
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

    @classmethod
    def load_json(cls, path) -> "ExpertRegistry":
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError("registry checkpoint does not exist: {}".format(target))
        with target.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        registry = cls()
        registry.load_state_dict(state)
        return registry
