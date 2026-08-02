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

from .metadata import ExpertMetadata, ExpertStatus, METADATA_VERSION


REGISTRY_VERSION = 1


def _ordered_unique(values: Iterable[int]) -> Tuple[int, ...]:
    return tuple(dict.fromkeys(int(value) for value in values))


class ExpertRegistry:
    def __init__(self) -> None:
        self._experts = OrderedDict()  # type: OrderedDict[int, ExpertMetadata]
        self._active_ids = ()  # type: Tuple[int, ...]
        self._trainable_ids = ()  # type: Tuple[int, ...]

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
        self.set_active_ids(state.get("active_expert_ids", []))
        self.set_trainable_ids(state.get("trainable_expert_ids", []))
        self.validate()

    def save_json(self, path) -> None:
        target = Path(path)
        if target.exists():
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
