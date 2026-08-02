"""Versioned, JSON-safe metadata for Compose experts.

The legacy ``ExpertPool`` fields remain accepted so checkpoints produced by
the Compose foundation continue to load.  V6 code should use the canonical
fields documented by :class:`ExpertMetadata`.
"""

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Optional


METADATA_VERSION = 1


class ExpertStatus(str, Enum):
    REGISTERED = "registered"
    FROZEN = "frozen"
    TRAINABLE = "trainable"
    TRAINING = "trainable"  # Legacy API alias.
    ARCHIVED = "archived"

    @classmethod
    def _missing_(cls, value):
        # Stage-00/01 manifests used ``training``.  Read them as the canonical
        # Stage-02 status without writing the deprecated spelling again.
        if value == "training":
            return cls.TRAINABLE
        return None


@dataclass
class ExpertMetadata:
    expert_id: int
    adapter_name: str = ""
    rank: int = 1
    alpha: float = 1.0
    status: ExpertStatus = ExpertStatus.REGISTERED
    creation_task: Optional[int] = None
    creation_task_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    trainable: bool = False
    active: bool = False
    support_count: int = 0
    reuse_tasks: List[int] = field(default_factory=list)
    positive_contribution_count: int = 0
    parent_expert_id: Optional[int] = None
    metadata_version: int = METADATA_VERSION
    extra: Dict[str, Any] = field(default_factory=dict)

    # Backward-compatible fields used by the pre-V6 ExpertPool/checkpoints.
    name: Optional[str] = None
    origin_task_id: Optional[str] = None
    source_checkpoint: Optional[str] = None
    trained_steps: int = 0
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.expert_id = int(self.expert_id)
        if self.expert_id < 0:
            raise ValueError("expert_id must be non-negative")

        self.adapter_name = self.adapter_name or self.name or ""
        if not self.adapter_name:
            raise ValueError("adapter_name must not be empty")
        if self.name is None:
            self.name = self.adapter_name

        self.rank = int(self.rank)
        self.alpha = float(self.alpha)
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

        if not isinstance(self.status, ExpertStatus):
            self.status = ExpertStatus(self.status)
        self.support_count = int(self.support_count)
        self.positive_contribution_count = int(self.positive_contribution_count)
        self.trained_steps = int(self.trained_steps)
        if self.support_count < 0:
            raise ValueError("support_count must be non-negative")
        if self.positive_contribution_count < 0:
            raise ValueError("positive_contribution_count must be non-negative")
        if self.trained_steps < 0:
            raise ValueError("trained_steps must be non-negative")
        if self.metadata_version <= 0:
            raise ValueError("metadata_version must be positive")

        self.reuse_tasks = list(dict.fromkeys(int(value) for value in self.reuse_tasks))
        self.tags = list(dict.fromkeys(str(value) for value in self.tags))
        if self.creation_task is not None:
            self.creation_task = int(self.creation_task)
        if self.parent_expert_id is not None:
            self.parent_expert_id = int(self.parent_expert_id)

        if self.creation_task_name is None and self.origin_task_id is not None:
            self.creation_task_name = self.origin_task_id
        if self.origin_task_id is None and self.creation_task_name is not None:
            self.origin_task_id = self.creation_task_name
        if self.checkpoint_path is None and self.source_checkpoint is not None:
            self.checkpoint_path = self.source_checkpoint
        if self.source_checkpoint is None and self.checkpoint_path is not None:
            self.source_checkpoint = self.checkpoint_path

        if self.status is ExpertStatus.ARCHIVED and (self.active or self.trainable):
            raise ValueError("archived experts cannot be active or trainable")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpertMetadata":
        if not isinstance(data, dict):
            raise TypeError("expert metadata must be a dictionary")
        known = {item.name for item in fields(cls)}
        values = {key: value for key, value in data.items() if key in known}
        unknown = {key: value for key, value in data.items() if key not in known}
        extra = dict(values.get("extra") or {})
        # Unknown fields are preserved instead of silently discarded.  This
        # gives forward-compatible round trips while keeping the schema strict.
        extra.update(unknown)
        values["extra"] = extra
        values["status"] = ExpertStatus(values.get("status", "registered"))
        if not values.get("adapter_name"):
            values["adapter_name"] = values.get("name") or ""
        return cls(**values)
