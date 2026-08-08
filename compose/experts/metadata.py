"""Versioned, JSON-safe metadata for Compose experts.

The legacy ``ExpertPool`` fields remain accepted so checkpoints produced by
the Compose foundation continue to load.  Compose code should use the canonical
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


class ExpertLifecycleStatus(str, Enum):
    """Commit lifecycle of an expert (Compose Stage E2).

    Distinct from :class:`ExpertStatus` (training role): lifecycle tracks the
    submission state machine ``candidate -> provisional -> formal -> archived``
    while ``ExpertStatus`` tracks frozen/trainable roles during execution.
    """

    CANDIDATE = "candidate"
    PROVISIONAL = "provisional"
    FORMAL = "formal"
    ARCHIVED = "archived"
    REJECTED = "rejected"

    @classmethod
    def _missing_(cls, value):
        if value in ("pending", "committed"):
            # Legacy internal spellings read back as the canonical status.
            return cls.PROVISIONAL
        return None


#: The only lifecycle statuses that make an expert part of the formal model
#: (teacher search, Router candidate set, inference selection, composition,
#: RMS, anchor generation, cross-task reuse statistics, task-boundary eval).
#: ``candidate`` / ``rejected`` / ``archived`` never participate in formal
#: behavior; a checkpoint existing on disk never implies formal availability
#: (empty-registry fix, Stage R1).
ACTIVE_LIFECYCLE_STATUSES = frozenset(
    {ExpertLifecycleStatus.PROVISIONAL, ExpertLifecycleStatus.FORMAL}
)


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

    # Compose Stage E2 canonical lifecycle fields (all optional for
    # backward-compatible loads; defaults are inferred in __post_init__).
    lifecycle_status: Optional[ExpertLifecycleStatus] = None
    created_seed: Optional[int] = None
    key_path: Optional[str] = None
    key_sha256: Optional[str] = None
    rms_stats_path: Optional[str] = None
    mean_conditional_gain: float = 0.0
    key_accuracy: float = 0.0
    parent_contexts: List[Dict[str, Any]] = field(default_factory=list)
    config_hash: Optional[str] = None
    pool_version_created: Optional[int] = None
    target_modules: List[str] = field(default_factory=list)
    lora_alpha: Optional[float] = None  # canonical alias of ``alpha``
    created_task_id: Optional[int] = None  # canonical alias of ``creation_task``

    # Backward-compatible fields used by the legacy ExpertPool/checkpoints.
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
        if self.created_seed is not None:
            self.created_seed = int(self.created_seed)
        if self.pool_version_created is not None:
            self.pool_version_created = int(self.pool_version_created)

        if self.creation_task_name is None and self.origin_task_id is not None:
            self.creation_task_name = self.origin_task_id
        if self.origin_task_id is None and self.creation_task_name is not None:
            self.origin_task_id = self.creation_task_name
        if self.checkpoint_path is None and self.source_checkpoint is not None:
            self.checkpoint_path = self.source_checkpoint
        if self.source_checkpoint is None and self.checkpoint_path is not None:
            self.source_checkpoint = self.checkpoint_path

        # Canonical alias synchronization (task book field names).
        if self.lora_alpha is not None:
            self.alpha = float(self.lora_alpha)
        elif self.alpha is not None:
            self.lora_alpha = float(self.alpha)
        if self.created_task_id is not None:
            self.creation_task = int(self.created_task_id)
        if self.creation_task is not None and self.created_task_id is None:
            self.created_task_id = self.creation_task

        self.mean_conditional_gain = float(self.mean_conditional_gain)
        self.key_accuracy = float(self.key_accuracy)
        self.parent_contexts = [dict(value) for value in self.parent_contexts]
        self.target_modules = list(
            dict.fromkeys(str(value) for value in self.target_modules)
        )

        if self.lifecycle_status is None:
            # Infer from persistence: an expert with a written checkpoint was
            # committed (provisional); a bare registration is still a candidate.
            self.lifecycle_status = (
                ExpertLifecycleStatus.PROVISIONAL
                if (self.checkpoint_path or self.source_checkpoint)
                else ExpertLifecycleStatus.CANDIDATE
            )
        elif not isinstance(self.lifecycle_status, ExpertLifecycleStatus):
            self.lifecycle_status = ExpertLifecycleStatus(self.lifecycle_status)

        if self.status is ExpertStatus.ARCHIVED and (self.active or self.trainable):
            raise ValueError("archived experts cannot be active or trainable")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        if self.lifecycle_status is not None:
            data["lifecycle_status"] = self.lifecycle_status.value
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
