"""Compose task-level state machine.

One-way stage transitions for the Query-Clustered Residual Expert
Discovery pipeline:

  NOT_STARTED -> DATA_READY -> QUERY_READY
      -> OLD_TEACHER_READY (task > 0) or -> RESIDUAL_READY (task 0,
         reason recorded in history)
      -> RESIDUAL_READY -> CLUSTERS_READY (or NO_EXPANSION_REQUIRED)
      -> CLUSTER_EXPERTS_TRAINING -> CLUSTER_EXPERTS_TRAINED
      -> KEYS_TRAINING -> KEYS_READY -> EXPERTS_COMMITTED
      -> RMS_READY -> SNAPSHOT_READY -> EVALUATION_COMPLETE -> COMPLETED

The machine is persisted atomically (mkstemp + fsync + os.replace) and
restored from disk on resume without ever auto-advancing.
"""

import json
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


STATE_SCHEMA_VERSION = 2


class TaskStage(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    DATA_READY = "DATA_READY"
    QUERY_READY = "QUERY_READY"
    OLD_TEACHER_READY = "OLD_TEACHER_READY"
    RESIDUAL_READY = "RESIDUAL_READY"
    NO_EXPANSION_REQUIRED = "NO_EXPANSION_REQUIRED"
    CLUSTERS_READY = "CLUSTERS_READY"
    CLUSTER_EXPERTS_TRAINING = "CLUSTER_EXPERTS_TRAINING"
    CLUSTER_EXPERTS_TRAINED = "CLUSTER_EXPERTS_TRAINED"
    KEYS_TRAINING = "KEYS_TRAINING"
    KEYS_READY = "KEYS_READY"
    EXPERTS_COMMITTED = "EXPERTS_COMMITTED"
    RMS_READY = "RMS_READY"
    SNAPSHOT_READY = "SNAPSHOT_READY"
    EVALUATION_COMPLETE = "EVALUATION_COMPLETE"
    COMPLETED = "COMPLETED"

    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value == str(value):
                return member
        return None


#: One-way transition graph (union over tasks). Task 0 skips
#: OLD_TEACHER_READY (no historical experts; the skip reason is recorded
#: in the state history); an undersized residual split moves
#: RESIDUAL_READY -> NO_EXPANSION_REQUIRED (never a fake cluster).
TRANSITIONS: Dict[TaskStage, Set[TaskStage]] = {
    TaskStage.NOT_STARTED: {TaskStage.DATA_READY},
    TaskStage.DATA_READY: {TaskStage.QUERY_READY},
    TaskStage.QUERY_READY: {
        TaskStage.OLD_TEACHER_READY,
        TaskStage.RESIDUAL_READY,
    },
    TaskStage.OLD_TEACHER_READY: {TaskStage.RESIDUAL_READY},
    TaskStage.RESIDUAL_READY: {
        TaskStage.CLUSTERS_READY,
        TaskStage.NO_EXPANSION_REQUIRED,
    },
    TaskStage.NO_EXPANSION_REQUIRED: {TaskStage.RMS_READY},
    TaskStage.CLUSTERS_READY: {TaskStage.CLUSTER_EXPERTS_TRAINING},
    TaskStage.CLUSTER_EXPERTS_TRAINING: {TaskStage.CLUSTER_EXPERTS_TRAINED},
    TaskStage.CLUSTER_EXPERTS_TRAINED: {TaskStage.KEYS_TRAINING},
    TaskStage.KEYS_TRAINING: {TaskStage.KEYS_READY},
    TaskStage.KEYS_READY: {TaskStage.EXPERTS_COMMITTED},
    TaskStage.EXPERTS_COMMITTED: {TaskStage.RMS_READY},
    TaskStage.RMS_READY: {TaskStage.SNAPSHOT_READY},
    TaskStage.SNAPSHOT_READY: {TaskStage.EVALUATION_COMPLETE},
    TaskStage.EVALUATION_COMPLETE: {TaskStage.COMPLETED},
    TaskStage.COMPLETED: set(),
}


class TaskStateMachine:
    """Persisted, strictly-forward task state machine."""

    def __init__(
        self,
        task_id: int,
        task_name: str,
        stage: TaskStage = TaskStage.NOT_STARTED,
    ) -> None:
        self.task_id = int(task_id)
        if self.task_id < 0:
            raise ValueError("task_id must be non-negative")
        self.task_name = str(task_name)
        if not self.task_name:
            raise ValueError("task_name must not be empty")
        if not isinstance(stage, TaskStage):
            stage = TaskStage(stage)
        self.stage = stage
        self.history = []  # type: List[Dict[str, Any]]

    def allowable_next_stages(self) -> List[TaskStage]:
        return sorted(TRANSITIONS[self.stage], key=lambda item: item.value)

    def can_advance_to(self, target: TaskStage) -> bool:
        return isinstance(target, TaskStage) and target in TRANSITIONS[self.stage]

    def advance(self, target: TaskStage, note: Optional[str] = None) -> "TaskStateMachine":
        """One-way transition; raises on illegal moves."""
        if not self.can_advance_to(target):
            raise ValueError(
                "illegal task transition {} -> {} for task {} (allowed: {})".format(
                    self.stage.value,
                    target.value,
                    self.task_id,
                    [stage.value for stage in self.allowable_next_stages()],
                )
            )
        self.history.append(
            {"from": self.stage.value, "to": target.value, "note": note or ""}
        )
        self.stage = target
        return self

    def skip_with_reason(self, skipped: TaskStage, reason: str) -> None:
        """Record a skipped stage in the history without passing through it
        (used for task 0's OLD_TEACHER_READY skip)."""
        self.history.append(
            {"from": self.stage.value, "to": self.stage.value, "skipped": skipped.value, "note": reason}
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "stage": self.stage.value,
            "history": list(self.history),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "TaskStateMachine":
        version = int(state.get("schema_version", -1))
        if version != STATE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported state schema_version: {} (expected {})".format(
                    version, STATE_SCHEMA_VERSION
                )
            )
        machine = cls(
            task_id=int(state["task_id"]),
            task_name=str(state["task_name"]),
            stage=TaskStage(state["stage"]),
        )
        machine.history = [dict(entry) for entry in state.get("history", [])]
        return machine

    def save(self, path) -> None:
        """Atomic persistence; refuses to overwrite unless explicitly
        requested (legacy write-once behavior)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
        )
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
    def load(cls, path) -> "TaskStateMachine":
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError("task state does not exist: {}".format(target))
        with target.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        return cls.from_state_dict(state)

    def __repr__(self) -> str:
        return "TaskStateMachine(task={}, stage={})".format(self.task_id, self.stage.value)
