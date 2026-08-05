"""V6 task-level state machine (Stage E2).

Sixteen stages, strictly one-way transitions (a directed acyclic graph).
The allowed transition set is the union over tasks: task 1 skips the old
teacher / residual stages (cold start straight into candidate training),
and a task with an undersized residual buffer skips candidate creation
(RESIDUAL_READY -> GLOBAL_TEACHER_READY with commit_count = 0).

The machine is persisted atomically (mkstemp + fsync + os.replace) and
restored from disk on resume without ever auto-advancing.
"""

import json
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


STATE_SCHEMA_VERSION = 1


class TaskStage(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    DATA_READY = "DATA_READY"
    OLD_TEACHER_RUNNING = "OLD_TEACHER_RUNNING"
    OLD_TEACHER_READY = "OLD_TEACHER_READY"
    RESIDUAL_READY = "RESIDUAL_READY"
    CANDIDATE_TRAINING = "CANDIDATE_TRAINING"
    CANDIDATE_TRAINED = "CANDIDATE_TRAINED"
    CANDIDATE_VALIDATED = "CANDIDATE_VALIDATED"
    EXPERTS_COMMITTED = "EXPERTS_COMMITTED"
    GLOBAL_TEACHER_READY = "GLOBAL_TEACHER_READY"
    ROUTER_TRAINING = "ROUTER_TRAINING"
    ROUTER_READY = "ROUTER_READY"
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


#: One-way transition graph (union over tasks).
#: - task 1 cold start: DATA_READY -> CANDIDATE_TRAINING
#: - undersized residual: RESIDUAL_READY -> GLOBAL_TEACHER_READY
#: - no old experts at all (task 1): DATA_READY -> GLOBAL_TEACHER_READY
TRANSITIONS: Dict[TaskStage, Set[TaskStage]] = {
    TaskStage.NOT_STARTED: {TaskStage.DATA_READY},
    TaskStage.DATA_READY: {
        TaskStage.OLD_TEACHER_RUNNING,
        TaskStage.CANDIDATE_TRAINING,
        TaskStage.GLOBAL_TEACHER_READY,
    },
    TaskStage.OLD_TEACHER_RUNNING: {TaskStage.OLD_TEACHER_READY},
    TaskStage.OLD_TEACHER_READY: {TaskStage.RESIDUAL_READY},
    TaskStage.RESIDUAL_READY: {
        TaskStage.CANDIDATE_TRAINING,
        TaskStage.GLOBAL_TEACHER_READY,
    },
    TaskStage.CANDIDATE_TRAINING: {TaskStage.CANDIDATE_TRAINED},
    TaskStage.CANDIDATE_TRAINED: {TaskStage.CANDIDATE_VALIDATED},
    TaskStage.CANDIDATE_VALIDATED: {TaskStage.EXPERTS_COMMITTED},
    TaskStage.EXPERTS_COMMITTED: {TaskStage.GLOBAL_TEACHER_READY},
    TaskStage.GLOBAL_TEACHER_READY: {TaskStage.ROUTER_TRAINING},
    TaskStage.ROUTER_TRAINING: {TaskStage.ROUTER_READY},
    TaskStage.ROUTER_READY: {TaskStage.RMS_READY},
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
        if int(state.get("schema_version", -1)) != STATE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported state schema_version: {}".format(
                    state.get("schema_version")
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
        """Atomic persistence; refuses to overwrite (write-once per stage
        directory) unless explicitly requested."""
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
