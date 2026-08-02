"""Bounded train-only frozen Query anchors for continual routing."""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import torch


@dataclass(frozen=True)
class AnchorRecord:
    sample_id: str
    expert_id: int
    creation_task: int
    split: str
    feature_hash: str
    query: Tuple[float, ...]

    def __post_init__(self) -> None:
        if self.split != "train":
            raise ValueError("anchors may only come from train")
        if len(self.query) != 128:
            raise ValueError("anchor Query must have 128 dimensions")


class AnchorMemory:
    def __init__(self, capacity_per_expert: int = 32) -> None:
        self.capacity_per_expert = int(capacity_per_expert)
        self._records: Dict[int, List[AnchorRecord]] = {}

    def add(self, record: AnchorRecord, current_task: int) -> bool:
        if record.creation_task > int(current_task):
            raise ValueError("future-task anchor leakage")
        rows = self._records.setdefault(record.expert_id, [])
        if any(item.sample_id == record.sample_id for item in rows):
            return False
        if len(rows) >= self.capacity_per_expert:
            return False
        rows.append(record)
        return True

    def records(self, current_task: int) -> Tuple[AnchorRecord, ...]:
        return tuple(item for expert in sorted(self._records) for item in self._records[expert] if item.creation_task <= current_task)

    def state_dict(self):
        return {"capacity_per_expert": self.capacity_per_expert, "records": [asdict(item) for item in self.records(10**9)]}

    def load_state_dict(self, state) -> None:
        self.capacity_per_expert = int(state["capacity_per_expert"])
        self._records = {}
        for value in state["records"]:
            self.add(AnchorRecord(**value), current_task=int(value["creation_task"]))
