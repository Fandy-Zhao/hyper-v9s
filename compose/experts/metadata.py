from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ExpertStatus(str, Enum):
    REGISTERED = "registered"
    TRAINING = "training"
    FROZEN = "frozen"


@dataclass
class ExpertMetadata:
    expert_id: int
    name: str
    status: ExpertStatus = ExpertStatus.REGISTERED
    source_checkpoint: Optional[str] = None
    trained_steps: int = 0
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.expert_id < 0:
            raise ValueError("expert_id must be non-negative")
        if not self.name:
            raise ValueError("expert name must not be empty")
        if self.trained_steps < 0:
            raise ValueError("trained_steps must be non-negative")

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ExpertMetadata":
        values = dict(data)
        values["status"] = ExpertStatus(values.get("status", "registered"))
        return cls(**values)
