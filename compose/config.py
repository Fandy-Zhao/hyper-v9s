from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class ComposeAdapterConfig:
    """Configuration for the independent Compose LoRA expert stack."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
