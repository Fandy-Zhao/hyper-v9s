from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class V7QueryConfig:
    visual_dim: int = 768
    text_dim: int = 768
    query_dim: int = 1536

    def __post_init__(self) -> None:
        if self.visual_dim != 768 or self.text_dim != 768 or self.query_dim != 1536:
            raise ValueError("V7 fixed query dimensions must be 768 + 768 = 1536")


@dataclass(frozen=True)
class V7CandidateConfig:
    count: int = 4
    rank: int = 8
    alpha: float = 16.0
    key_perturbation: float = 0.01

    def __post_init__(self) -> None:
        if self.count != 4 or self.rank != 8:
            raise ValueError("V7 requires exactly four rank-8 candidates")
        if self.key_perturbation <= 0:
            raise ValueError("candidate key perturbation must be positive")


@dataclass(frozen=True)
class V7RoutingConfig:
    top_k: int = 2
    pair_scale: float = 2.0 ** -0.5

    def __post_init__(self) -> None:
        if self.top_k != 2:
            raise ValueError("V7 routing is fixed to global Top-2")
        if self.pair_scale <= 0:
            raise ValueError("pair_scale must be positive")


@dataclass(frozen=True)
class V7TrainingConfig:
    lambda_key: float = 0.1
    key_learning_rate: float = 3.0e-4
    lora_learning_rate: float = 2.0e-4
    gradient_audit_every: int = 1


@dataclass(frozen=True)
class V7PruningConfig:
    candidate_prune_enabled: bool = True
    candidate_min_usage: float = 0.0
    candidate_min_removal_gain: float = 0.0
    candidate_redundancy_cosine: float = 0.98

    def __post_init__(self) -> None:
        if not 0 <= self.candidate_min_usage <= 1:
            raise ValueError("candidate_min_usage must be in [0, 1]")
        if not -1 <= self.candidate_redundancy_cosine <= 1:
            raise ValueError("candidate_redundancy_cosine must be in [-1, 1]")


@dataclass(frozen=True)
class V7Config:
    schema_version: int = 1
    method: str = "v7_global_coevolution"
    seed: int = 42
    query: V7QueryConfig = field(default_factory=V7QueryConfig)
    candidates: V7CandidateConfig = field(default_factory=V7CandidateConfig)
    routing: V7RoutingConfig = field(default_factory=V7RoutingConfig)
    training: V7TrainingConfig = field(default_factory=V7TrainingConfig)
    pruning: V7PruningConfig = field(default_factory=V7PruningConfig)

    def __post_init__(self) -> None:
        if self.method != "v7_global_coevolution":
            raise ValueError("V7 requires method: v7_global_coevolution")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "V7Config":
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            method=str(value.get("method", "v7_global_coevolution")),
            seed=int(value.get("seed", 42)),
            query=V7QueryConfig(**value.get("query", {})),
            candidates=V7CandidateConfig(**value.get("candidates", {})),
            routing=V7RoutingConfig(**value.get("routing", {})),
            training=V7TrainingConfig(**value.get("training", {})),
            pruning=V7PruningConfig(**value.get("pruning", {})),
        )

