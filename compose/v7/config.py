from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


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
    learning_rate: float = 2.0e-4
    # Backward-compatible override for older V7 configs. Formal configs use
    # ``learning_rate`` so the recipe matches Hugging Face/UCIT terminology.
    lora_learning_rate: Optional[float] = None
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 64
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    seed: int = 42
    gradient_checkpointing: bool = True
    group_by_modality_length: bool = True
    bf16: bool = True
    tf32: bool = True
    logging_steps: int = 1
    save_strategy: str = "epoch"
    model_max_length: int = 2048
    dataloader_num_workers: int = 4
    gradient_audit_every: int = 1

    def __post_init__(self) -> None:
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio must be in [0, 1]")

    @property
    def effective_lora_learning_rate(self) -> float:
        return float(
            self.learning_rate
            if self.lora_learning_rate is None
            else self.lora_learning_rate
        )


@dataclass(frozen=True)
class V7RuntimeConfig:
    image_aspect_ratio: str = "pad"
    mm_vision_select_layer: int = -2
    mm_vision_select_feature: str = "patch"
    mm_projector_type: str = "mlp2x_gelu"

    def __post_init__(self) -> None:
        if self.image_aspect_ratio != "pad":
            raise ValueError("formal V7 requires image_aspect_ratio: pad")


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
    runtime: V7RuntimeConfig = field(default_factory=V7RuntimeConfig)
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
            runtime=V7RuntimeConfig(**value.get("runtime", {})),
            pruning=V7PruningConfig(**value.get("pruning", {})),
        )

