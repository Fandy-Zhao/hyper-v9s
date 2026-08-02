"""Typed records for the answer-supervised expert-set teacher."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple


def stable_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OracleConfig:
    oracle_name: str
    composition_mode: str
    lambda_expert: float = 0.01
    delta_pair_raw: float = 0.02
    top_k_for_pair: int = 4
    max_pairs: int = 6
    include_eos: bool = False
    answer_mask_version: str = "stage04_answer_mask_v1"
    target_averaging: str = "token_mean"
    cache_schema_version: int = 1

    def __post_init__(self) -> None:
        if self.composition_mode not in ("direct_sum", "rms_calibrated"):
            raise ValueError("Oracle composition mode must be direct_sum or rms_calibrated")
        if self.lambda_expert < 0 or self.delta_pair_raw < 0:
            raise ValueError("Oracle penalties and thresholds must be non-negative")
        if self.top_k_for_pair <= 0 or self.max_pairs < 0:
            raise ValueError("invalid candidate search limits")
        if self.target_averaging != "token_mean":
            raise ValueError("Stage 04 requires token_mean target averaging")
        if self.cache_schema_version <= 0 or not self.answer_mask_version:
            raise ValueError("cache and answer-mask versions are required")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OracleConfig":
        fields = {key: value[key] for key in asdict(cls("x", "direct_sum")) if key in value}
        return cls(**fields)

    @property
    def config_hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class AnswerNLL:
    sum_nll: float
    mean_nll: float
    token_count: int
    exact_teacher_forced: Optional[bool] = None
    per_token_nll: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.token_count <= 0:
            raise ValueError("answer NLL requires at least one supervised token")
        if not math.isfinite(self.sum_nll) or not math.isfinite(self.mean_nll):
            raise ValueError("answer NLL must be finite")

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["per_token_nll"] = list(self.per_token_nll)
        return result


@dataclass
class CandidateScore:
    expert_ids: Tuple[int, ...]
    nll: AnswerNLL
    score: float
    raw_gain_over_best_single: Optional[float] = None
    penalized_gain_over_best_single: Optional[float] = None
    valid_pair: Optional[bool] = None

    def __post_init__(self) -> None:
        self.expert_ids = tuple(sorted(int(value) for value in self.expert_ids))
        if len(self.expert_ids) != len(set(self.expert_ids)) or len(self.expert_ids) > 2:
            raise ValueError("candidate sets require at most two unique expert IDs")
        if not math.isfinite(float(self.score)):
            raise ValueError("candidate score must be finite")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_ids": list(self.expert_ids),
            **self.nll.to_dict(),
            "score": float(self.score),
            "raw_gain_over_best_single": self.raw_gain_over_best_single,
            "penalized_gain_over_best_single": self.penalized_gain_over_best_single,
            "valid_pair": self.valid_pair,
        }


@dataclass
class OracleRecord:
    sample_id: str
    task_id: int
    task_name: str
    split: str
    answer_token_count: int
    candidate_expert_ids: Tuple[int, ...]
    composition_mode: str
    empty: CandidateScore
    singles: Tuple[CandidateScore, ...]
    pairs: Tuple[CandidateScore, ...]
    selected: CandidateScore
    config_hash: str
    expert_pool_hash: str
    dataset_manifest_hash: str
    temporal_scope: str
    best_single: Optional[CandidateScore] = None
    best_pair: Optional[CandidateScore] = None
    cache_key: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        best_single = self.best_single.to_dict() if self.best_single else None
        best_pair = self.best_pair.to_dict() if self.best_pair else None
        return {
            "sample_id": self.sample_id,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "split": self.split,
            "answer_token_count": self.answer_token_count,
            "candidate_expert_ids": list(self.candidate_expert_ids),
            "composition_mode": self.composition_mode,
            "temporal_scope": self.temporal_scope,
            "empty": self.empty.to_dict(),
            "singles": [value.to_dict() for value in self.singles],
            "pairs": [value.to_dict() for value in self.pairs],
            "best_single": best_single,
            "best_pair": best_pair,
            "best_single_ids": best_single["expert_ids"] if best_single else None,
            "best_single_nll": best_single["mean_nll"] if best_single else None,
            "best_pair_ids": best_pair["expert_ids"] if best_pair else None,
            "best_pair_nll": best_pair["mean_nll"] if best_pair else None,
            "selected_expert_ids": list(self.selected.expert_ids),
            "selected_cardinality": len(self.selected.expert_ids),
            "selected_mean_nll": self.selected.nll.mean_nll,
            "selected_score": self.selected.score,
            "pair_synergy": self.best_pair.raw_gain_over_best_single if self.best_pair else None,
            "pair_required": len(self.selected.expert_ids) == 2,
            "config_hash": self.config_hash,
            "expert_pool_hash": self.expert_pool_hash,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "cache_key": self.cache_key,
            "diagnostics": self.diagnostics,
        }
