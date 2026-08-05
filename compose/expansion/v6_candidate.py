"""V6 candidate pool construction and statistics (Stage E6).

Task 1: one candidate slot, cold-started on all task-1 train samples.
Task t > 1: two candidate slots; keys initialized from residual queries via
K-means++, falling back to random orthogonal keys; samples assigned to the
slot with the highest cosine (``selected_slot = argmax_m cos(q_i, k_m)``);
the training set is ``old_teacher_set + selected_candidate``.

Only the selected slot receives gradient; unselected slots stay at zero
gradient. The total loss is

    L = L_answer + lambda_key*L_key + lambda_margin*L_margin
        + lambda_balance*L_balance + lambda_diversity*L_diversity

The balance weight must stay small; slot balance is not forced. An empty
slot may be re-initialized once; if it stays empty it is allowed to die.
All switches live in the configuration (configs/v6_ucit_engineering.yaml).
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .candidate_pool import (
    CandidateExpertPool,
    CandidatePoolConfig,
    kmeans_plus_plus_keys,
    random_orthogonal_keys,
)

MAX_BALANCE_WEIGHT = 0.01  # task book: balance weight must stay small
MAX_SLOT_REINIT_ATTEMPTS = 1  # an empty slot may be re-initialized once


@dataclass(frozen=True)
class V6CandidateConfig:
    slot_count: int = 2  # task 1 -> 1; task t > 1 -> 2
    rank: int = 8
    alpha: float = 16.0
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    query_dim: int = 128
    key_initialization: str = "kmeans_plus_plus"  # | "random_orthogonal"
    key_init_seed: int = 42
    lambda_key: float = 0.1
    lambda_margin: float = 0.1
    lambda_balance: float = 0.001
    lambda_diversity: float = 0.01
    margin: float = 0.1
    min_support: int = 8
    min_key_recall: float = 0.5
    max_positive_jaccard: float = 0.8
    # Contribution-corrected assignment is OFF in the first version.
    use_contribution_corrected_assignment: bool = False
    # An empty slot may be re-initialized once; then it may die.
    allow_empty_slot_reinit: bool = True

    def __post_init__(self) -> None:
        if self.slot_count not in (1, 2):
            raise ValueError("V6 candidate pools use one or two slots")
        if self.rank <= 0 or self.alpha <= 0 or self.query_dim <= 0:
            raise ValueError("rank, alpha and query_dim must be positive")
        if self.key_initialization not in ("kmeans_plus_plus", "random_orthogonal"):
            raise ValueError("unknown key initialization: {!r}".format(self.key_initialization))
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")
        if self.lambda_balance < 0 or self.lambda_balance > MAX_BALANCE_WEIGHT:
            raise ValueError(
                "lambda_balance must lie in [0, {}] (balance weight must be "
                "small)".format(MAX_BALANCE_WEIGHT)
            )
        for name in ("lambda_key", "lambda_margin", "lambda_diversity", "margin"):
            if getattr(self, name) < 0:
                raise ValueError("{} must be non-negative".format(name))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SlotStatistics:
    slot_id: int
    sample_count: int
    share: float
    key: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AssignmentStatistics:
    per_slot: List[SlotStatistics]
    entropy: float
    empty_slot_ids: Tuple[int, ...]
    total_samples: int
    pair_key_cosine: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["empty_slot_ids"] = list(self.empty_slot_ids)
        return data


def build_v6_candidate_pool(
    config: V6CandidateConfig,
    residual_queries: Optional[Tensor],
    adapters: Sequence[nn.Module],
) -> Tuple[CandidateExpertPool, Dict[str, Any]]:
    """Initialize keys (K-means++ with orthogonal fallback) and build the pool.

    ``residual_queries`` are the frozen query features of residual samples;
    when None or too small, random orthogonal keys are used. The init record
    documents the method and seed.
    """
    count = config.slot_count
    keys = None
    method = config.key_initialization
    if (
        residual_queries is not None
        and residual_queries.ndim == 2
        and residual_queries.shape[0] >= count
    ):
        try:
            keys = kmeans_plus_plus_keys(
                residual_queries, count=count, seed=config.key_init_seed
            )
        except (ValueError, RuntimeError):
            keys = None
    if keys is None:
        method = "random_orthogonal"
        keys = random_orthogonal_keys(config.query_dim, count=count, seed=config.key_init_seed)
    if keys.dim() == 1:
        keys = keys.unsqueeze(0)
    pool_config = CandidatePoolConfig(
        query_dim=config.query_dim,
        slot_count=count,
        min_support=config.min_support,
        min_key_recall=config.min_key_recall,
        max_positive_jaccard=config.max_positive_jaccard,
    )
    pool = CandidateExpertPool(adapters=adapters, keys=keys, config=pool_config)
    init_record = {
        "method": method,
        "seed": config.key_init_seed,
        "slot_count": count,
        "query_samples": int(residual_queries.shape[0]) if residual_queries is not None else 0,
        "fallback_used": method != config.key_initialization,
    }
    return pool, init_record


def assignment_statistics(pool: CandidateExpertPool, queries: Tensor) -> AssignmentStatistics:
    """Per-slot sample counts, assignment entropy and empty slots."""
    if queries.ndim != 2 or queries.shape[0] == 0:
        raise ValueError("queries must be a non-empty [samples, dim] tensor")
    assignments = pool.assign(queries)
    counts = [int((assignments == slot_id).sum().item()) for slot_id in range(pool.config.slot_count)]
    total = int(assignments.numel())
    shares = [count / total for count in counts]
    entropy = -sum(share * math.log(share) for share in shares if share > 0)
    per_slot = [
        SlotStatistics(
            slot_id=slot_id,
            sample_count=counts[slot_id],
            share=shares[slot_id],
            key=[float(value) for value in pool.keys[slot_id].detach().cpu()],
        )
        for slot_id in range(pool.config.slot_count)
    ]
    pair_cosine = None
    if pool.config.slot_count == 2:
        pair_cosine = float(
            F.normalize(pool.keys[0].detach(), dim=0)
            @ F.normalize(pool.keys[1].detach(), dim=0)
        )
    return AssignmentStatistics(
        per_slot=per_slot,
        entropy=float(entropy),
        empty_slot_ids=tuple(slot_id for slot_id in range(pool.config.slot_count)
                             if counts[slot_id] == 0),
        total_samples=total,
        pair_key_cosine=pair_cosine,
    )


def reinit_empty_slots(
    pool: CandidateExpertPool,
    queries: Tensor,
    reinit_count: Dict[int, int],
    config: V6CandidateConfig,
) -> Tuple[int, Dict[str, Any]]:
    """Re-initialize empty slots once with K-means++ over their nearest
    residual queries; a slot that stays empty is allowed to die."""
    stats = assignment_statistics(pool, queries)
    record = {"reinitialized": [], "allowed_to_die": []}
    for slot_id in stats.empty_slot_ids:
        attempts = reinit_count.get(slot_id, 0)
        if attempts >= MAX_SLOT_REINIT_ATTEMPTS or not config.allow_empty_slot_reinit:
            record["allowed_to_die"].append(slot_id)
            continue
        # Re-seed from queries closest to the current slot key.
        with torch.no_grad():
            distances = 1.0 - F.normalize(queries.float(), dim=-1) @ F.normalize(
                pool.keys[slot_id].detach(), dim=0
            )
            nearest = queries[distances.argmin().item()].detach().float()
            pool.slots[slot_id].key.copy_(F.normalize(nearest, dim=0))
        reinit_count[slot_id] = attempts + 1
        record["reinitialized"].append(slot_id)
    return len(record["reinitialized"]), record
