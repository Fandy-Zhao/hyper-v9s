"""Two-slot keyed residual expert pool for Compose P2."""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from compose.experts import ExpertMetadata, ExpertRegistry


@dataclass(frozen=True)
class CandidatePoolConfig:
    query_dim: int = 128
    slot_count: int = 2
    min_support: int = 8
    min_key_recall: float = 0.5
    max_positive_jaccard: float = 0.8

    def __post_init__(self):
        if self.slot_count not in (1, 2):
            raise ValueError("candidate pools support one or two slots "
                             "(task 1: 1, task t>1: 2)")
        if self.query_dim <= 0 or self.min_support <= 0:
            raise ValueError("query_dim and min_support must be positive")


@dataclass(frozen=True)
class SlotValidation:
    support: int
    mean_conditional_gain: float
    validation_accuracy_delta: float
    key_recall_at_1: float
    positive_sample_ids: Tuple[str, ...]


class CandidateSlot(nn.Module):
    def __init__(self, adapter: nn.Module, key: Tensor):
        super().__init__()
        if key.ndim != 1:
            raise ValueError("candidate key must be rank one")
        self.adapter = adapter
        self.key = nn.Parameter(F.normalize(key.detach().float(), dim=0))
        self.register_buffer("usage_count", torch.zeros((), dtype=torch.long))


def kmeans_plus_plus_keys(queries: Tensor, count: int = 2, seed: int = 0) -> Tensor:
    if queries.ndim != 2 or queries.shape[0] < count:
        raise ValueError("queries must contain at least count rows")
    generator = torch.Generator(device=queries.device).manual_seed(int(seed))
    normalized = F.normalize(queries.float(), dim=-1)
    first = int(torch.randint(normalized.shape[0], (1,), generator=generator, device=queries.device))
    centers = [normalized[first]]
    while len(centers) < count:
        distances = 1.0 - normalized @ torch.stack(centers).T
        minimum = distances.clamp_min(0).min(dim=1).values.square()
        if float(minimum.sum()) == 0.0:
            index = next(index for index in range(normalized.shape[0]) if not any(torch.equal(normalized[index], center) for center in centers))
        else:
            index = int(torch.multinomial(minimum, 1, generator=generator))
        centers.append(normalized[index])
    return torch.stack(centers)


def random_orthogonal_keys(query_dim: int, count: int = 2, seed: int = 0) -> Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    matrix = torch.randn(query_dim, count, generator=generator)
    return torch.linalg.qr(matrix, mode="reduced").Q.T.contiguous()


class CandidateExpertPool(nn.Module):
    def __init__(self, adapters: Sequence[nn.Module], keys: Tensor, config: CandidatePoolConfig = None):
        super().__init__()
        self.config = config or CandidatePoolConfig(query_dim=int(keys.shape[1]))
        if len(adapters) != self.config.slot_count or keys.shape != (self.config.slot_count, self.config.query_dim):
            raise ValueError(
                "expected {} adapters and keys of shape ({}, {})".format(
                    self.config.slot_count, self.config.slot_count, self.config.query_dim
                )
            )
        self.slots = nn.ModuleList(CandidateSlot(adapter, key) for adapter, key in zip(adapters, keys))

    @property
    def keys(self):
        return torch.stack([slot.key for slot in self.slots])

    def assign(self, queries: Tensor) -> Tensor:
        similarities = F.normalize(queries.float(), dim=-1) @ F.normalize(self.keys, dim=-1).T
        tie = torch.arange(self.config.slot_count, device=queries.device, dtype=similarities.dtype) * -1e-7
        return torch.argmax(similarities + tie, dim=-1)

    def forward(self, hidden_states: Tensor, queries: Tensor, selected_old_output: Tensor):
        if hidden_states.shape[0] != queries.shape[0] or hidden_states.shape[0] != selected_old_output.shape[0]:
            raise ValueError("batch dimensions must match")
        assignments = self.assign(queries)
        result = selected_old_output.clone()
        for slot_id, slot in enumerate(self.slots):
            rows = torch.where(assignments.eq(slot_id))[0]
            if rows.numel():
                delta = slot.adapter(hidden_states.index_select(0, rows)).to(result.dtype)
                result.index_add_(0, rows, delta)
                if self.training:
                    slot.usage_count.add_(rows.numel())
        return result, assignments

    def independent_optimizers(self, **kwargs):
        optimizers = []
        parameter_ids = set()
        for slot in self.slots:
            parameters = list(slot.parameters())
            current = {id(parameter) for parameter in parameters}
            if current & parameter_ids:
                raise ValueError("candidate slots share optimizer parameters")
            parameter_ids |= current
            optimizers.append(torch.optim.AdamW(parameters, **kwargs))
        return tuple(optimizers)

    def commit(self, validations: Sequence[SlotValidation]):
        if len(validations) != self.config.slot_count:
            raise ValueError("one validation record per slot is required")
        positive = [set(item.positive_sample_ids) for item in validations]
        if len(positive) == 2:
            union = positive[0] | positive[1]
            jaccard = len(positive[0] & positive[1]) / len(union) if union else 0.0
        else:
            jaccard = 0.0  # single-slot pool has no cross-slot redundancy
        decisions = []
        for slot_id, item in enumerate(validations):
            passed = (
                item.support >= self.config.min_support
                and item.mean_conditional_gain > 0
                and item.validation_accuracy_delta >= 0
                and item.key_recall_at_1 >= self.config.min_key_recall
                and jaccard <= self.config.max_positive_jaccard
            )
            decisions.append({
                "slot_id": slot_id,
                "status": "provisional" if passed else "rejected",
                "commit": passed,
                "validation": asdict(item),
                "positive_jaccard": jaccard,
            })
        return {"commit_count": sum(item["commit"] for item in decisions), "slots": decisions}

    def commit_to_registry(self, registry: ExpertRegistry, decision: Dict[str, object],
                           first_expert_id: int, creation_task: int,
                           rank: int = 8, alpha: float = 16.0):
        if not isinstance(registry, ExpertRegistry):
            raise TypeError("registry must be an ExpertRegistry")
        committed = []
        for slot in decision.get("slots", []):
            if not slot.get("commit"):
                continue
            expert_id = int(first_expert_id) + int(slot["slot_id"])
            metadata = ExpertMetadata(
                expert_id=expert_id,
                adapter_name="candidate-slot-{}".format(slot["slot_id"]),
                rank=int(rank),
                alpha=float(alpha),
                creation_task=int(creation_task),
                support_count=int(slot["validation"]["support"]),
                positive_contribution_count=len(slot["validation"]["positive_sample_ids"]),
                extra={"lifecycle_state": "provisional", "candidate_slot": int(slot["slot_id"])},
            )
            registry.register(metadata)
            committed.append(metadata)
        return tuple(committed)


def candidate_pool_loss(
    answer_loss: Tensor,
    selected_similarity: Tensor,
    unselected_similarity: Tensor,
    usage: Tensor,
    activation_cosine: Tensor,
    lambda_key: float,
    lambda_margin: float,
    lambda_balance: float,
    lambda_diversity: float,
    margin: float = 0.1,
) -> Dict[str, Tensor]:
    key_loss = (1.0 - selected_similarity).mean()
    margin_loss = F.relu(margin - selected_similarity + unselected_similarity).mean()
    target = usage.new_full(usage.shape, 1.0 / usage.numel())
    balance_loss = (usage - target).square().mean()
    diversity_loss = activation_cosine.clamp_min(0).mean()
    total = answer_loss + lambda_key * key_loss + lambda_margin * margin_loss + lambda_balance * balance_loss + lambda_diversity * diversity_loss
    return {"total": total, "answer": answer_loss, "key": key_loss, "margin": margin_loss, "balance": balance_loss, "diversity": diversity_loss}
