"""V7 key lifecycle and reproducible current-candidate initialization."""

import hashlib
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .query import full_train_task_center


def tensor_checksum(value: Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def pairwise_key_cosine(keys: Tensor) -> Tensor:
    normalized = F.normalize(keys.detach().float(), dim=-1)
    return normalized @ normalized.T


def initialize_candidate_keys(
    queries: Tensor,
    num_train_samples: int,
    count: int = 4,
    perturbation: float = 0.01,
    seed: int = 42,
) -> Tuple[Tensor, Tensor, Dict[str, object]]:
    if count != 4:
        raise ValueError("V7 initializes exactly four candidates")
    if perturbation <= 0:
        raise ValueError("perturbation must be positive")
    center, coverage = full_train_task_center(queries, num_train_samples)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(count, center.numel(), generator=generator).to(center.device)
    # Distinct tangential directions keep every key equally close to the same
    # center without fabricating clusters.
    noise = noise - (noise @ center).unsqueeze(1) * center.unsqueeze(0)
    noise = F.normalize(noise, dim=-1)
    keys = F.normalize(center.unsqueeze(0) + float(perturbation) * noise, dim=-1)
    cosine = pairwise_key_cosine(keys)
    off_diagonal = cosine[~torch.eye(count, dtype=torch.bool, device=cosine.device)]
    if torch.any(off_diagonal >= 1.0 - 1.0e-8):
        raise AssertionError("candidate keys must not be identical")
    return keys, center, {
        **coverage,
        "seed": int(seed),
        "perturbation": float(perturbation),
        "pairwise_cosine": cosine.cpu().tolist(),
    }


class V7ExpertKeyPool(nn.Module):
    """All keys are selectable; only current keys are trainable."""

    def __init__(self, query_dim: int = 1536, pool_version: int = 0) -> None:
        super().__init__()
        if query_dim != 1536:
            raise ValueError("V7 expert keys must be 1536-D")
        self.query_dim = int(query_dim)
        self.pool_version = int(pool_version)
        self.keys = nn.ParameterDict()
        self.metadata: Dict[int, Dict[str, object]] = {}

    def add(
        self,
        expert_id: int,
        key: Tensor,
        origin_task: int,
        lifecycle: str,
        trainable: bool,
        rms_state: Optional[Mapping[str, object]] = None,
    ) -> None:
        expert_id = int(expert_id)
        if str(expert_id) in self.keys:
            raise ValueError("duplicate expert key {}".format(expert_id))
        if key.shape != (self.query_dim,):
            raise ValueError("expert key must have shape [1536]")
        normalized = F.normalize(key.detach().float(), dim=0)
        self.keys[str(expert_id)] = nn.Parameter(normalized, requires_grad=bool(trainable))
        self.metadata[expert_id] = {
            "expert_id": expert_id,
            "origin_task": int(origin_task),
            "lifecycle": str(lifecycle),
            "rms_state": dict(rms_state or {}),
        }

    @property
    def expert_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self.metadata))

    @property
    def current_ids(self) -> Tuple[int, ...]:
        return tuple(
            expert_id
            for expert_id in self.expert_ids
            if self.metadata[expert_id]["lifecycle"] == "current"
        )

    @property
    def historical_ids(self) -> Tuple[int, ...]:
        return tuple(
            expert_id
            for expert_id in self.expert_ids
            if self.metadata[expert_id]["lifecycle"] == "historical"
        )

    def normalized(self, expert_ids: Optional[Sequence[int]] = None) -> Tensor:
        values = self.expert_ids if expert_ids is None else tuple(int(v) for v in expert_ids)
        if not values:
            return torch.empty(0, self.query_dim)
        return torch.stack([F.normalize(self.keys[str(value)], dim=0) for value in values])

    def freeze_historical(self) -> None:
        for expert_id in self.historical_ids:
            self.keys[str(expert_id)].requires_grad_(False)
            self.keys[str(expert_id)].grad = None

    def freeze_all(self) -> None:
        for key in self.keys.values():
            key.requires_grad_(False)
            key.grad = None

    def historical_checksums(self) -> Dict[int, str]:
        return {
            expert_id: tensor_checksum(self.keys[str(expert_id)])
            for expert_id in self.historical_ids
        }

    def commit(self, retained_ids: Iterable[int], metrics: Mapping[int, Mapping[str, object]]) -> None:
        retained = {int(value) for value in retained_ids}
        for expert_id in list(self.current_ids):
            if expert_id in retained:
                self.metadata[expert_id]["lifecycle"] = "historical"
                self.metadata[expert_id]["validation"] = dict(metrics.get(expert_id, {}))
                self.keys[str(expert_id)].data.copy_(
                    F.normalize(self.keys[str(expert_id)].detach(), dim=0)
                )
                self.keys[str(expert_id)].requires_grad_(False)
                self.keys[str(expert_id)].grad = None
            else:
                self.metadata[expert_id]["lifecycle"] = "pruned"
                self.keys[str(expert_id)].requires_grad_(False)
                self.keys[str(expert_id)].grad = None
        self.pool_version += len(retained)

    def selectable_ids(self, excluded: Iterable[int] = ()) -> Tuple[int, ...]:
        excluded_set = {int(value) for value in excluded}
        return tuple(
            expert_id
            for expert_id in self.expert_ids
            if expert_id not in excluded_set
            and self.metadata[expert_id]["lifecycle"] != "pruned"
        )

    def export_state(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "query_dim": self.query_dim,
            "pool_version": self.pool_version,
            "keys": {key: value.detach().cpu() for key, value in self.keys.items()},
            "metadata": {str(key): dict(value) for key, value in self.metadata.items()},
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "V7ExpertKeyPool":
        pool = cls(int(state["query_dim"]), int(state.get("pool_version", 0)))
        metadata = state["metadata"]
        for raw_id in sorted(metadata, key=int):
            entry = dict(metadata[raw_id])
            lifecycle = str(entry["lifecycle"])
            pool.add(
                int(raw_id), state["keys"][raw_id], int(entry["origin_task"]),
                lifecycle, lifecycle == "current", entry.get("rms_state", {}),
            )
            pool.metadata[int(raw_id)].update(entry)
        pool.freeze_historical()
        return pool
