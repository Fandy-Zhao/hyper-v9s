"""Learnable normalized expert keys bound to registry metadata."""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ExpertKeyMetadata:
    expert_id: int
    creation_task: int
    checkpoint_sha256: str
    key_version: int = 1
    archived: bool = False
    initialization: str = "seeded_random"

    def __post_init__(self) -> None:
        if self.expert_id < 0 or self.creation_task < 0 or self.key_version <= 0:
            raise ValueError("invalid expert key metadata")
        if not self.checkpoint_sha256:
            raise ValueError("checkpoint hash is required")


class ExpertKeyStore(nn.Module):
    def __init__(self, metadata: Sequence[ExpertKeyMetadata], query_dim: int = 128, seed: int = 42) -> None:
        super().__init__()
        if query_dim <= 0:
            raise ValueError("query_dim must be positive")
        ordered = sorted(metadata, key=lambda item: item.expert_id)
        if len({item.expert_id for item in ordered}) != len(ordered):
            raise ValueError("duplicate expert IDs")
        self.query_dim = int(query_dim)
        self.metadata = {item.expert_id: item for item in ordered}
        generator = torch.Generator().manual_seed(int(seed))
        self.keys = nn.ParameterDict({
            str(item.expert_id): nn.Parameter(F.normalize(torch.randn(query_dim, generator=generator), dim=0))
            for item in ordered
        })

    @property
    def expert_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self.metadata))

    def validate_registry(self, registry) -> None:
        registry_ids = tuple(sorted(item.expert_id for item in registry.list_all()))
        if registry_ids != self.expert_ids:
            raise ValueError(f"registry/key expert IDs disagree: {registry_ids} != {self.expert_ids}")
        for item in registry.list_all():
            meta = self.metadata[item.expert_id]
            if int(item.creation_task) != meta.creation_task or str(item.checkpoint_sha256) != meta.checkpoint_sha256:
                raise ValueError(f"registry/key metadata mismatch for expert {item.expert_id}")

    def visible_expert_ids(self, task_id: int, historical_only: bool = False) -> Tuple[int, ...]:
        boundary = int(task_id) - int(bool(historical_only))
        return tuple(item.expert_id for item in self.metadata.values() if not item.archived and item.creation_task <= boundary)

    def normalized(self, expert_ids: Optional[Iterable[int]] = None) -> Tensor:
        ids = self.expert_ids if expert_ids is None else tuple(int(value) for value in expert_ids)
        if not ids:
            return torch.empty(0, self.query_dim, device=next(self.parameters()).device)
        return torch.stack([F.normalize(self.keys[str(value)], dim=0) for value in ids])

    @torch.no_grad()
    def initialize_from_queries(self, expert_id: int, queries: Tensor, minimum_positives: int = 2) -> bool:
        if expert_id not in self.metadata:
            raise KeyError(expert_id)
        if queries.ndim != 2 or queries.shape[1] != self.query_dim:
            raise ValueError("queries must have shape [samples, 128]")
        if len(queries) < minimum_positives:
            return False
        self.keys[str(expert_id)].copy_(F.normalize(queries.float().mean(dim=0), dim=0))
        return True

    def metadata_state(self) -> Dict[str, object]:
        return {"query_dim": self.query_dim, "experts": [asdict(self.metadata[value]) for value in self.expert_ids]}
