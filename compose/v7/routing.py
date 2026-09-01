"""Global, deterministic, per-sample Top-2 routing for V7."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from compose.adapters.types import ComposeSelection, pad_selection

from .pool import V7ExpertKeyPool


@dataclass(frozen=True)
class GlobalTop2Result:
    expert_ids: Tensor
    similarities: Tensor
    selection: ComposeSelection
    route_types: Tuple[str, ...]


def route_signature_groups(expert_ids: Tensor) -> Dict[Tuple[int, int], Tensor]:
    groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for row, values in enumerate(expert_ids.detach().cpu().tolist()):
        groups[tuple(sorted(int(value) for value in values))].append(row)
    return {
        signature: torch.tensor(rows, dtype=torch.long, device=expert_ids.device)
        for signature, rows in groups.items()
    }


class GlobalTop2Router(nn.Module):
    def __init__(self, key_pool: V7ExpertKeyPool) -> None:
        super().__init__()
        self.key_pool = key_pool

    def forward(
        self, queries: Tensor, excluded: Iterable[int] = ()
    ) -> GlobalTop2Result:
        if queries.ndim != 2 or queries.shape[1] != self.key_pool.query_dim:
            raise ValueError("queries must have shape [B, 1536]")
        queries = F.normalize(queries.detach().float(), dim=-1)
        visible_ids = self.key_pool.selectable_ids(excluded)
        if len(visible_ids) < 2:
            raise ValueError("global Top-2 requires at least two selectable experts")
        keys = self.key_pool.normalized(visible_ids).to(queries.device)
        scores = queries @ keys.T
        # Stable tie breaking is deterministic and never imposes an old/new quota.
        tie = -torch.arange(len(visible_ids), device=scores.device, dtype=scores.dtype) * 1.0e-7
        values, local = torch.topk(scores + tie, k=2, dim=-1, sorted=True)
        id_table = torch.tensor(visible_ids, device=scores.device, dtype=torch.long)
        selected = id_table[local]
        rows = [
            pad_selection(tuple(int(v) for v in row), (1.0, 1.0))
            for row in selected.detach().cpu().tolist()
        ]
        selection = ComposeSelection(
            torch.tensor([list(value[0]) for value in rows], dtype=torch.long),
            torch.tensor([list(value[1]) for value in rows], dtype=torch.float32),
        )
        current = set(self.key_pool.current_ids)
        route_types = []
        for row in selected.detach().cpu().tolist():
            count = sum(int(value) in current for value in row)
            route_types.append(("OldOld", "OldNew", "NewNew")[count])
        # Return true cosine values, without the deterministic tie epsilon.
        cosine = torch.gather(scores, 1, local)
        return GlobalTop2Result(selected, cosine, selection, tuple(route_types))

