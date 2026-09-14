"""Global, deterministic, per-sample Top-2 routing for V7.

An expert may own several routing keys (a frozen canonical key plus one
learnable reuse key per later task it was judged reusable for), so the two
steps that V7 could fuse must be separated:

1. cosine similarity of the query against **every active route key**;
2. **max-aggregation per expert id** -- an expert is as reachable as its
   best-matching key -- while recording *which* key fired;
3. Top-2 over **distinct expert ids**.

It is wrong to Top-K over route keys and deduplicate afterwards: that can hand
one expert both slots through two of its own keys, and ``ComposeLinear`` counts
active slots, so the same LoRA would be composed twice.  The expert stays the
routing unit; keys only decide which expert is recalled.
"""

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
    # One entry per slot: the route key that made that expert reachable, and
    # its key type.  The key loss updates exactly this key.
    key_ids: Tuple[Tuple[str, str], ...] = ()
    key_types: Tuple[Tuple[str, str], ...] = ()


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

    def expert_score_matrix(
        self, queries: Tensor, excluded: Iterable[int] = ()
    ) -> Tuple[Tensor, Tensor, Tensor, List[str]]:
        """``(per_expert_scores [B,E], expert_ids [E], key_scores [B,K], key_ids)``.

        Columns of ``per_expert_scores`` are in ``key_pool.selectable_ids``
        order, which is the ascending expert-id order the tie-break has always
        used.
        """
        queries = F.normalize(queries.detach().float(), dim=-1)
        visible_ids = self.key_pool.selectable_ids(excluded)
        if len(visible_ids) < 2:
            raise ValueError("global Top-2 requires at least two selectable experts")
        key_ids = list(self.key_pool.active_key_ids(excluded))
        if not key_ids:
            raise ValueError("global Top-2 requires at least one active route key")
        keys = self.key_pool.normalized_keys(key_ids).to(queries.device)
        key_scores = queries @ keys.T
        key_expert = torch.tensor(
            [self.key_pool.route_keys[key_id].expert_id for key_id in key_ids],
            dtype=torch.long, device=key_scores.device,
        )
        pool_expert_ids = torch.unique(key_expert, sorted=True)
        if pool_expert_ids.tolist() != list(visible_ids):
            raise AssertionError(
                "route-key expert set {} disagrees with the selectable experts {}".format(
                    pool_expert_ids.tolist(), list(visible_ids)
                )
            )
        slot_of_key = torch.searchsorted(pool_expert_ids, key_expert)
        per_expert = torch.full(
            (key_scores.shape[0], pool_expert_ids.numel()),
            float("-inf"), dtype=key_scores.dtype, device=key_scores.device,
        ).scatter_reduce_(
            1, slot_of_key.unsqueeze(0).expand(key_scores.shape[0], -1),
            key_scores, reduce="amax", include_self=True,
        )
        return per_expert, pool_expert_ids, key_scores, key_ids

    def winning_key_positions(
        self,
        key_scores: Tensor,
        per_expert: Tensor,
        key_ids: Sequence[str],
        excluded: Iterable[int] = (),
    ) -> Tensor:
        """``[B, E]`` position in ``key_ids`` of the key that fired, or ``K``."""
        excluded_set = {int(value) for value in excluded}
        key_expert = torch.tensor(
            [
                self.key_pool.route_keys[key_id].expert_id
                for key_id in key_ids
            ],
            dtype=torch.long, device=key_scores.device,
        )
        visible = torch.tensor(
            list(self.key_pool.selectable_ids(excluded)),
            dtype=torch.long, device=key_scores.device,
        )
        slot_of_key = torch.searchsorted(visible, key_expert)
        assigned = key_scores == per_expert.index_select(1, slot_of_key)
        positions = torch.arange(key_scores.shape[1], device=key_scores.device)
        best = torch.where(
            assigned, positions.unsqueeze(0).expand_as(key_scores),
            torch.full_like(key_scores, key_scores.shape[1], dtype=torch.long),
        )
        return torch.full(
            (key_scores.shape[0], visible.numel()), key_scores.shape[1],
            dtype=torch.long, device=key_scores.device,
        ).scatter_reduce_(
            1, slot_of_key.unsqueeze(0).expand(key_scores.shape[0], -1),
            best, reduce="amin", include_self=True,
        )

    def forward(
        self, queries: Tensor, excluded: Iterable[int] = ()
    ) -> GlobalTop2Result:
        if queries.ndim != 2 or queries.shape[1] != self.key_pool.query_dim:
            raise ValueError("queries must have shape [B, 1536]")
        excluded = tuple(excluded)
        per_expert, pool_expert_ids, key_scores, key_ids = self.expert_score_matrix(
            queries, excluded
        )
        num_experts = per_expert.shape[1]
        # Stable tie breaking is deterministic and never imposes an old/new quota.
        tie = -torch.arange(num_experts, device=per_expert.device, dtype=per_expert.dtype) * 1.0e-7
        _, local = torch.topk(per_expert + tie, k=2, dim=-1, sorted=True)
        selected = pool_expert_ids[local]
        winning = self.winning_key_positions(
            key_scores, per_expert, key_ids, excluded
        ).gather(1, local)
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
        fired: List[Tuple[str, str]] = []
        fired_types: List[Tuple[str, str]] = []
        winning_rows = winning.detach().cpu().tolist()
        for row in range(selected.shape[0]):
            row_ids: List[str] = []
            row_types: List[str] = []
            for slot in range(2):
                position = int(winning_rows[row][slot])
                if position >= len(key_ids):
                    raise AssertionError("a selected expert has no matching active key")
                row_ids.append(key_ids[position])
                row_types.append(self.key_pool.route_keys[key_ids[position]].key_type)
            fired.append(tuple(row_ids))
            fired_types.append(tuple(row_types))
        # Return true cosine values, without the deterministic tie epsilon.
        cosine = torch.gather(per_expert, 1, local)
        return GlobalTop2Result(
            selected, cosine, selection, tuple(route_types),
            tuple(fired), tuple(fired_types),
        )
