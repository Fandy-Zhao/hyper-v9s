"""Expert-level Multi-Key Router.

V7 routes over ``num_experts`` keys where key id ``==`` expert id, so
``topk`` over keys is ``topk`` over experts by construction.  In V8 an expert may
own several keys, so the two steps must be separated:

1. cosine similarity of the query against **every active key**;
2. **max-aggregation per expert id** -- an expert is as reachable as its
   best-matching alias;
3. Top-K over **distinct expert ids**, so one expert can never occupy two slots.

The expert remains the routing unit and the composition unit; keys only decide
*which* expert is recalled.  This is what makes ``1 Expert : N Keys`` possible
without changing the forward pass that V7 already validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from compose.adapters.types import PAD_EXPERT_ID
from compose.v8.config import V8RoutingConfig
from compose.v8.pool import MultiKeyExpertPool, MultiKeyPoolError


TIE_EPSILON = 1.0e-7


@dataclass
class MultiKeyRouteResult:
    """Routing outcome for a batch of queries."""

    expert_ids: torch.Tensor          # (N, top_k) expert ids, best first
    expert_scores: torch.Tensor       # (N, top_k) max-aggregated similarity
    pool_expert_ids: torch.Tensor     # (E,) expert id of each column of `per_expert`
    per_expert_scores: torch.Tensor   # (N, E) max-aggregated similarity
    key_ids: List[List[Optional[str]]]  # (N, top_k) which key fired for each slot

    def as_rows(self, sample_ids: Sequence[str]) -> List[Dict[str, object]]:
        rows = []
        for index, sample_id in enumerate(sample_ids):
            rows.append({
                "sample_id": str(sample_id),
                "expert_ids": [int(value) for value in self.expert_ids[index].tolist()],
                "scores": [float(value) for value in self.expert_scores[index].tolist()],
                "key_ids": list(self.key_ids[index]),
            })
        return rows


class MultiKeyRouter(nn.Module):
    """Deterministic cosine router with per-expert max aggregation."""

    def __init__(self, config: Optional[V8RoutingConfig] = None) -> None:
        super().__init__()
        self.config = config or V8RoutingConfig()

    # ------------------------------------------------------------------
    def score_matrix(
        self,
        queries: torch.Tensor,
        pool: MultiKeyExpertPool,
        excluded_experts: Iterable[int] = (),
        return_keys: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[str]]]:
        """``(per_expert_scores (N,E), pool_expert_ids (E,), key_scores (N,K), key_ids)``."""
        key_ids = pool.active_key_ids(excluded_experts)
        if not key_ids:
            raise MultiKeyPoolError("the pool has no active keys to route over")
        queries = F.normalize(queries.detach().float(), dim=-1)
        keys = pool.normalized(key_ids).to(queries.device)
        key_scores = queries @ keys.T                                     # (N, K)
        key_expert = torch.tensor(
            [pool.key_records[key_id]["expert_id"] for key_id in key_ids],
            dtype=torch.long, device=key_scores.device,
        )
        pool_expert_ids = torch.unique(key_expert, sorted=True)           # (E,)
        slot_of_key = torch.searchsorted(pool_expert_ids, key_expert)     # (K,)
        per_expert = torch.full(
            (key_scores.shape[0], pool_expert_ids.numel()),
            float("-inf"), dtype=key_scores.dtype, device=key_scores.device,
        )
        per_expert.scatter_reduce_(
            1,
            slot_of_key.unsqueeze(0).expand(key_scores.shape[0], -1),
            key_scores,
            reduce="amax",
            include_self=True,
        )
        return per_expert, pool_expert_ids, key_scores, (key_ids if return_keys else None)

    # ------------------------------------------------------------------
    def forward(
        self,
        queries: torch.Tensor,
        pool: MultiKeyExpertPool,
        excluded_experts: Iterable[int] = (),
    ) -> MultiKeyRouteResult:
        per_expert, pool_expert_ids, key_scores, key_ids = self.score_matrix(
            queries, pool, excluded_experts, return_keys=True
        )
        num_experts = per_expert.shape[1]
        top_k = min(int(self.config.top_k), int(num_experts))
        # Deterministic tie-break: earlier (lower) expert id wins a tie.
        tie = -torch.arange(num_experts, device=per_expert.device,
                            dtype=per_expert.dtype) * TIE_EPSILON
        values, local = torch.topk(per_expert + tie, k=top_k, dim=-1, sorted=True)
        expert_ids = pool_expert_ids[local]

        # Attribute every selected expert back to the key that made it reachable.
        key_expert = torch.tensor(
            [pool.key_records[key_id]["expert_id"] for key_id in key_ids],
            dtype=torch.long, device=key_scores.device,
        )
        slot_of_key = torch.searchsorted(pool_expert_ids, key_expert)
        assigned = key_scores == per_expert.index_select(1, slot_of_key)
        positions = torch.arange(key_scores.shape[1], device=key_scores.device)
        best_position = torch.where(
            assigned, positions.unsqueeze(0).expand_as(key_scores),
            torch.full_like(key_scores, key_scores.shape[1], dtype=torch.long),
        )
        best_position = torch.full(
            (key_scores.shape[0], pool_expert_ids.numel()),
            key_scores.shape[1], dtype=torch.long, device=key_scores.device,
        ).scatter_reduce_(
            1, slot_of_key.unsqueeze(0).expand(key_scores.shape[0], -1),
            best_position, reduce="amin", include_self=True,
        )
        selected_positions = best_position.gather(1, local)
        fired: List[List[Optional[str]]] = []
        for row in range(queries.shape[0]):
            fired.append([
                key_ids[int(selected_positions[row, slot])]
                if int(selected_positions[row, slot]) < len(key_ids) else None
                for slot in range(top_k)
            ])

        scores = torch.where(
            torch.isfinite(values), values,
            torch.zeros_like(values),
        )
        return MultiKeyRouteResult(
            expert_ids=expert_ids,
            expert_scores=scores,
            pool_expert_ids=pool_expert_ids,
            per_expert_scores=per_expert,
            key_ids=fired,
        )

    # ------------------------------------------------------------------
    def top_m_experts(
        self,
        queries: torch.Tensor,
        pool: MultiKeyExpertPool,
        m: int,
        excluded_experts: Iterable[int] = (),
        restrict_to: Optional[Iterable[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-M **distinct experts** by max-aggregated key similarity.

        This is the teacher's recall step (STEP B): the historical candidate set
        from which singles and then pairs are searched.  ``restrict_to`` limits
        the candidate experts (e.g. only historical ones).
        """
        per_expert, pool_expert_ids, _, _ = self.score_matrix(
            queries, pool, excluded_experts
        )
        if restrict_to is not None:
            allowed = torch.tensor(sorted({int(value) for value in restrict_to}),
                                   dtype=pool_expert_ids.dtype,
                                   device=pool_expert_ids.device)
            mask = torch.isin(pool_expert_ids, allowed)
            if not bool(mask.any()):
                empty = torch.zeros(queries.shape[0], 0, dtype=pool_expert_ids.dtype,
                                    device=pool_expert_ids.device)
                return empty, torch.zeros(queries.shape[0], 0,
                                          dtype=per_expert.dtype, device=per_expert.device)
            per_expert = per_expert[:, mask]
            pool_expert_ids = pool_expert_ids[mask]
        num_experts = per_expert.shape[1]
        k = min(int(m), int(num_experts))
        tie = -torch.arange(num_experts, device=per_expert.device,
                            dtype=per_expert.dtype) * TIE_EPSILON
        values, local = torch.topk(per_expert + tie, k=k, dim=-1, sorted=True)
        return pool_expert_ids[local], values

    # ------------------------------------------------------------------
    def score_for_experts(
        self,
        queries: torch.Tensor,
        pool: MultiKeyExpertPool,
        expert_ids: Sequence[int],
    ) -> torch.Tensor:
        """Aggregated similarity for an explicit expert shortlist, no Top-K."""
        per_expert, pool_expert_ids, _, _ = self.score_matrix(queries, pool)
        wanted = torch.tensor([int(value) for value in expert_ids],
                              dtype=pool_expert_ids.dtype, device=pool_expert_ids.device)
        positions = torch.searchsorted(pool_expert_ids, wanted)
        if positions.numel() and int(positions.max()) >= pool_expert_ids.numel():
            raise MultiKeyPoolError("score_for_experts received an inactive expert")
        if positions.numel() and not torch.equal(pool_expert_ids[positions], wanted):
            raise MultiKeyPoolError("score_for_experts received an inactive expert")
        return per_expert.index_select(1, positions)


def unique_expert_rows(
    sample_ids: Sequence[str],
    expert_ids: torch.Tensor,
) -> Dict[str, List[int]]:
    """Per-sample **distinct** expert list, order preserved.

    A repeated id is a bug, not a policy: ``ComposeLinear`` counts active *slots*,
    so ``[a, a]`` would compose expert ``a`` with weight ``2 * kappa / sqrt(2)``
    rather than ``kappa``.  Duplicates are collapsed here and the collapse is
    reported by :func:`duplicate_row_count` so it can be asserted to be zero.
    """
    rows: Dict[str, List[int]] = {}
    for sample_id, row in zip(sample_ids, expert_ids.detach().cpu().tolist()):
        seen: List[int] = []
        for value in row:
            if int(value) not in seen:
                seen.append(int(value))
        rows[str(sample_id)] = seen
    return rows


def duplicate_row_count(sample_ids: Sequence[str], expert_ids: torch.Tensor) -> int:
    """How many rows repeat a *real* expert id.

    Must be 0 for a legal selection: ``ComposeLinear`` counts active slots, so a
    repeated id doubles that expert's gate instead of selecting it once.  Pad
    slots (``-1``) repeat by construction and are not duplicates.
    """
    count = 0
    for row in expert_ids.detach().cpu().tolist():
        values = [int(value) for value in row if int(value) != PAD_EXPERT_ID]
        if len(values) != len(set(values)):
            count += 1
    return count


def v7_pair_manifest(
    sample_ids: Sequence[str],
    selection: Mapping[str, Sequence[int]],
) -> Dict[str, object]:
    """Emit the V7 ``eval_task`` manifest for the **pair-only** subset.

    ``compose.eval.eval_task`` hard-requires exactly two distinct ids
    (``eval_task.py:304``), so only two-expert samples can be expressed there.
    This exists purely to cross-check V8's ``Reuse2`` forward against the
    already-validated V7 evaluator; the V8 native path is
    :mod:`compose.v8.selection`, which represents all four states.
    """
    manifest: Dict[str, object] = {}
    for sample_id in sample_ids:
        experts = [int(value) for value in selection[str(sample_id)]]
        if len(experts) != 2 or len(set(experts)) != 2:
            raise ValueError(
                f"sample {sample_id} has {len(experts)} distinct experts; the V7 "
                "manifest cannot express policies other than a distinct pair"
            )
        manifest[str(sample_id)] = {
            "global_top2": experts,
            "v8_expert_ids": experts,
            "v8_policy": "pair",
        }
    return manifest


def distinct_expert_rate(expert_ids: torch.Tensor) -> float:
    """Fraction of rows whose two slots name two different experts."""
    if expert_ids.numel() == 0:
        return 0.0
    if expert_ids.shape[1] < 2:
        return 0.0
    return float((expert_ids[:, 0] != expert_ids[:, 1]).float().mean().item())


__all__ = [
    "MultiKeyRouteResult",
    "MultiKeyRouter",
    "TIE_EPSILON",
    "distinct_expert_rate",
    "duplicate_row_count",
    "unique_expert_rows",
    "v7_pair_manifest",
]
