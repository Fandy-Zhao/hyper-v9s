"""Global multi-key expert aggregation over V9-S keys (spec §20).

V8's :class:`~compose.v8.routing.MultiKeyRouter` scores ``normalize(key)`` for
every key in the pool, and so does this module: a V9-S key is absolute, so the
score matrix is exactly the normalized stored parameters.  The aggregation
semantics are V8's, unchanged -- per-expert ``max`` over the expert's retained
keys, distinct-expert Top-K, deterministic tie-break by expert id.

One expert's memory is ``{base key} ∪ {retained task keys}``.  Recalling the
expert when *any* of its keys matches is what lets a capability learned on task 1
stay reachable after the task keys of the intervening tasks were reset: the base
key never moves, so the expert is never orphaned.

This is a query-only computation.  It reads no answer, no task index, and no
training recall file; :mod:`compose.v9.inference` is the module that is scanned
for that property.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from compose.v8.config import (
    KEY_TYPE_TASK_ALIAS,
    STATE_BASE_ONLY,
    STATE_REUSE1,
    STATE_REUSE2,
    V8RoutingConfig,
)
from compose.v8.selection import build_selection

from .keys import V9KeyPool


#: Tie-break weight: a perturbation smaller than any real cosine gap, so it can
#: order a near-tie without ever changing which experts are similar.
TIE_EPSILON = 1e-6


class V9MultiKeyError(RuntimeError):
    """Raised when the aggregation is asked for something undefined."""


@dataclass
class V9MultiKeyRouteResult:
    """``[N, top_k]`` expert choices plus the key that made each one reachable."""

    expert_ids: torch.Tensor          # [N, K] long
    expert_scores: torch.Tensor       # [N, K] float
    pool_expert_ids: torch.Tensor     # [E] long
    per_expert_scores: torch.Tensor   # [N, E] float
    key_ids: List[List[Optional[str]]]

    @property
    def rows(self) -> int:
        return int(self.expert_ids.shape[0])


def memory_key_ids(key_pool: V9KeyPool, expert_id: int) -> List[str]:
    """Every key in an expert's retained memory, base key first.

    Order is deterministic -- base, then task keys by task index -- so the
    arg-max that attributes a score back to a key never depends on dictionary
    insertion order.
    """
    expert_id = int(expert_id)
    base = key_pool.base_key_id(expert_id)
    ordered = [base]
    task_keys = [
        key_id
        for key_id in key_pool.memory_key_ids(expert_id)
        if key_pool.key_records[key_id]["key_type"] == KEY_TYPE_TASK_ALIAS
        and key_id != base
    ]
    task_keys.sort(key=lambda key_id: (key_pool.key_records[key_id]["task_id"], key_id))
    ordered.extend(task_keys)
    return ordered


def aggregatable_expert_ids(
    key_pool: V9KeyPool, expert_ids: Optional[Iterable[int]] = None
) -> List[int]:
    """The experts an aggregation may route over, sorted by id.

    Defaults to the **committed** experts only: a candidate that has not passed
    its task-end audit is not deployable, and defaulting to "everything live"
    would make an uncommitted expert reachable at test time by omission.  The
    task-end audit is the one caller that passes an explicit list, because
    asking "would this candidate help if it were committed?" is precisely its
    job.
    """
    if expert_ids is None:
        selected = key_pool.historical_ids
    else:
        selected = [int(value) for value in expert_ids]
    live = set(key_pool.live_expert_ids())
    unknown = sorted(set(selected) - live)
    if unknown:
        raise V9MultiKeyError(
            "aggregation named experts that are not live in this pool: {}".format(
                unknown[:10]
            )
        )
    return sorted(set(selected))


class V9MultiKeyRouter(nn.Module):
    """Cosine aggregation over effective keys, with per-expert max pooling."""

    def __init__(self, config: Optional[V8RoutingConfig] = None) -> None:
        super().__init__()
        self.config = config or V8RoutingConfig()

    def score_matrix(
        self,
        queries: torch.Tensor,
        key_pool: V9KeyPool,
        expert_ids: Optional[Iterable[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """``(per_expert [N,E], pool_expert_ids [E], key_scores [N,K], key_ids)``."""
        pool_expert_ids = aggregatable_expert_ids(key_pool, expert_ids)
        if not pool_expert_ids:
            raise V9MultiKeyError("the pool has no live expert to route over")
        key_ids: List[str] = []
        key_expert: List[int] = []
        for expert_id in pool_expert_ids:
            for key_id in memory_key_ids(key_pool, expert_id):
                key_ids.append(key_id)
                key_expert.append(expert_id)
        keys = key_pool.effective_key_matrix(key_ids, detach=True).to(queries.device)
        query_matrix = F.normalize(queries.detach().float(), dim=-1)
        key_scores = query_matrix @ keys.T                       # [N, K]
        expert_tensor = torch.tensor(
            key_expert, dtype=torch.long, device=key_scores.device
        )
        pool_tensor = torch.tensor(
            pool_expert_ids, dtype=torch.long, device=key_scores.device
        )
        slot_of_key = torch.searchsorted(pool_tensor, expert_tensor)
        per_expert = torch.full(
            (key_scores.shape[0], pool_tensor.numel()),
            float("-inf"),
            dtype=key_scores.dtype,
            device=key_scores.device,
        )
        per_expert.scatter_reduce_(
            1,
            slot_of_key.unsqueeze(0).expand(key_scores.shape[0], -1),
            key_scores,
            reduce="amax",
            include_self=True,
        )
        return per_expert, pool_tensor, key_scores, key_ids

    def forward(
        self,
        queries: torch.Tensor,
        key_pool: V9KeyPool,
        expert_ids: Optional[Iterable[int]] = None,
        top_k: Optional[int] = None,
    ) -> V9MultiKeyRouteResult:
        per_expert, pool_expert_ids, key_scores, key_ids = self.score_matrix(
            queries, key_pool, expert_ids
        )
        num_experts = int(per_expert.shape[1])
        budget = min(int(self.config.top_k if top_k is None else top_k), num_experts)
        if budget < 1:
            raise V9MultiKeyError(
                "an aggregation row must name at least one expert (got {})".format(budget)
            )
        # Deterministic tie-break: the lower expert id wins an exact tie.
        tie = -torch.arange(
            num_experts, device=per_expert.device, dtype=per_expert.dtype
        ) * TIE_EPSILON
        _, local = torch.topk(per_expert + tie, k=budget, dim=-1, sorted=True)
        # ``distinct_expert_topk``: the topk above runs over *experts*, not keys,
        # so an expert can never occupy two slots.
        selected = pool_expert_ids[local]
        scores = torch.gather(per_expert, 1, local)
        scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))

        # Attribute each selected expert back to the key that produced its max.
        pool_tensor = pool_expert_ids
        key_expert = torch.tensor(
            [key_pool.key_records[key_id]["expert_id"] for key_id in key_ids],
            dtype=torch.long,
            device=key_scores.device,
        )
        slot_of_key = torch.searchsorted(pool_tensor, key_expert)
        best = torch.full(
            (key_scores.shape[0], pool_tensor.numel()),
            len(key_ids),
            dtype=torch.long,
            device=key_scores.device,
        )
        positions = torch.arange(
            len(key_ids), device=key_scores.device, dtype=torch.long
        ).unsqueeze(0).expand_as(key_scores)
        # A key is "the winner" when its score equals the expert's max; the first
        # such position in ``key_ids`` order wins, which is why the memory order
        # above is deterministic.
        matches_max = key_scores.eq(per_expert.index_select(1, slot_of_key))
        best.scatter_reduce_(
            1,
            slot_of_key.unsqueeze(0).expand(key_scores.shape[0], -1),
            torch.where(matches_max, positions, torch.full_like(positions, len(key_ids))),
            reduce="amin",
            include_self=True,
        )
        winning = best.gather(1, local)
        fired: List[List[Optional[str]]] = []
        for row in range(int(queries.shape[0])):
            fired.append(
                [
                    key_ids[int(winning[row, slot])]
                    if int(winning[row, slot]) < len(key_ids)
                    else None
                    for slot in range(budget)
                ]
            )
        return V9MultiKeyRouteResult(
            expert_ids=selected,
            expert_scores=scores,
            pool_expert_ids=pool_expert_ids,
            per_expert_scores=per_expert,
            key_ids=fired,
        )

    def selection(
        self,
        queries: torch.Tensor,
        sample_ids: Sequence[str],
        key_pool: V9KeyPool,
        expert_ids: Optional[Iterable[int]] = None,
        top_k: Optional[int] = None,
        gates: Optional[Dict[str, Sequence[float]]] = None,
    ):
        """The ``ComposeSelection`` a forward pass reads.

        Built through V8's :func:`~compose.v8.selection.build_selection`, so the
        inference composition convention -- uniform gates, ``normalization
        = "none"``, four slots -- is the one V8 already validated.
        """
        if len(sample_ids) != int(queries.shape[0]):
            raise V9MultiKeyError(
                "{} sample ids for {} queries".format(len(sample_ids), int(queries.shape[0]))
            )
        result = self.forward(queries, key_pool, expert_ids, top_k=top_k)
        experts: Dict[str, List[int]] = {}
        states: Dict[str, str] = {}
        for index, sample_id in enumerate(sample_ids):
            row = [int(value) for value in result.expert_ids[index].tolist()]
            sample_id = str(sample_id)
            experts[sample_id] = row
            states[sample_id] = _state_of(len(row))
        return build_selection(
            sample_ids, states, experts, device=queries.device, gates_by_sample=gates
        )


def _state_of(count: int) -> str:
    """V8's cardinality states, reused so the composition scale rule is shared."""
    if count >= 2:
        return STATE_REUSE2
    if count == 1:
        return STATE_REUSE1
    return STATE_BASE_ONLY


def aggregation_diagnostics(result: V9MultiKeyRouteResult) -> Dict[str, object]:
    """Aggregate-only diagnostics: key diversity and slot occupancy."""
    expert_ids = result.expert_ids
    counts = torch.bincount(
        expert_ids.reshape(-1), minlength=int(result.pool_expert_ids.numel())
    )
    return {
        "rows": int(expert_ids.shape[0]),
        "pool_experts": int(result.pool_expert_ids.numel()),
        "top_k": int(expert_ids.shape[1]),
        "distinct_experts": int(torch.unique(expert_ids).numel()),
        "usage": {
            str(int(expert_id)): int(count)
            for expert_id, count in zip(result.pool_expert_ids.tolist(), counts.tolist())
        },
    }


__all__ = [
    "TIE_EPSILON",
    "V9MultiKeyError",
    "V9MultiKeyRouteResult",
    "V9MultiKeyRouter",
    "aggregatable_expert_ids",
    "aggregation_diagnostics",
    "memory_key_ids",
]
