"""V9 test-time routing: queries in, experts out, nothing else (spec §20).

The deployment rule, unchanged from V8:

    expert_score(i, k) = max over k's retained keys of cos(q_i, e)
    -> global multi-key aggregation
    -> Top-M / Top-C
    -> Top-1 / Top-2
    -> RMS-calibrated LoRA composition
    -> generate the response

Three things are structurally absent from this module, not merely unused:

* **the response text** -- there is no parameter, buffer or input through which
  a label could enter;
* **the task index** -- which task a sample came from is not an argument of any
  function here, so a key cannot be chosen by task identity;
* **training recall state** -- the per-task historical Top-C cache and the
  exploration slot are training devices; routing here ranges over the pool's own
  keys.

:func:`assert_v9_inference_purity` scans this file's executable code for
supervision identifiers and is run as an acceptance check, so the property is
machine-verified rather than asserted in prose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from compose.v8.config import V8RoutingConfig
from compose.v8.inference import assert_inference_purity

from .keys import V9KeyPool
from .multi_key import V9MultiKeyRouteResult, V9MultiKeyRouter


class V9InferenceError(RuntimeError):
    """Raised when test-time routing is asked for something it must not read."""


class V9InferenceRouter(nn.Module):
    """Query-only router over the committed expert pool.

    Holds no supervision of any kind.  ``expert_ids`` narrows the routing pool;
    the deployment default is every live expert, and the task-end audit passes
    an explicit list to ask the counterfactual question about a candidate that
    has not been committed yet.
    """

    def __init__(
        self,
        key_pool: V9KeyPool,
        routing_config: Optional[V8RoutingConfig] = None,
        expert_ids: Optional[Iterable[int]] = None,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.key_pool = key_pool
        self.router = V9MultiKeyRouter(routing_config)
        self.expert_ids = (
            None if expert_ids is None else tuple(int(v) for v in expert_ids)
        )
        self.top_k = None if top_k is None else int(top_k)

    def forward(self, queries: torch.Tensor) -> V9MultiKeyRouteResult:
        return self.router(
            queries, self.key_pool, self.expert_ids, top_k=self.top_k
        )

    def selection(self, queries: torch.Tensor, sample_ids: Sequence[str]):
        if len(sample_ids) != int(queries.shape[0]):
            raise V9InferenceError(
                "{} sample ids for {} queries".format(
                    len(sample_ids), int(queries.shape[0])
                )
            )
        return self.router.selection(
            queries,
            sample_ids,
            self.key_pool,
            expert_ids=self.expert_ids,
            top_k=self.top_k,
        )

    def route_policy(self, queries: torch.Tensor) -> Dict[str, Any]:
        """The persisted deployment policy: expert ids, scores, winning keys."""
        result = self.forward(queries)
        return {
            "rows": [
                {
                    "expert_ids": [
                        int(value) for value in result.expert_ids[index].tolist()
                    ],
                    "scores": [
                        float(value) for value in result.expert_scores[index].tolist()
                    ],
                    "key_ids": [
                        (None if key_id is None else str(key_id))
                        for key_id in result.key_ids[index]
                    ],
                }
                for index in range(result.rows)
            ],
            "policy": "v9_effective_key_cosine_topk",
        }


def validate_inference_policy(
    key_pool: V9KeyPool, policy: Mapping[str, Any]
) -> None:
    """A deployment policy may only name experts that exist and are live."""
    live = set(key_pool.live_expert_ids())
    for index, row in enumerate(policy.get("rows", [])):
        for expert_id in row.get("expert_ids", []):
            if int(expert_id) not in live:
                raise V9InferenceError(
                    "policy row {} names expert {}, which is not a live expert "
                    "of this pool".format(index, expert_id)
                )


def assert_v9_inference_purity(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Scan this module's executable code for supervision references."""
    return assert_inference_purity(
        Path(path) if path is not None else Path(__file__)
    )


__all__ = [
    "V9InferenceError",
    "V9InferenceRouter",
    "assert_v9_inference_purity",
    "validate_inference_policy",
]
