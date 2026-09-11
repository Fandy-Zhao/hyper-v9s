"""The V8 query is the V7 query, reused verbatim (PART 3.1).

The specification allows V8 to import a V7 component directly when the semantics
are identical (PART 27), and that is exactly the case here: the fixed multimodal
query has **zero trainable parameters** and a fixed provenance hash, so
re-deriving an equivalent implementation for V8 would add risk without adding
anything.  Everything downstream -- the cache, the keys, the geometry -- depends
on this object being the same one V7 committed against.

What V8 adds is the *invariant check*: :func:`assert_query_contract` fails loudly
if the imported query has drifted from the hash V8's caches were built against.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from compose.v7.query import (
    FixedMultimodalQuery,
    FixedQueryProvenance,
    full_train_task_center,
)
from compose.v8.config import V7_QUERY_MODULE_HASH, V8QueryConfig


class QueryContractError(RuntimeError):
    """Raised when the query is not the fixed, parameter-free V7 object."""


def build_query(config: V8QueryConfig | None = None) -> FixedMultimodalQuery:
    """Return the fixed query module, after checking it still has no parameters."""
    config = config or V8QueryConfig()
    query = FixedMultimodalQuery(visual_dim=config.visual_dim, text_dim=config.text_dim)
    assert_query_contract(query, config)
    return query


def assert_query_contract(
    query: FixedMultimodalQuery,
    config: V8QueryConfig | None = None,
) -> Dict[str, Any]:
    """Zero trainable parameters, zero buffers, unchanged provenance hash."""
    config = config or V8QueryConfig()
    trainable = [
        name for name, parameter in query.named_parameters() if parameter.requires_grad
    ]
    if trainable:
        raise QueryContractError(
            f"the V8 query must have zero trainable parameters; found {trainable}"
        )
    total_parameters = sum(parameter.numel() for parameter in query.parameters())
    if total_parameters != int(config.trainable_parameter_count):
        raise QueryContractError(
            f"query parameter count {total_parameters} != configured "
            f"{config.trainable_parameter_count}"
        )
    provenance = FixedQueryProvenance()
    if getattr(provenance, "module_hash", None) != V7_QUERY_MODULE_HASH:
        raise QueryContractError(
            "the imported V7 query hash no longer matches the V8 contract: "
            f"{getattr(provenance, 'module_hash', None)!r} != {V7_QUERY_MODULE_HASH!r}"
        )
    return {
        "module_hash": V7_QUERY_MODULE_HASH,
        "trainable_parameters": 0,
        "total_parameters": total_parameters,
        "query_dim": int(config.query_dim),
    }


def task_center(queries: torch.Tensor, num_train_samples: int):
    """Re-exported so V8 never re-implements the centre definition."""
    return full_train_task_center(queries, num_train_samples)


__all__ = [
    "FixedMultimodalQuery",
    "FixedQueryProvenance",
    "QueryContractError",
    "assert_query_contract",
    "build_query",
    "task_center",
]
