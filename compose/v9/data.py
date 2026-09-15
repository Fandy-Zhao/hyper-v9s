"""V9 data plumbing: the fixed query plus the task's cached historical recall.

Two caches, both frozen for the whole task, both built once:

* the **fixed query** ``q_i = L2Norm(concat(LN(z_visual), LN(z_text)))`` -- the
  V7 query cache, reused verbatim (spec §4).  Queries are never re-extracted per
  epoch, and the frozen visual/text encoders never run again for a sample that
  already has one.
* the **historical Top-C** block -- ``[N, max(Top-C, wide Top-C)]`` expert ids,
  one row per training sample, computed from the frozen base keys and the fixed
  queries at task start (spec §6).  Current candidates are *not* stored here:
  they are the same for every row and are appended per step, so a candidate key
  changing during training never invalidates this cache.

Neither cache is consulted at test time.  Inference routes over the global
multi-key geometry (spec §20) and never looks at a training recall file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from compose.adapters.types import PAD_EXPERT_ID
from compose.train.data import V7QueryCollator, V7QueryDataset

from .config import V9Config
from .keys import V9KeyPool
from .retrieval import (
    HistoricalTopC,
    V9RetrievalError,
    load_or_build_historical_topc,
    retrieval_diagnostics,
)


class V9DataError(RuntimeError):
    """Raised when a V9 batch would be built from mismatched caches."""


class V9QueryDataset(V7QueryDataset):
    """Full train split joined with the fixed query *and* the task's Top-C.

    The historical block is stored as one ``[N, slots]`` index tensor and sliced
    per item, so the recall cost per sample is one row copy, not a lookup.

    Rows are aligned through sample ids, not positions: the Top-C artefact
    records the id order it was built against, and any mismatch is a hard error.
    A silent positional shift would give every sample another sample's recall
    set, which would train the keys against the wrong queries without ever
    looking broken.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        data_args,
        query_cache,
        query_tensor=None,
        historical_topc: Optional[HistoricalTopC] = None,
    ) -> None:
        super().__init__(
            data_path, tokenizer, data_args, query_cache, query_tensor=query_tensor
        )
        # The parent's ids, not a second derivation of them: the historical
        # block has to be aligned to the same sample ids the query cache was
        # addressed by, and two rules for "which sample is this" is one rule
        # too many.
        self.historical_topc = historical_topc
        self._historical_rows: Optional[torch.Tensor] = None
        if historical_topc is None:
            return
        try:
            self._historical_rows = historical_topc.aligned_rows(self.dataset_ids)
        except V9RetrievalError as error:
            raise V9DataError(str(error)) from error

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self._historical_rows is not None:
            item["historical_topc"] = self._historical_rows[index]
        return item


class V9QueryCollator(V7QueryCollator):
    """V7 collator plus the per-sample historical recall row."""

    def __call__(self, instances):
        rows = (
            torch.stack([instance["historical_topc"] for instance in instances])
            if "historical_topc" in instances[0]
            else None
        )
        stripped = [
            {
                key: value
                for key, value in instance.items()
                if key != "historical_topc"
            }
            for instance in instances
        ]
        batch = super().__call__(stripped)
        if rows is not None:
            batch["historical_topc"] = rows
        return batch


@dataclass
class V9TaskData:
    """Everything the trainer needs for one task, resolved from disk."""

    dataset: V9QueryDataset
    collator: V9QueryCollator
    query_fingerprint: Optional[str]
    retrieval_fingerprint: Optional[str]
    retrieval_diagnostics: Dict[str, Any]

    def as_data_module(self) -> Dict[str, Any]:
        return {"train_dataset": self.dataset, "data_collator": self.collator}


def build_task_retrieval(
    key_pool: V9KeyPool,
    queries: torch.Tensor,
    sample_ids: Sequence[str],
    config: V9Config,
    task_index: int,
    cache_path: str,
    force_build: bool = False,
) -> HistoricalTopC:
    """Load or build the task's historical recall set (spec §6).

    Costs one ``[N, 1536] @ [1536, H]`` product per task, never per step.

    ``force_build`` ignores any file already at ``cache_path``.  It exists for
    splits other than the training split: a validation recall set written to a
    path that happens to hold the training rows would be structurally plausible
    and silently wrong.

    It rebuilds *into* ``cache_path``; it does not skip writing it.  The rebuild
    is a ``cache=False`` in-memory derivation, so the write has to be explicit
    here -- and it goes to ``cache_path``, which is the path the caller is about
    to record in the manifest and hand to the training process.  Building into
    ``os.devnull`` and stopping there left that process loading a path nothing
    had created, half an hour later.  An explicit cache path is a request for a
    cache.
    """
    retrieval = config.historical_retrieval
    if force_build:
        retrieval = replace(retrieval, cache=False)
    historical_ids = key_pool.historical_ids
    base_keys = (
        key_pool.base_key_matrix(historical_ids)
        if historical_ids
        else torch.zeros(0, key_pool.query_dim, dtype=torch.float32)
    )
    topc = load_or_build_historical_topc(
        cache_path=cache_path,
        queries=queries,
        historical_expert_ids=historical_ids,
        base_keys=base_keys,
        config=retrieval,
        wide=config.wide_retrieval,
        task_index=task_index,
        seed=config.key.candidate_init_seed,
        sample_ids=sample_ids,
    )
    if force_build:
        topc.save(cache_path)
    return topc


def write_retrieval_manifest(
    path: str,
    topc: HistoricalTopC,
    sample_ids: Sequence[str],
    diagnostics: Optional[Dict[str, Any]] = None,
) -> None:
    """Record the recall geometry a task trained against, for the audit trail."""
    payload = {
        "kind": "v9_retrieval_manifest",
        "task_index": int(topc.task_index),
        "top_c": int(topc.top_c),
        "wide_top_c": int(topc.wide_top_c),
        "slots": int(topc.slots),
        "rows": int(topc.rows),
        "historical_expert_ids": [int(v) for v in topc.historical_expert_ids],
        "fingerprint": topc.fingerprint,
        "sample_ids_sha256": _sample_id_digest(sample_ids),
        "diagnostics": dict(diagnostics or {}),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)


def _sample_id_digest(sample_ids: Sequence[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def candidate_rows_preview(
    topc: Optional[HistoricalTopC], candidate_ids: Sequence[int], rows: int = 3
) -> List[List[int]]:
    """A few routing rows for the startup log -- never a per-sample dump."""
    preview: List[List[int]] = []
    for row in range(min(int(rows), 0 if topc is None else topc.rows)):
        historical = [
            int(value)
            for value in topc.expert_ids[row].tolist()
            if int(value) != PAD_EXPERT_ID
        ]
        preview.append(historical + [int(value) for value in candidate_ids])
    if not preview:
        preview.append([int(value) for value in candidate_ids])
    return preview


__all__ = [
    "V9DataError",
    "V9QueryCollator",
    "V9QueryDataset",
    "V9TaskData",
    "build_task_retrieval",
    "candidate_rows_preview",
    "retrieval_diagnostics",
    "write_retrieval_manifest",
]
