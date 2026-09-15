"""Cheap frozen-base-key recall that bounds each step's candidate set.

Putting *every* historical expert through the multi-LoRA forward would make the
cost scale with the pool, which is exactly the V8 problem V9-S exists to
remove.  Instead each sample gets a small per-step candidate set:

    C_i = Top-C historical experts  ∪  all current candidates

ranked by ``cosine(q_i, base_key_k)`` over the **frozen** base keys.  Because
both the queries and the base keys are frozen for the whole task, the historical
part of that set cannot change during the task: it is computed once at task
start and cached, so no optimizer step ever re-runs the same recall.

**Periodic wide recall replaces the fixed explorer.**  A capable historical
expert whose base key ranks ``C+1`` never enters the candidate set, never
receives answer gradient, and its key can never be corrected -- a failure that
is invisible and self-confirming.  V9 v1 held a slot open for such an expert on
*every* row: it paid the cost on every step and handed the answer an expert it
never asked for.  V9-S instead widens the recall on a fraction ``ratio`` of
optimizer steps, carrying ``wide_top_c`` historical columns instead of
``historical_top_c``.

The cache is therefore built once at the widest recall the task ever uses.  This
costs nothing extra: both rankings are the same similarity sort truncated at
different lengths, so the base Top-C *is* the first ``top_c`` columns of the
wide row.  A wide step does not rebuild anything -- it simply stops masking the
tail columns.

This cache is a **training-time recall device only**.  Test-time routing is the
V8 global multi-key aggregation over every retained key of every expert; it
never consults this file, and it never sees a task id.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from compose.adapters.types import PAD_EXPERT_ID

from .config import V9RetrievalConfig, V9WideRetrievalConfig


RETRIEVAL_SCHEMA_VERSION = 3


class V9RetrievalError(RuntimeError):
    """Raised when the retrieval cache cannot be built or trusted."""


def is_wide_step(step: int, ratio: float, seed: int = 42) -> bool:
    """Deterministic per-step wide-recall decision.

    A step-index modulo would tie the wide steps to fixed positions in the
    schedule and, at small ratios, could place every one of them inside a single
    stage.  A seeded draw per step spreads them over the whole run and is
    reproducible: every rank computes the same answer from
    ``(seed, step)`` alone, so no collective is needed to keep the ranks
    agreeing on the row width.
    """
    ratio = float(ratio)
    if ratio <= 0.0:
        return False
    if ratio >= 1.0:
        return True
    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) * 1000003 + int(step) * 101 + 7
    )
    return bool(torch.rand(1, generator=generator).item() < ratio)


def pool_fingerprint(
    historical_expert_ids: Sequence[int], base_keys: torch.Tensor
) -> str:
    """Fingerprint of the frozen recall geometry this cache was built against.

    Stored with the cache so a later task that changed the pool (or a resumed
    run pointed at the wrong directory) is detected instead of silently routing
    against a stale candidate set.
    """
    digest = hashlib.sha256()
    digest.update(RETRIEVAL_SCHEMA_VERSION.to_bytes(4, "little"))
    for expert_id in historical_expert_ids:
        digest.update(int(expert_id).to_bytes(8, "little", signed=True))
    raw = base_keys.detach().to("cpu", torch.float32).contiguous()
    digest.update(str(tuple(raw.shape)).encode("utf-8"))
    digest.update(raw.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass
class HistoricalTopC:
    """``[N, max(top_c, wide_top_c)]`` expert ids for one task's train split.

    Rows are aligned one-to-one with the dataset records, and **sorted by
    descending base-key cosine**, so the first ``top_c`` columns are the base
    Top-C and the first ``wide_top_c`` columns are the wide recall.  Slots with
    no historical expert behind them hold ``PAD_EXPERT_ID``.

    ``sample_ids`` records the id order the rows were built against, so a later
    task that loads the artefact aligns by id instead of by position.  A silent
    positional shift would hand every sample another sample's recall set, which
    would train the keys against the wrong queries without ever looking broken.
    """

    expert_ids: torch.Tensor
    top_c: int
    wide_top_c: int
    task_index: int
    fingerprint: str
    historical_expert_ids: List[int]
    sample_ids: Optional[List[str]] = None

    def aligned_rows(self, sample_ids: Sequence[str]) -> torch.Tensor:
        """Re-order the cache into the caller's id order, failing loudly.

        Returns the rows unchanged when the artefact carries no id record (an
        older cache) or the orders already agree.
        """
        wanted = [str(value) for value in sample_ids]
        if self.sample_ids is None:
            if len(wanted) != self.rows:
                raise V9RetrievalError(
                    "historical Top-C has {} rows for {} samples and records no "
                    "id order to align them by".format(self.rows, len(wanted))
                )
            return self.expert_ids
        position = {str(value): index for index, value in enumerate(self.sample_ids)}
        missing = [value for value in wanted if value not in position]
        if missing:
            raise V9RetrievalError(
                "the historical Top-C cache misses {} train samples (first: {}); "
                "rebuild it for this task's train split".format(
                    len(missing), missing[:3]
                )
            )
        index = torch.tensor(
            [position[value] for value in wanted], dtype=torch.long
        )
        return self.expert_ids.index_select(0, index)

    @property
    def slots(self) -> int:
        """Cached row width -- the widest recall this task will use."""
        return int(self.expert_ids.shape[1])

    @property
    def rows(self) -> int:
        return int(self.expert_ids.shape[0])

    def active_columns(self, wide: bool) -> int:
        """How many leading columns are live on a normal / wide step."""
        return min(int(self.wide_top_c if wide else self.top_c), self.slots)

    def active_mask(self, wide: bool, device: Optional[torch.device] = None) -> torch.Tensor:
        """``[slots]`` bool: the historical columns this step may route to.

        The masked tail is not deleted -- the row keeps its width so one
        ``ComposeSelection`` shape serves every step -- it is zeroed out of the
        gate, and a zero-gated slot is skipped by the composition entirely.
        """
        mask = torch.zeros(self.slots, dtype=torch.bool, device=device)
        mask[: self.active_columns(wide)] = True
        return mask

    def select(self, indices: torch.Tensor) -> torch.Tensor:
        return self.expert_ids.index_select(0, indices.to(self.expert_ids.device))

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(
            {
                "schema_version": RETRIEVAL_SCHEMA_VERSION,
                "kind": "v9s_historical_topc",
                "expert_ids": self.expert_ids.to(torch.int32).cpu(),
                "top_c": self.top_c,
                "wide_top_c": self.wide_top_c,
                "task_index": self.task_index,
                "fingerprint": self.fingerprint,
                "historical_expert_ids": [int(v) for v in self.historical_expert_ids],
                "sample_ids": (
                    None if self.sample_ids is None
                    else [str(v) for v in self.sample_ids]
                ),
            },
            temporary,
        )
        os.replace(temporary, target)

    @classmethod
    def load(cls, path: str) -> "HistoricalTopC":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("kind") != "v9s_historical_topc":
            raise V9RetrievalError(f"{path} is not a V9-S historical Top-C cache")
        if int(payload.get("schema_version", -1)) != RETRIEVAL_SCHEMA_VERSION:
            raise V9RetrievalError(f"{path} has an incompatible schema version")
        sample_ids = payload.get("sample_ids")
        return cls(
            expert_ids=payload["expert_ids"].to(torch.long),
            top_c=int(payload["top_c"]),
            wide_top_c=int(payload["wide_top_c"]),
            task_index=int(payload["task_index"]),
            fingerprint=str(payload["fingerprint"]),
            historical_expert_ids=[int(v) for v in payload["historical_expert_ids"]],
            sample_ids=None if sample_ids is None else [str(v) for v in sample_ids],
        )


def build_historical_topc(
    queries: torch.Tensor,
    historical_expert_ids: Sequence[int],
    base_keys: torch.Tensor,
    config: V9RetrievalConfig,
    wide: Optional[V9WideRetrievalConfig] = None,
    task_index: int = 0,
    seed: int = 42,
    sample_ids: Optional[Sequence[str]] = None,
    memory_key_expert_ids: Optional[Sequence[int]] = None,
) -> HistoricalTopC:
    """Rank frozen retained keys against the fixed queries, once per task.

    ``queries`` is ``[N, 1536]`` and ``base_keys`` is ``[H, 1536]`` with rows in
    the order of ``historical_expert_ids``.  Both are L2-normalised here, so the
    inner product *is* the cosine similarity the recall is defined on.
    """
    history = [int(value) for value in historical_expert_ids]
    ids = None if sample_ids is None else [str(value) for value in sample_ids]
    if ids is not None and len(ids) != int(queries.shape[0]):
        raise V9RetrievalError(
            "{} sample ids recorded for {} queries".format(
                len(ids), int(queries.shape[0])
            )
        )
    top_c = min(int(config.top_c), len(history))
    wide_top_c = top_c
    if wide is not None and wide.enabled:
        wide_top_c = min(max(int(wide.top_c), top_c), len(history))
    slots = max(top_c, wide_top_c)
    rows = int(queries.shape[0])
    expert_ids = torch.full((rows, slots), PAD_EXPERT_ID, dtype=torch.long)
    if slots == 0:
        return HistoricalTopC(
            expert_ids=expert_ids,
            top_c=0,
            wide_top_c=0,
            task_index=int(task_index),
            fingerprint=pool_fingerprint(history, base_keys),
            historical_expert_ids=history,
            sample_ids=ids,
        )
    query_matrix = F.normalize(queries.detach().float().reshape(rows, -1), dim=-1)
    key_matrix = F.normalize(base_keys.detach().float().reshape(base_keys.shape[0], -1), dim=-1)
    owners = list(history if memory_key_expert_ids is None else memory_key_expert_ids)
    if key_matrix.shape[0] != len(owners):
        raise V9RetrievalError("frozen key matrix and owner list have different lengths")
    positions = {expert_id: index for index, expert_id in enumerate(history)}
    if any(int(owner) not in positions for owner in owners):
        raise V9RetrievalError("frozen recall key belongs to a non-historical expert")
    similarity = query_matrix @ key_matrix.T
    owner_columns = torch.tensor([positions[int(owner)] for owner in owners], dtype=torch.long)
    expert_scores = torch.full((rows, len(history)), float("-inf"), dtype=similarity.dtype)
    expert_scores.scatter_reduce_(1, owner_columns.unsqueeze(0).expand(rows, -1), similarity, reduce="amax", include_self=True)
    order = expert_scores.argsort(dim=1, descending=True, stable=True)
    history_tensor = torch.tensor(history, dtype=torch.long)
    ranked = history_tensor.index_select(0, order.reshape(-1)).reshape(rows, -1)
    expert_ids[:, :slots] = ranked[:, :slots]
    return HistoricalTopC(
        expert_ids=expert_ids,
        top_c=top_c,
        wide_top_c=wide_top_c,
        task_index=int(task_index),
        fingerprint=pool_fingerprint(history, base_keys),
        historical_expert_ids=history,
        sample_ids=ids,
    )


def load_or_build_historical_topc(
    cache_path: str,
    queries: torch.Tensor,
    historical_expert_ids: Sequence[int],
    base_keys: torch.Tensor,
    config: V9RetrievalConfig,
    wide: Optional[V9WideRetrievalConfig] = None,
    task_index: int = 0,
    seed: int = 42,
    sample_ids: Optional[Sequence[str]] = None,
    memory_key_expert_ids: Optional[Sequence[int]] = None,
) -> HistoricalTopC:
    """Return the task's historical recall set, building it at most once.

    ``cache=False`` forces a rebuild, which is the escape hatch for a
    deliberate re-derivation; the default path reuses the artefact so a resumed
    run reproduces the same candidate sets it trained with.
    """
    history = [int(value) for value in historical_expert_ids]
    ids = None if sample_ids is None else [str(value) for value in sample_ids]
    expected_top_c = min(int(config.top_c), len(history))
    expected_wide = expected_top_c
    if wide is not None and wide.enabled:
        expected_wide = min(max(int(wide.top_c), expected_top_c), len(history))
    fingerprint = pool_fingerprint(history, base_keys)
    if config.cache and os.path.isfile(cache_path):
        cached = HistoricalTopC.load(cache_path)
        if (
            cached.fingerprint == fingerprint
            and cached.task_index == int(task_index)
            # A changed Top-C or wide width is a different recipe, not a stale
            # file: fall through and rebuild rather than silently training
            # against the previous task's geometry.
            and cached.top_c == expected_top_c
            and cached.wide_top_c == expected_wide
        ):
            if cached.rows != int(queries.shape[0]):
                raise V9RetrievalError(
                    "cached historical Top-C has {} rows for {} queries; the "
                    "train split changed without the pool changing".format(
                        cached.rows, int(queries.shape[0])
                    )
                )
            if ids is not None and cached.sample_ids is not None:
                if set(ids) != set(cached.sample_ids):
                    raise V9RetrievalError(
                        "cached historical Top-C was built for a different train "
                        "split of the same size; rebuild it for this task"
                    )
            return cached
    topc = build_historical_topc(
        queries=queries,
        historical_expert_ids=history,
        base_keys=base_keys,
        config=config,
        wide=wide,
        task_index=task_index,
        seed=seed,
        sample_ids=ids,
        memory_key_expert_ids=memory_key_expert_ids,
    )
    if config.cache:
        topc.save(cache_path)
    return topc


def compose_candidate_rows(
    historical_rows: torch.Tensor,
    candidate_ids: Sequence[int],
) -> torch.Tensor:
    """Concatenate the per-sample historical block with the shared candidate block.

    Every row gets the *same* candidate ids in the same slots, so a V9-S routing
    row is dense: only the historical half varies per sample.  That is what
    makes one ``[B, historical_slots + M]`` selection legal without per-row
    padding.

    The row width follows the width of ``historical_rows`` rather than the
    configured maximum, because the retrieval builder clamps the recall to the
    number of historical experts that actually exist -- on task 0 the
    historical block is empty and the row is just the candidate block.
    """
    rows = int(historical_rows.shape[0])
    candidate_tensor = torch.tensor(
        [int(value) for value in candidate_ids], dtype=torch.long,
        device=historical_rows.device,
    ).unsqueeze(0).expand(rows, -1)
    return torch.cat([historical_rows, candidate_tensor], dim=1)


def retrieval_diagnostics(topc: HistoricalTopC) -> Dict[str, Any]:
    """Recall health: how much of the pool the Top-C actually reaches.

    ``distinct_in_topc`` near the number of rows means the recall has collapsed
    onto a handful of historical experts, which is the failure mode wide recall
    exists to hold open -- a collapsed Top-C trains the same few keys forever
    and never shows up as a loss.
    """
    base = topc.expert_ids[:, : topc.active_columns(False)]
    wide = topc.expert_ids[:, : topc.active_columns(True)]
    base_valid = base[base != PAD_EXPERT_ID]
    wide_valid = wide[wide != PAD_EXPERT_ID]
    return {
        "rows": topc.rows,
        "top_c": topc.top_c,
        "wide_top_c": topc.wide_top_c,
        "historical_slots": topc.slots,
        "distinct_in_topc": int(torch.unique(base_valid).numel()),
        "distinct_in_wide_recall": int(torch.unique(wide_valid).numel()),
        "topc_histogram": {
            str(int(value)): int((base_valid == value).sum().item())
            for value in torch.unique(base_valid)
        },
        #: Experts the wide recall reaches that the base Top-C never does --
        #: the population the old fixed explorer was holding a slot for.
        "wide_only_experts": sorted(
            {
                int(value)
                for value in torch.unique(wide_valid)
                if int(value) not in set(int(v) for v in torch.unique(base_valid))
            }
        ),
        "rows_without_history": int(
            (base == PAD_EXPERT_ID).all(dim=1).sum().item()
        ),
        "fingerprint": topc.fingerprint,
    }


__all__ = [
    "HistoricalTopC",
    "RETRIEVAL_SCHEMA_VERSION",
    "V9RetrievalError",
    "build_historical_topc",
    "compose_candidate_rows",
    "is_wide_step",
    "load_or_build_historical_topc",
    "pool_fingerprint",
    "retrieval_diagnostics",
]
