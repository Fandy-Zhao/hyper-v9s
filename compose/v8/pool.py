"""Multi-Key Expert Pool: ``1 Expert : N Keys``.

V7 stores one 1536-D routing key per expert inside a single ``nn.ParameterDict``
keyed by ``str(expert_id)`` (``compose/v7/pool.py``).  That structure cannot
express the V8 hypothesis -- *one historical expert re-recalled by several later
task distributions* -- because the expert id and the key id are the same object.

V8 splits the two identities and keeps the **expert** as the unit of capability:

* an **expert record** (LoRA adapter + RMS calibration + lifecycle), and
* a **key record** (``expert_id``, ``task_id``, ``key_type``, lifecycle, bookkeeping).

Every expert keeps exactly one ``origin`` key -- the V7 key, migrated with an
identical tensor -- and gains at most one ``task_alias`` key per later task, created
lazily and only when the answer-supervised teacher found that task's samples which
this expert genuinely solves.  Historical experts and their committed keys are
frozen: only current-task alias keys are trainable.

Nothing here re-derives the query, re-normalizes a migrated key, or trains an
old expert.  The migration is a pure re-labelling of V7's own bytes.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from compose.v8.config import (
    KEY_TYPE_ORIGIN,
    KEY_TYPE_TASK_ALIAS,
    KEY_TYPES,
)


class MultiKeyPoolError(RuntimeError):
    """Raised when a pool invariant would be violated."""


LIFECYCLE_CANDIDATE = "candidate"
LIFECYCLE_HISTORICAL = "historical"
LIFECYCLE_PRUNED = "pruned"
LIFECYCLES = (LIFECYCLE_CANDIDATE, LIFECYCLE_HISTORICAL, LIFECYCLE_PRUNED)


def tensor_checksum(value: torch.Tensor) -> str:
    """sha256 over the raw tensor payload (detached, contiguous, cpu)."""
    raw = value.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(raw.shape)).encode("utf-8"))
    digest.update(str(raw.dtype).encode("utf-8"))
    # NumPy has no native bfloat16 dtype.  Viewing the contiguous payload as
    # bytes preserves the exact representation for every torch dtype and keeps
    # the checksum independent of any lossy dtype conversion.
    digest.update(raw.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_unit_key(key_id: str) -> None:
    if "." in key_id:
        raise MultiKeyPoolError(
            f"key id {key_id!r} contains '.', which nn.ParameterDict forbids"
        )


class MultiKeyExpertPool(nn.Module):
    """Expert catalogue + per-expert multi-key store.

    ``self.keys`` is an ``nn.ParameterDict`` mapping ``key_id -> (query_dim,)``.
    ``key_records[key_id]`` carries everything the router, the key loss and the
    commit step need to know about *why that key exists*.
    """

    def __init__(self, query_dim: int = 1536) -> None:
        super().__init__()
        if query_dim != 1536:
            raise MultiKeyPoolError("V8 keeps the 1536-D V7 query space")
        self.query_dim = int(query_dim)
        self.keys: nn.ParameterDict = nn.ParameterDict()
        self.expert_records: Dict[int, Dict[str, Any]] = {}
        self.key_records: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # ids
    # ------------------------------------------------------------------
    @staticmethod
    def key_id_for(expert_id: int, task_id: int, key_type: str) -> str:
        if key_type not in KEY_TYPES:
            raise MultiKeyPoolError(f"unknown key type {key_type!r}")
        key_id = f"e{int(expert_id)}_t{int(task_id)}_{key_type}"
        _validate_unit_key(key_id)
        return key_id

    @staticmethod
    def parse_key_id(key_id: str) -> Tuple[int, int, str]:
        parts = key_id.split("_")
        if len(parts) != 4 or not parts[0].startswith("e") or not parts[1].startswith("t"):
            raise MultiKeyPoolError(f"malformed key id {key_id!r}")
        expert_id = int(parts[0][1:])
        task_id = int(parts[1][1:])
        key_type = "_".join(parts[2:])
        if key_type not in KEY_TYPES:
            raise MultiKeyPoolError(f"unknown key type in {key_id!r}")
        return expert_id, task_id, key_type

    # ------------------------------------------------------------------
    # experts
    # ------------------------------------------------------------------
    def add_expert(
        self,
        expert_id: int,
        origin_task: int,
        creation_task: Optional[int] = None,
        lora_path: Optional[str] = None,
        rms_state: Optional[Dict[str, Any]] = None,
        lifecycle: str = LIFECYCLE_CANDIDATE,
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        expert_id = int(expert_id)
        if expert_id in self.expert_records:
            raise MultiKeyPoolError(f"duplicate expert id {expert_id}")
        if lifecycle not in LIFECYCLES:
            raise MultiKeyPoolError(f"unknown lifecycle {lifecycle!r}")
        self.expert_records[expert_id] = {
            "expert_id": expert_id,
            "origin_task": int(origin_task),
            "creation_task": int(origin_task if creation_task is None else creation_task),
            "lora_path": lora_path,
            "rms_state": rms_state,
            "lifecycle": lifecycle,
            "trainable": False,
            "extra": dict(extra or {}),
        }
        return expert_id

    def expert_ids(self, lifecycle: Optional[str] = None) -> List[int]:
        ids = [
            expert_id
            for expert_id, record in self.expert_records.items()
            if lifecycle is None or record["lifecycle"] == lifecycle
        ]
        return sorted(ids)

    def live_expert_ids(self) -> List[int]:
        """Experts that may be routed to: everything except pruned/candidate."""
        return [
            expert_id
            for expert_id in self.expert_ids()
            if self.expert_records[expert_id]["lifecycle"] != LIFECYCLE_PRUNED
        ]

    def expert_record(self, expert_id: int) -> Dict[str, Any]:
        try:
            return self.expert_records[int(expert_id)]
        except KeyError as exc:  # pragma: no cover - defensive
            raise MultiKeyPoolError(f"unknown expert {expert_id}") from exc

    # ------------------------------------------------------------------
    # keys
    # ------------------------------------------------------------------
    def add_key(
        self,
        expert_id: int,
        task_id: int,
        key_type: str,
        value: torch.Tensor,
        lifecycle: str = LIFECYCLE_CANDIDATE,
        trainable: bool = False,
        support_count: int = 0,
        teacher_gain: float = 0.0,
        validation_gain: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        expert_id = int(expert_id)
        task_id = int(task_id)
        if expert_id not in self.expert_records:
            raise MultiKeyPoolError(f"key for unknown expert {expert_id}")
        if key_type not in KEY_TYPES:
            raise MultiKeyPoolError(f"unknown key type {key_type!r}")
        key_id = self.key_id_for(expert_id, task_id, key_type)
        if key_id in self.key_records:
            raise MultiKeyPoolError(f"duplicate key {key_id}")
        value = value.detach().to(dtype=torch.float32).reshape(-1)
        if value.numel() != self.query_dim:
            raise MultiKeyPoolError(
                f"key {key_id} has dim {value.numel()}, expected {self.query_dim}"
            )
        parameter = nn.Parameter(value.clone(), requires_grad=bool(trainable))
        self.keys[key_id] = parameter
        self.key_records[key_id] = {
            "key_id": key_id,
            "expert_id": expert_id,
            "task_id": task_id,
            "key_type": key_type,
            "lifecycle": lifecycle,
            "trainable": bool(trainable),
            "support_count": int(support_count),
            "teacher_gain": float(teacher_gain),
            "validation_gain": float(validation_gain),
            "extra": dict(extra or {}),
        }
        return key_id

    def key_record(self, key_id: str) -> Dict[str, Any]:
        try:
            return self.key_records[key_id]
        except KeyError as exc:
            raise MultiKeyPoolError(f"unknown key {key_id}") from exc

    def key_ids(
        self,
        expert_id: Optional[int] = None,
        task_id: Optional[int] = None,
        key_type: Optional[str] = None,
        lifecycle: Optional[str] = None,
    ) -> List[str]:
        ids = []
        for key_id, record in self.key_records.items():
            if expert_id is not None and record["expert_id"] != int(expert_id):
                continue
            if task_id is not None and record["task_id"] != int(task_id):
                continue
            if key_type is not None and record["key_type"] != key_type:
                continue
            if lifecycle is not None and record["lifecycle"] != lifecycle:
                continue
            ids.append(key_id)
        return sorted(ids, key=lambda k: (self.key_records[k]["expert_id"],
                                          self.key_records[k]["task_id"],
                                          self.key_records[k]["key_type"]))

    def origin_key_id(self, expert_id: int) -> str:
        record = self.expert_record(expert_id)
        return self.key_id_for(expert_id, record["origin_task"], KEY_TYPE_ORIGIN)

    def alias_key_id(self, expert_id: int, task_id: int) -> str:
        return self.key_id_for(expert_id, task_id, KEY_TYPE_TASK_ALIAS)

    def has_alias(self, expert_id: int, task_id: int) -> bool:
        return self.alias_key_id(expert_id, task_id) in self.key_records

    def active_key_ids(self, excluded_experts: Iterable[int] = ()) -> List[str]:
        """Keys eligible for routing: live lifecycles, excluding hidden experts."""
        excluded = {int(value) for value in excluded_experts}
        active = []
        for key_id in self.key_ids():
            record = self.key_records[key_id]
            if record["lifecycle"] == LIFECYCLE_PRUNED:
                continue
            if record["expert_id"] in excluded:
                continue
            if self.expert_records[record["expert_id"]]["lifecycle"] == LIFECYCLE_PRUNED:
                continue
            active.append(key_id)
        return active

    def active_expert_ids(self, excluded_experts: Iterable[int] = ()) -> List[int]:
        excluded = {int(value) for value in excluded_experts}
        return [
            expert_id
            for expert_id in self.live_expert_ids()
            if expert_id not in excluded
            and self.active_key_ids_for_expert(expert_id)
        ]

    def active_key_ids_for_expert(self, expert_id: int) -> List[str]:
        return [
            key_id
            for key_id in self.key_ids(expert_id=int(expert_id))
            if self.key_records[key_id]["lifecycle"] != LIFECYCLE_PRUNED
        ]

    # ------------------------------------------------------------------
    # tensors
    # ------------------------------------------------------------------
    def raw_matrix(self, key_ids: Sequence[str]) -> torch.Tensor:
        if not key_ids:
            return torch.zeros(0, self.query_dim)
        return torch.stack([self.keys[key_id] for key_id in key_ids], dim=0)

    def normalized(self, key_ids: Sequence[str]) -> torch.Tensor:
        return F.normalize(self.raw_matrix(key_ids).detach().float(), dim=-1)

    def expert_index(self, excluded_experts: Iterable[int] = ()):
        """``(key_ids, expert_ids_tensor, stacked_normalized_keys)`` for routing."""
        key_ids = self.active_key_ids(excluded_experts)
        expert_ids = [self.key_records[key_id]["expert_id"] for key_id in key_ids]
        return key_ids, torch.tensor(expert_ids, dtype=torch.long), self.normalized(key_ids)

    def pairwise_key_cosine(self, key_ids: Optional[Sequence[str]] = None) -> Tuple[List[str], torch.Tensor]:
        ids = list(key_ids) if key_ids is not None else self.key_ids()
        matrix = self.normalized(ids)
        if matrix.shape[0] == 0:
            return ids, torch.zeros(0, 0)
        return ids, matrix @ matrix.T

    def alias_redundancy(self, expert_id: int, task_id: int) -> float:
        """Cosine of the (would-be) alias key against the expert's other keys."""
        alias_id = self.alias_key_id(expert_id, task_id)
        others = [
            key_id
            for key_id in self.key_ids(expert_id=int(expert_id))
            if key_id != alias_id and self.key_records[key_id]["lifecycle"] != LIFECYCLE_PRUNED
        ]
        if not others or alias_id not in self.key_records:
            return 0.0
        alias = F.normalize(self.keys[alias_id].detach().float().reshape(1, -1), dim=-1)
        other = self.normalized(others)
        return float((alias @ other.T).max().item())

    # ------------------------------------------------------------------
    # freezing
    # ------------------------------------------------------------------
    def set_key_trainable(self, key_id: str, trainable: bool) -> None:
        record = self.key_record(key_id)
        record["trainable"] = bool(trainable)
        self.keys[key_id].requires_grad = bool(trainable)

    def set_key_lifecycle(self, key_id: str, lifecycle: str) -> None:
        """Move a key between lifecycles (e.g. retire a redundant alias key).

        Pruning is a routing decision, not a training one, so this deliberately
        does not touch ``requires_grad``; callers that want both should call
        :meth:`set_key_trainable` as well.
        """
        if lifecycle not in LIFECYCLES:
            raise MultiKeyPoolError(f"unknown lifecycle {lifecycle!r}")
        self.key_record(key_id)["lifecycle"] = str(lifecycle)

    def freeze_historical(self, current_task: Optional[int] = None) -> List[str]:
        """Freeze every key that does not belong to the current task.

        With ``current_task=None`` everything is frozen (evaluation mode).
        Returns the ids that were frozen.
        """
        frozen = []
        for key_id, record in self.key_records.items():
            if current_task is not None and record["task_id"] == int(current_task):
                continue
            if record["trainable"] or self.keys[key_id].requires_grad:
                self.set_key_trainable(key_id, False)
            frozen.append(key_id)
        return sorted(frozen)

    def freeze_all(self) -> List[str]:
        return self.freeze_historical(current_task=None)

    def trainable_key_ids(self) -> List[str]:
        return sorted(
            key_id
            for key_id, record in self.key_records.items()
            if record["trainable"] and self.keys[key_id].requires_grad
        )

    def historical_key_ids(self) -> List[str]:
        return sorted(
            key_id
            for key_id, record in self.key_records.items()
            if record["key_type"] == KEY_TYPE_ORIGIN
            or record["lifecycle"] == LIFECYCLE_HISTORICAL
        )

    # ------------------------------------------------------------------
    # integrity
    # ------------------------------------------------------------------
    def historical_checksums(self) -> Dict[str, str]:
        return {key_id: tensor_checksum(self.keys[key_id])
                for key_id in self.historical_key_ids()}

    def key_checksums(self) -> Dict[str, str]:
        return {key_id: tensor_checksum(self.keys[key_id]) for key_id in self.key_ids()}

    def audit(self) -> Dict[str, Any]:
        experts = self.expert_ids()
        origins = [self.origin_key_id(expert_id) for expert_id in experts]
        aliases = self.key_ids(key_type=KEY_TYPE_TASK_ALIAS)
        return {
            "query_dim": self.query_dim,
            "num_experts": len(experts),
            "num_live_experts": len(self.live_expert_ids()),
            "num_keys": len(self.key_records),
            "num_origin_keys": len(origins),
            "num_alias_keys": len(aliases),
            "num_trainable_keys": len(self.trainable_key_ids()),
            "experts_with_multiple_keys": sorted(
                expert_id for expert_id in experts
                if len(self.active_key_ids_for_expert(expert_id)) > 1
            ),
            "alias_keys_per_task": {
                str(task_id): len(self.key_ids(task_id=task_id, key_type=KEY_TYPE_TASK_ALIAS))
                for task_id in sorted({record["task_id"] for record in self.key_records.values()})
            },
            "lifecycle_counts": {
                state: sum(1 for record in self.expert_records.values()
                           if record["lifecycle"] == state)
                for state in LIFECYCLES
            },
            "key_lifecycle_counts": {
                state: sum(1 for record in self.key_records.values()
                           if record["lifecycle"] == state)
                for state in LIFECYCLES
            },
        }

    def validate(self) -> Dict[str, Any]:
        """Hard invariants; raises instead of returning a wrong pool."""
        for expert_id in self.expert_records:
            origin = self.origin_key_id(expert_id)
            if origin not in self.key_records:
                raise MultiKeyPoolError(f"expert {expert_id} has no origin key")
            record = self.key_records[origin]
            if record["key_type"] != KEY_TYPE_ORIGIN:
                raise MultiKeyPoolError(f"{origin} is not marked as an origin key")
            if self.expert_records[expert_id]["origin_task"] != record["task_id"]:
                raise MultiKeyPoolError(
                    f"expert {expert_id} origin key is on task {record['task_id']}, "
                    f"expected {self.expert_records[expert_id]['origin_task']}"
                )
        for key_id, record in self.key_records.items():
            if self.keys[key_id].numel() != self.query_dim:
                raise MultiKeyPoolError(f"{key_id} has the wrong dimension")
            if record["key_type"] == KEY_TYPE_TASK_ALIAS:
                expert = self.expert_records[record["expert_id"]]
                if record["task_id"] == expert["origin_task"]:
                    raise MultiKeyPoolError(
                        f"alias key {key_id} sits on the origin task; use the origin key"
                    )
            elif record["key_type"] == KEY_TYPE_ORIGIN:
                # An origin key says "this expert was created on task t", so a
                # second origin key on any other task is not a second historical
                # key -- it is a mislabelled alias that would route as if the
                # expert had been formed on that task.
                expert = self.expert_records[record["expert_id"]]
                if record["task_id"] != expert["origin_task"]:
                    raise MultiKeyPoolError(
                        f"origin key {key_id} is on task {record['task_id']}, but "
                        f"expert {record['expert_id']} originates on task "
                        f"{expert['origin_task']}"
                    )
        return self.audit()

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def export_state(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "pool_kind": "v8_multi_key",
            "query_dim": self.query_dim,
            "pool_version": len(self.key_records),
            "keys": {key_id: self.keys[key_id].detach().to("cpu").clone()
                     for key_id in self.key_ids()},
            "metadata": {
                "experts": {str(expert_id): dict(record)
                            for expert_id, record in self.expert_records.items()},
                "key_records": {key_id: dict(record)
                                for key_id, record in self.key_records.items()},
            },
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        current_task: Optional[int] = None,
    ) -> "MultiKeyExpertPool":
        if state.get("pool_kind") != "v8_multi_key":
            raise MultiKeyPoolError(
                "not a V8 multi-key state; migrate V7 pools with load_v7_pool"
            )
        pool = cls(query_dim=int(state.get("query_dim", 1536)))
        metadata = state.get("metadata", {})
        experts = metadata.get("experts", {})
        records = metadata.get("key_records", {})
        keys: Mapping[str, torch.Tensor] = state["keys"]
        for expert_id, record in experts.items():
            if int(expert_id) in pool.expert_records:
                continue
            fields = {key: value for key, value in record.items()
                      if key not in ("expert_id", "trainable")}
            pool.add_expert(expert_id=int(expert_id), **fields)
        for key_id, tensor in keys.items():
            record = dict(records[key_id])
            trainable = bool(record.pop("trainable", False))
            if current_task is not None:
                trainable = trainable and record["task_id"] == int(current_task)
            pool.add_key(
                expert_id=record.pop("expert_id"),
                task_id=record.pop("task_id"),
                key_type=record.pop("key_type"),
                value=tensor,
                lifecycle=record.pop("lifecycle", LIFECYCLE_CANDIDATE),
                trainable=trainable,
                support_count=record.pop("support_count", 0),
                teacher_gain=record.pop("teacher_gain", 0.0),
                validation_gain=record.pop("validation_gain", 0.0),
                extra=record.pop("extra", {}),
            )
        pool.validate()
        return pool

    # -- V7 migration ---------------------------------------------------
    @classmethod
    def load_v7_pool(
        cls,
        v7_state: Mapping[str, Any],
        expert_manifest: Optional[Mapping[str, Any]] = None,
        frozen: bool = True,
    ) -> "MultiKeyExpertPool":
        """Migrate a V7 ``V7ExpertKeyPool`` state to the multi-key structure.

        ``E_i -> K_i`` becomes ``E_i -> K(i, origin_task)`` with the **identical
        tensor bytes** and the **identical** ``rms_state``.  No re-normalization,
        no re-training, no re-derivation of the key.
        """
        keys: Mapping[str, torch.Tensor] = v7_state.get("keys", {})
        metadata: Mapping[str, Any] = v7_state.get("metadata", {})
        pool = cls(query_dim=int(v7_state.get("query_dim", 1536)))
        manifest_experts: Dict[int, Dict[str, Any]] = {}
        if expert_manifest:
            for entry in expert_manifest.get("experts", []) or []:
                manifest_experts[int(entry["expert_id"])] = entry
        for raw_id, tensor in sorted(keys.items(), key=lambda item: int(item[0])):
            expert_id = int(raw_id)
            record = dict(metadata.get(expert_id) or metadata.get(str(expert_id)) or {})
            origin_task = int(record.get("origin_task", expert_id // 4))
            lifecycle_v7 = str(record.get("lifecycle", "historical"))
            lifecycle = (LIFECYCLE_PRUNED if lifecycle_v7 == LIFECYCLE_PRUNED
                         else (LIFECYCLE_HISTORICAL if lifecycle_v7 == "historical"
                               else LIFECYCLE_CANDIDATE))
            entry = manifest_experts.get(expert_id, {})
            pool.add_expert(
                expert_id=expert_id,
                origin_task=origin_task,
                creation_task=origin_task,
                lora_path=entry.get("lora_path"),
                rms_state=record.get("rms_state", entry.get("rms_state")),
                lifecycle=lifecycle,
                extra={"v7_lifecycle": lifecycle_v7,
                       "migrated_from": "compose.v7.pool.V7ExpertKeyPool"},
            )
            pool.add_key(
                expert_id=expert_id,
                task_id=origin_task,
                key_type=KEY_TYPE_ORIGIN,
                value=tensor,
                lifecycle=lifecycle if lifecycle != LIFECYCLE_CANDIDATE else LIFECYCLE_HISTORICAL,
                trainable=not frozen,
                support_count=int(entry.get("support_count", 0) or 0),
                extra={"v7_lifecycle": lifecycle_v7},
            )
        pool.validate()
        return pool

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (f"experts={len(self.expert_records)}, keys={len(self.key_records)}, "
                f"query_dim={self.query_dim}")


def alias_key_init(positive_queries: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """``K_init(k, t) = Normalize(mean(q_i for i in teacher positives))``.

    PART 11: the alias key starts at the centroid of the current-task queries
    that this historical expert actually solves, then is refined by the key loss.
    """
    if positive_queries.numel() == 0:
        raise MultiKeyPoolError("alias key initialisation requires at least one positive")
    centroid = positive_queries.detach().float().mean(dim=0)
    return F.normalize(centroid.reshape(-1), dim=-1, eps=eps)


__all__ = [
    "LIFECYCLE_CANDIDATE",
    "LIFECYCLE_HISTORICAL",
    "LIFECYCLE_PRUNED",
    "MultiKeyExpertPool",
    "MultiKeyPoolError",
    "alias_key_init",
    "tensor_checksum",
]
