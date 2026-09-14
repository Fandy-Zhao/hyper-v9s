"""V7 key lifecycle, the route-key registry and reproducible key initialization.

V7 originally made ``expert id == key id``: one 1536-D routing key per expert,
stored as ``self.keys[str(expert_id)]``.  That structure cannot express the
reuse hypothesis -- *a reusable historical expert recalled by the current task*
-- because the two identities are the same object.

The pool now keeps a **route-key registry**:

    Route key  ->  Expert id  ->  LoRA

* every expert keeps exactly one ``canonical`` key whose ``key_id`` is still
  ``str(expert_id)``, so every committed artifact, checkpoint and caller that
  indexed ``keys[str(expert_id)]`` keeps working byte for byte;
* a reusable historical expert gains one additional ``reuse`` key per later
  task, created by the *same* initializer as the current candidates;
* the current candidates' keys are ``candidate`` keys.

Expert lifecycle and key lifecycle are separate:
``metadata[expert_id]["lifecycle"]`` describes the *expert*, while
``route_keys[key_id].lifecycle`` / ``.trainable`` describe the *key*.  Key
trainability is decided by the explicit ``trainable_key_ids`` set, never by
``expert.lifecycle == "current"`` -- that is what allows a historical expert to
own a learnable current-task key without dragging its LoRA into the optimizer.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .query import full_train_task_center

KEY_TYPE_CANONICAL = "canonical"
KEY_TYPE_REUSE = "reuse"
KEY_TYPE_CANDIDATE = "candidate"
KEY_TYPES = (KEY_TYPE_CANONICAL, KEY_TYPE_REUSE, KEY_TYPE_CANDIDATE)

KEY_LIFECYCLE_CURRENT = "current"
KEY_LIFECYCLE_HISTORICAL = "historical"
KEY_LIFECYCLE_PRUNED = "pruned"
KEY_LIFECYCLES = (KEY_LIFECYCLE_CURRENT, KEY_LIFECYCLE_HISTORICAL, KEY_LIFECYCLE_PRUNED)

POOL_SCHEMA_VERSION = 2
_REUSE_KEY_ID = re.compile(r"^e(\d+)_t(\d+)_reuse$")


def tensor_checksum(value: Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def pairwise_key_cosine(keys: Tensor) -> Tensor:
    normalized = F.normalize(keys.detach().float(), dim=-1)
    return normalized @ normalized.T


@dataclass(frozen=True)
class RouteKeyEntry:
    """Why one 1536-D routing tensor exists in the pool.

    ``key_id`` is the address inside ``keys``; ``expert_id`` is the LoRA the key
    points at.  Several keys may share one ``expert_id``.
    """

    key_id: str
    expert_id: int
    key_type: str
    origin_task: int
    lifecycle: str
    trainable: bool

    def __post_init__(self) -> None:
        if self.key_type not in KEY_TYPES:
            raise ValueError("unknown route key type {!r}".format(self.key_type))
        if self.lifecycle not in KEY_LIFECYCLES:
            raise ValueError("unknown route key lifecycle {!r}".format(self.lifecycle))

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RouteKeyEntry":
        return cls(
            key_id=str(value["key_id"]),
            expert_id=int(value["expert_id"]),
            key_type=str(value["key_type"]),
            origin_task=int(value["origin_task"]),
            lifecycle=str(value["lifecycle"]),
            trainable=bool(value["trainable"]),
        )

    def replaced(self, **changes) -> "RouteKeyEntry":
        fields = self.to_dict()
        fields.update(changes)
        return RouteKeyEntry.from_dict(fields)


# ----------------------------------------------------------------------
# current-task key initialization (one definition, used by both callers)
# ----------------------------------------------------------------------
def tangent_perturbed_keys(center: Tensor, noise: Tensor, perturbation: float) -> Tensor:
    """The single key initializer kernel, shared by candidates and reuse keys.

    ``noise`` is ``[N, D]`` standard normal; the result is ``[N, D]``: project
    the noise onto the tangent plane of the task center, L2-normalize it, scale
    it by ``perturbation`` and L2-normalize the perturbation of the center.
    ``keys = Normalize(center + perturbation * Normalize(Tangent(noise)))``.
    """
    if perturbation <= 0:
        raise ValueError("perturbation must be positive")
    if noise.ndim != 2 or noise.shape[1] != center.numel():
        raise ValueError("noise must have shape [N, query_dim]")
    noise = noise.to(center.device)
    noise = noise - (noise @ center).unsqueeze(1) * center.unsqueeze(0)
    noise = F.normalize(noise, dim=-1)
    return F.normalize(center.unsqueeze(0) + float(perturbation) * noise, dim=-1)


def initialize_current_task_key(
    center: Tensor,
    perturbation: float = 0.01,
    seed: int = 42,
    query_dim: int = 1536,
) -> Tensor:
    """One current-task key from an already-computed task center.

    Same kernel, same normalization and same perturbation scale as
    :func:`initialize_candidate_keys`; only the generator stream differs, which
    is what breaks the symmetry between the keys of one task.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(1, int(query_dim), generator=generator)
    return tangent_perturbed_keys(center, noise, perturbation)[0]


def reuse_key_seed(seed: int, task_index: int, expert_id: int) -> int:
    """Deterministic, per-(task, expert) stream for reuse-key initialization.

    It is a different stream from the candidate stream but drawn from the same
    distribution and pushed through the same kernel -- no key is copied, no
    second initialization algorithm is invented.
    """
    return int(seed) * 100003 + int(task_index) * 1009 + int(expert_id)


def initialize_candidate_keys(
    queries: Tensor,
    num_train_samples: int,
    count: int = 4,
    perturbation: float = 0.01,
    seed: int = 42,
) -> Tuple[Tensor, Tensor, Dict[str, object]]:
    if count != 4:
        raise ValueError("V7 initializes exactly four candidates")
    if perturbation <= 0:
        raise ValueError("perturbation must be positive")
    center, coverage = full_train_task_center(queries, num_train_samples)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(count, center.numel(), generator=generator).to(center.device)
    # Distinct tangential directions keep every key equally close to the same
    # center without fabricating clusters.
    keys = tangent_perturbed_keys(center, noise, perturbation)
    cosine = pairwise_key_cosine(keys)
    off_diagonal = cosine[~torch.eye(count, dtype=torch.bool, device=cosine.device)]
    if torch.any(off_diagonal >= 1.0 - 1.0e-8):
        raise AssertionError("candidate keys must not be identical")
    return keys, center, {
        **coverage,
        "seed": int(seed),
        "perturbation": float(perturbation),
        "pairwise_cosine": cosine.cpu().tolist(),
    }


class V7ExpertKeyPool(nn.Module):
    """Expert catalogue plus the per-expert route-key registry.

    ``self.keys`` is an ``nn.ParameterDict`` mapping ``key_id -> [1536]``;
    ``self.route_keys`` says which expert each key points at and whether the key
    is a canonical, reuse or candidate key.
    """

    def __init__(self, query_dim: int = 1536, pool_version: int = 0) -> None:
        super().__init__()
        if query_dim != 1536:
            raise ValueError("V7 expert keys must be 1536-D")
        self.query_dim = int(query_dim)
        self.pool_version = int(pool_version)
        self.keys = nn.ParameterDict()
        self.metadata: Dict[int, Dict[str, object]] = {}
        self.route_keys: Dict[str, RouteKeyEntry] = {}
        self.trainable_key_ids: Set[str] = set()

    # ------------------------------------------------------------------
    # ids
    # ------------------------------------------------------------------
    @staticmethod
    def canonical_key_id(expert_id: int) -> str:
        return str(int(expert_id))

    @staticmethod
    def reuse_key_id(expert_id: int, task_index: int) -> str:
        key_id = "e{}_t{}_reuse".format(int(expert_id), int(task_index))
        if "." in key_id:
            raise ValueError("route key ids may not contain '.'")
        return key_id

    @staticmethod
    def parse_reuse_key_id(key_id: str) -> Optional[Tuple[int, int]]:
        match = _REUSE_KEY_ID.match(str(key_id))
        return (int(match.group(1)), int(match.group(2))) if match else None

    # ------------------------------------------------------------------
    # experts
    # ------------------------------------------------------------------
    def add(
        self,
        expert_id: int,
        key: Tensor,
        origin_task: int,
        lifecycle: str,
        trainable: bool,
        rms_state: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Register ``expert_id`` with its one canonical routing key."""
        expert_id = int(expert_id)
        key_id = self.canonical_key_id(expert_id)
        if key_id in self.route_keys or str(expert_id) in self.keys:
            raise ValueError("duplicate expert key {}".format(expert_id))
        if key.shape != (self.query_dim,):
            raise ValueError("expert key must have shape [1536]")
        normalized = F.normalize(key.detach().float(), dim=0)
        self.keys[key_id] = nn.Parameter(normalized, requires_grad=bool(trainable))
        self.metadata[expert_id] = {
            "expert_id": expert_id,
            "origin_task": int(origin_task),
            "lifecycle": str(lifecycle),
            "rms_state": dict(rms_state or {}),
        }
        self.route_keys[key_id] = RouteKeyEntry(
            key_id=key_id,
            expert_id=expert_id,
            key_type=(
                KEY_TYPE_CANDIDATE
                if str(lifecycle) == KEY_LIFECYCLE_CURRENT
                else KEY_TYPE_CANONICAL
            ),
            origin_task=int(origin_task),
            lifecycle=str(lifecycle),
            trainable=bool(trainable),
        )
        self._apply_trainable(key_id, bool(trainable))

    @property
    def expert_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self.metadata))

    @property
    def current_ids(self) -> Tuple[int, ...]:
        """Experts created by the *current* task (the candidate experts).

        This stays an expert-level question: a reusable historical expert that
        owns a learnable current-task reuse key is still a historical expert.
        """
        return tuple(
            expert_id
            for expert_id in self.expert_ids
            if self.metadata[expert_id]["lifecycle"] == "current"
        )

    @property
    def historical_ids(self) -> Tuple[int, ...]:
        return tuple(
            expert_id
            for expert_id in self.expert_ids
            if self.metadata[expert_id]["lifecycle"] == "historical"
        )

    # ------------------------------------------------------------------
    # keys
    # ------------------------------------------------------------------
    def add_reuse_key(
        self,
        expert_id: int,
        task_index: int,
        key: Tensor,
        trainable: bool = True,
        lifecycle: str = KEY_LIFECYCLE_CURRENT,
    ) -> str:
        """Attach an additional routing key to an existing historical expert."""
        expert_id = int(expert_id)
        task_index = int(task_index)
        if expert_id not in self.metadata:
            raise ValueError("reuse key for unknown expert {}".format(expert_id))
        if self.metadata[expert_id]["lifecycle"] != "historical":
            raise ValueError(
                "reuse keys belong to frozen historical experts; expert {} is {}".format(
                    expert_id, self.metadata[expert_id]["lifecycle"]
                )
            )
        if task_index <= int(self.metadata[expert_id]["origin_task"]):
            raise ValueError(
                "reuse key task {} must be later than expert {} origin task {}".format(
                    task_index, expert_id, self.metadata[expert_id]["origin_task"]
                )
            )
        key_id = self.reuse_key_id(expert_id, task_index)
        if key_id in self.route_keys:
            raise ValueError("duplicate reuse key {}".format(key_id))
        value = key.detach().float().reshape(-1)
        if value.shape != (self.query_dim,):
            raise ValueError("reuse key must have shape [1536]")
        value = F.normalize(value, dim=0)
        self.keys[key_id] = nn.Parameter(value, requires_grad=bool(trainable))
        self.route_keys[key_id] = RouteKeyEntry(
            key_id=key_id,
            expert_id=expert_id,
            key_type=KEY_TYPE_REUSE,
            origin_task=task_index,
            lifecycle=str(lifecycle),
            trainable=bool(trainable),
        )
        self._apply_trainable(key_id, bool(trainable))
        return key_id

    def route_key(self, key_id: str) -> RouteKeyEntry:
        try:
            return self.route_keys[str(key_id)]
        except KeyError as exc:
            raise KeyError("unknown route key {}".format(key_id)) from exc

    def is_trainable_key(self, key_id: str) -> bool:
        return str(key_id) in self.trainable_key_ids

    @property
    def key_ids(self) -> Tuple[str, ...]:
        return tuple(
            sorted(
                self.route_keys,
                key=lambda key_id: (
                    self.route_keys[key_id].expert_id,
                    self.route_keys[key_id].origin_task,
                    self.route_keys[key_id].key_id,
                ),
            )
        )

    def keys_of_expert(
        self,
        expert_id: int,
        key_type: Optional[str] = None,
        lifecycle: Optional[str] = None,
    ) -> Tuple[str, ...]:
        expert_id = int(expert_id)
        return tuple(
            key_id
            for key_id in self.key_ids
            if self.route_keys[key_id].expert_id == expert_id
            and (key_type is None or self.route_keys[key_id].key_type == key_type)
            and (lifecycle is None or self.route_keys[key_id].lifecycle == lifecycle)
        )

    def trainable_keys_of(self, expert_id: int) -> Tuple[str, ...]:
        """Trainable route keys of one expert, in canonical registry order.

        This is the *only* definition of "the keys the key loss may update": a
        candidate expert's candidate key, or a reusable historical expert's
        current-task reuse key.  Never the canonical historical key.
        """
        return tuple(
            key_id
            for key_id in self.keys_of_expert(int(expert_id))
            if key_id in self.trainable_key_ids
        )

    def active_key_ids(self, excluded: Iterable[int] = ()) -> Tuple[str, ...]:
        """Keys eligible for routing: live key and live, non-excluded expert."""
        excluded_set = {int(value) for value in excluded}
        active = []
        for key_id in self.key_ids:
            entry = self.route_keys[key_id]
            if entry.lifecycle == KEY_LIFECYCLE_PRUNED:
                continue
            if entry.expert_id in excluded_set:
                continue
            if self.metadata[entry.expert_id]["lifecycle"] == KEY_LIFECYCLE_PRUNED:
                continue
            active.append(key_id)
        return tuple(active)

    def has_reuse_key(self, expert_id: int, task_index: int) -> bool:
        return self.reuse_key_id(expert_id, task_index) in self.route_keys

    def normalized_keys(self, key_ids: Sequence[str]) -> Tensor:
        if not key_ids:
            return torch.empty(0, self.query_dim)
        return torch.stack(
            [F.normalize(self.keys[str(key_id)], dim=0) for key_id in key_ids]
        )

    def normalized(self, expert_ids: Optional[Sequence[int]] = None) -> Tensor:
        """One normalized row per expert (its canonical key)."""
        values = self.expert_ids if expert_ids is None else tuple(int(v) for v in expert_ids)
        if not values:
            return torch.empty(0, self.query_dim)
        return self.normalized_keys([self.canonical_key_id(v) for v in values])

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def _apply_trainable(self, key_id: str, trainable: bool) -> None:
        key_id = str(key_id)
        parameter = self.keys[key_id]
        parameter.requires_grad_(bool(trainable))
        if not trainable:
            parameter.grad = None
            self.trainable_key_ids.discard(key_id)
        else:
            self.trainable_key_ids.add(key_id)
        entry = self.route_keys[key_id]
        if entry.trainable != bool(trainable):
            self.route_keys[key_id] = entry.replaced(trainable=bool(trainable))

    def _set_key_lifecycle(self, key_id: str, lifecycle: str) -> None:
        key_id = str(key_id)
        if lifecycle not in KEY_LIFECYCLES:
            raise ValueError("unknown route key lifecycle {!r}".format(lifecycle))
        entry = self.route_keys[key_id]
        if entry.lifecycle != lifecycle:
            self.route_keys[key_id] = entry.replaced(lifecycle=lifecycle)

    def freeze_historical(self) -> None:
        """Freeze the *canonical* keys of historical experts.

        Reuse keys are untouched: their trainability is recorded explicitly in
        ``trainable_key_ids`` and is a decision of the current task, not of the
        expert's lifecycle.
        """
        for expert_id in self.historical_ids:
            key_id = self.canonical_key_id(expert_id)
            if key_id in self.trainable_key_ids:
                continue
            self._apply_trainable(key_id, False)

    def freeze_all(self) -> None:
        for key_id in self.key_ids:
            self._apply_trainable(key_id, False)

    def historical_checksums(self) -> Dict[str, str]:
        """Checksums of every frozen key of the pool.

        Covers the canonical keys of historical experts and every reuse key
        already committed to the pool by an earlier task.
        """
        return {
            key_id: tensor_checksum(self.keys[key_id])
            for key_id in self.key_ids
            if self.route_keys[key_id].lifecycle == KEY_LIFECYCLE_HISTORICAL
        }

    # ------------------------------------------------------------------
    # task end
    # ------------------------------------------------------------------
    def drop_key(self, key_id: str) -> None:
        """Remove a route key that must not pollute the committed pool."""
        key_id = str(key_id)
        self._apply_trainable(key_id, False)
        del self.keys[key_id]
        self.route_keys.pop(key_id, None)
        self.trainable_key_ids.discard(key_id)

    def commit(
        self,
        retained_ids: Iterable[int],
        metrics: Mapping[int, Mapping[str, object]],
        reuse_key_retention: Optional[Mapping[int, bool]] = None,
    ) -> None:
        """Canonicalize the retained candidates and settle this task's reuse keys.

        ``reuse_key_retention`` maps a *reusable historical* expert id to
        ``True`` (freeze its current-task reuse key and keep it as an additional
        historical routing key) or ``False`` (discard it).  An expert that is
        absent from the mapping keeps whatever reuse key it already had.
        """
        retained = {int(value) for value in retained_ids}
        for expert_id in list(self.current_ids):
            key_id = self.canonical_key_id(expert_id)
            if expert_id in retained:
                self.metadata[expert_id]["lifecycle"] = "historical"
                self.metadata[expert_id]["validation"] = dict(metrics.get(expert_id, {}))
                self.keys[key_id].data.copy_(F.normalize(self.keys[key_id].detach(), dim=0))
                self._apply_trainable(key_id, False)
                self._set_key_lifecycle(key_id, KEY_LIFECYCLE_HISTORICAL)
                self.route_keys[key_id] = self.route_keys[key_id].replaced(
                    key_type=KEY_TYPE_CANONICAL
                )
            else:
                self.metadata[expert_id]["lifecycle"] = "pruned"
                self._apply_trainable(key_id, False)
                self._set_key_lifecycle(key_id, KEY_LIFECYCLE_PRUNED)
        self.pool_version += len(retained)
        for raw_expert_id, keep in dict(reuse_key_retention or {}).items():
            expert_id = int(raw_expert_id)
            if expert_id not in self.metadata:
                raise ValueError("reuse-key commit for unknown expert {}".format(expert_id))
            if self.metadata[expert_id]["lifecycle"] != "historical":
                raise ValueError(
                    "reuse-key commit expects a historical expert, {} is {}".format(
                        expert_id, self.metadata[expert_id]["lifecycle"]
                    )
                )
            for key_id in self.keys_of_expert(
                expert_id, key_type=KEY_TYPE_REUSE, lifecycle=KEY_LIFECYCLE_CURRENT
            ):
                if bool(keep):
                    self.keys[key_id].data.copy_(
                        F.normalize(self.keys[key_id].detach(), dim=0)
                    )
                    self._apply_trainable(key_id, False)
                    self._set_key_lifecycle(key_id, KEY_LIFECYCLE_HISTORICAL)
                    committed = dict(self.metadata[expert_id].get("reuse_keys") or {})
                    committed[str(self.route_keys[key_id].origin_task)] = {
                        "key_id": key_id,
                        "origin_task": int(self.route_keys[key_id].origin_task),
                    }
                    self.metadata[expert_id]["reuse_keys"] = committed
                else:
                    self.drop_key(key_id)

    def selectable_ids(self, excluded: Iterable[int] = ()) -> Tuple[int, ...]:
        excluded_set = {int(value) for value in excluded}
        return tuple(
            expert_id
            for expert_id in self.expert_ids
            if expert_id not in excluded_set
            and self.metadata[expert_id]["lifecycle"] != "pruned"
        )

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def export_state(self) -> Dict[str, object]:
        return {
            "schema_version": POOL_SCHEMA_VERSION,
            "query_dim": self.query_dim,
            "pool_version": self.pool_version,
            "keys": {key_id: value.detach().cpu() for key_id, value in self.keys.items()},
            "metadata": {str(key): dict(value) for key, value in self.metadata.items()},
            "route_keys": {key_id: entry.to_dict() for key_id, entry in self.route_keys.items()},
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "V7ExpertKeyPool":
        key_dim = int(state["query_dim"])
        pool = cls(key_dim, int(state.get("pool_version", 0)))
        metadata = state["metadata"]
        keys = state["keys"]
        route_keys = state.get("route_keys")
        if route_keys is None:
            # Schema 1: one key per expert, ``key_id == str(expert_id)``.
            if int(state.get("schema_version", 1)) >= POOL_SCHEMA_VERSION:
                raise ValueError("V7 key state declares the route-key schema but omits it")
            for raw_id in sorted(metadata, key=int):
                entry = dict(metadata[raw_id])
                lifecycle = str(entry["lifecycle"])
                pool.add(
                    int(raw_id), keys[raw_id], int(entry["origin_task"]),
                    lifecycle, lifecycle == "current", entry.get("rms_state", {}),
                )
                pool.metadata[int(raw_id)].update(entry)
        else:
            for raw_id in sorted(metadata, key=int):
                entry = dict(metadata[raw_id])
                lifecycle = str(entry["lifecycle"])
                canonical = cls.canonical_key_id(int(raw_id))
                if canonical not in keys:
                    raise ValueError(
                        "expert {} has no canonical key in the exported state".format(raw_id)
                    )
                pool.add(
                    int(raw_id), keys[canonical], int(entry["origin_task"]),
                    lifecycle, False, entry.get("rms_state", {}),
                )
                pool.metadata[int(raw_id)].update(entry)
            for key_id, raw_entry in sorted(route_keys.items()):
                entry = RouteKeyEntry.from_dict(raw_entry)
                if entry.key_id != str(key_id):
                    raise ValueError("route-key registry key/id mismatch for {}".format(key_id))
                if entry.key_type == KEY_TYPE_REUSE:
                    canonical = cls.reuse_key_id(entry.expert_id, entry.origin_task)
                    if canonical != str(key_id):
                        raise ValueError("malformed reuse key id {}".format(key_id))
                    pool.add_reuse_key(
                        entry.expert_id, entry.origin_task, keys[str(key_id)],
                        trainable=entry.trainable, lifecycle=entry.lifecycle,
                    )
                else:
                    if str(key_id) != cls.canonical_key_id(entry.expert_id):
                        raise ValueError("malformed canonical key id {}".format(key_id))
                    pool.route_keys[str(key_id)] = entry
                    pool._apply_trainable(str(key_id), entry.trainable)
        pool.freeze_historical()
        return pool
