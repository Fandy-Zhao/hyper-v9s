"""V9-S key structure: ``1 Expert : {base identity} + {task-scoped keys}``.

A *Multi-Key Functional Expert* is an expert plus a memory of routing keys.  The
**origin** key is the expert's long-term functional identity: it is frozen the
moment the expert exists and it never moves again.  A **task_alias** key is an
independent, absolute key scoped to one later task: when a historical expert is
recalled into task ``t``, the key it routes by during ``t`` is its alias for
``t``, and that key is the only part of the expert that trains.

This is V8's multi-key semantics, deliberately unchanged.  Two things V9 v1 did
are absent here and must stay absent on the V9-S main path:

* there is **no residual decomposition** -- no ``normalize(base + gamma*delta)``,
  no ``gamma``, no stored delta.  A task key is an absolute vector, so it can
  move anywhere the answer gradient's responsibility points, instead of being
  confined to a cone around a base key that may be the very thing that failed to
  recall it;
* there is **no in-place rewrite of a committed key**.  The base key is the
  freeze-audit's fixed point (see :meth:`historical_key_ids`); learning happens
  only in keys that do not exist yet at task start.

A newly created task key starts *at* the expert's own base key.  That preserves
the property the residual was introduced for -- a historical expert enters a new
task routing as exactly itself, never re-anchored onto the current task's query
mean -- without the decomposition, because the initialisation is a starting
point for an independent parameter rather than a constraint on it.

Nothing here re-derives the query, re-normalises a stored key in place, or
trains an old expert.  This class extends ``MultiKeyExpertPool`` rather than
duplicating it: the key store, the lifecycle machinery, the freeze switches and
the integrity checksums are the V8 ones.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from compose.v8.config import KEY_TYPE_ORIGIN, KEY_TYPE_TASK_ALIAS
from compose.v8.pool import (
    LIFECYCLE_CANDIDATE,
    LIFECYCLE_HISTORICAL,
    LIFECYCLE_PRUNED,
    MultiKeyExpertPool,
    MultiKeyPoolError,
)

#: ``legacy_v9_only``.
#:
#: The key role V9 v1 used (``normalize(base + gamma * delta)``).  V9-S never
#: creates a key of this type; the constant survives for exactly one purpose --
#: reading a checkpoint written by V9 v1 in :meth:`V9KeyPool.from_state`, which
#: converts each residual to the absolute key it resolved to.  Nothing else in
#: this package may reference it, and no V9-S run can produce one.
LEGACY_KEY_TYPE_TASK_RESIDUAL = "task_residual"


class V9KeyPoolError(MultiKeyPoolError):
    """Raised when a V9-S pool invariant would be violated."""


#: Key roles that may serve as an expert's routing identity for a task.
_ROUTING_KEY_TYPES = (KEY_TYPE_ORIGIN, KEY_TYPE_TASK_ALIAS)


class V9KeyPool(MultiKeyExpertPool):
    """Multi-key pool with V8's absolute task-key semantics.

    Extra structure over the V8 pool is small on purpose:

    * every expert owns exactly one **origin** key -- its base functional
      identity, frozen as soon as the expert is committed (candidates keep
      theirs trainable while they are being formed);
    * a historical expert may own at most one **task_alias** key per later
      task, and it is the only key of that expert that trains;
    * :meth:`effective_key_matrix` resolves any key id to the unit vector that
      actually routes, keeping the graph attached when asked.
    """

    def __init__(self, query_dim: int = 1536) -> None:
        super().__init__(query_dim=query_dim)

    # ------------------------------------------------------------------
    # ids
    # ------------------------------------------------------------------
    @property
    def historical_ids(self) -> List[int]:
        """Committed experts -- capability carriers that never train again."""
        return self.expert_ids(lifecycle=LIFECYCLE_HISTORICAL)

    @property
    def current_ids(self) -> List[int]:
        """This task's candidates, still being formed."""
        return self.expert_ids(lifecycle=LIFECYCLE_CANDIDATE)

    def task_key_id(self, expert_id: int, task_id: int) -> str:
        return self.alias_key_id(int(expert_id), int(task_id))

    def has_task_key(self, expert_id: int, task_id: int) -> bool:
        key_id = self.task_key_id(expert_id, task_id)
        return key_id in self.key_records and (
            self.key_records[key_id]["lifecycle"] != LIFECYCLE_PRUNED
        )

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def add_expert_with_base(
        self,
        expert_id: int,
        base_key: torch.Tensor,
        origin_task: int,
        lifecycle: str = LIFECYCLE_HISTORICAL,
        rms_state: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        trainable: Optional[bool] = None,
    ) -> int:
        """Register an expert together with its base functional key.

        The base key is frozen by default -- that is what makes it an expert's
        stable long-term identity.  A **candidate** is the exception: while it is
        being formed its key is the thing the answer-side responsibility trains,
        so ``trainable`` defaults to ``lifecycle == candidate``.  Committing a
        candidate freezes it in place (see :mod:`compose.v9.audit`), at which
        point it is a base key like any other.
        """
        expert_id = int(expert_id)
        if trainable is None:
            trainable = lifecycle == LIFECYCLE_CANDIDATE
        self.add_expert(
            expert_id=expert_id,
            origin_task=int(origin_task),
            creation_task=int(origin_task),
            rms_state=rms_state,
            lifecycle=lifecycle,
            extra=extra,
        )
        self.add_key(
            expert_id=expert_id,
            task_id=int(origin_task),
            key_type=KEY_TYPE_ORIGIN,
            value=base_key,
            lifecycle=lifecycle,
            trainable=bool(trainable),
        )
        return expert_id

    def add_task_key(
        self,
        expert_id: int,
        task_id: int,
        value: Optional[torch.Tensor] = None,
        trainable: bool = True,
    ) -> str:
        """Give a historical expert an independent key for one later task.

        Defaults to the expert's own base key, so at the start of the task the
        expert routes exactly as it did before.  The key is absolute: it is not
        a delta on the base, and after this call the base plays no further part
        in how the expert is recalled on this task.
        """
        expert_id = int(expert_id)
        task_id = int(task_id)
        record = self.expert_record(expert_id)
        if record["lifecycle"] != LIFECYCLE_HISTORICAL:
            raise V9KeyPoolError(
                f"expert {expert_id} is {record['lifecycle']!r}; only a "
                "historical expert takes a current-task key"
            )
        if task_id <= int(record["origin_task"]):
            raise V9KeyPoolError(
                f"expert {expert_id} originates on task {record['origin_task']}; "
                f"a task key on task {task_id} would shadow its base identity"
            )
        key_id = self.task_key_id(expert_id, task_id)
        if key_id in self.key_records:
            raise V9KeyPoolError(f"duplicate task key {key_id}")
        if value is None:
            value = self.keys[self.base_key_id(expert_id)].detach().clone()
        return self.add_key(
            expert_id=expert_id,
            task_id=task_id,
            key_type=KEY_TYPE_TASK_ALIAS,
            value=value,
            lifecycle=LIFECYCLE_CANDIDATE,
            trainable=bool(trainable),
        )

    # ------------------------------------------------------------------
    # routing identity
    # ------------------------------------------------------------------
    def routing_key_id(self, expert_id: int, task_id: int) -> str:
        """The single key that routes this expert during ``task_id``.

        A historical expert routes through its current-task key when one
        exists, otherwise through its frozen base key.  A candidate routes
        through its own (trainable) origin key.
        """
        expert_id = int(expert_id)
        if self.has_task_key(expert_id, task_id):
            return self.task_key_id(expert_id, task_id)
        return self.origin_key_id(expert_id)

    def routing_key_ids(self, expert_ids: Iterable[int], task_id: int) -> List[str]:
        return [self.routing_key_id(expert_id, task_id) for expert_id in expert_ids]

    def memory_key_ids(self, expert_id: int) -> List[str]:
        """Every key in an expert's retained memory: base plus later task keys.

        This is the inference-side view: the expert is recalled when *any* of
        its keys matches, so a capability learned on task 1 stays reachable on
        task 5 even after the task keys of the intervening tasks were pruned.
        """
        expert_id = int(expert_id)
        return [
            key_id
            for key_id in self.key_ids(expert_id=expert_id)
            if self.key_records[key_id]["key_type"] in _ROUTING_KEY_TYPES
            and self.key_records[key_id]["lifecycle"] != LIFECYCLE_PRUNED
        ]

    def base_key_id(self, expert_id: int) -> str:
        return self.origin_key_id(int(expert_id))

    def base_key_matrix(self, expert_ids: Sequence[int], detach: bool = True) -> torch.Tensor:
        """Stacked unit base keys, the cheap historical recall geometry."""
        return self.effective_key_matrix(
            [self.base_key_id(expert_id) for expert_id in expert_ids], detach=detach
        )

    def effective_key_matrix(
        self, key_ids: Sequence[str], detach: bool = False
    ) -> torch.Tensor:
        """Resolve key ids to the unit vectors that actually route.

        Every key resolves to its own normalised stored value -- there is no
        composition step, because every key is already absolute.  With
        ``detach=False`` the returned matrix stays connected to the trainable
        key parameters, which is what the routing and the contribution
        measurement need.
        """
        if not key_ids:
            return torch.zeros(0, self.query_dim, dtype=torch.float32)
        raw = self.raw_matrix(list(key_ids)).float()
        if detach:
            raw = raw.detach()
        # Normalising row-wise is the whole operation; keeping it in one call
        # means the fp32 upcast happens once for the whole block.
        return F.normalize(raw, dim=-1)

    # ------------------------------------------------------------------
    # trainability
    # ------------------------------------------------------------------
    def trainable_key_ids(self) -> List[str]:
        return sorted(
            key_id
            for key_id, record in self.key_records.items()
            if record["trainable"] and self.keys[key_id].requires_grad
        )

    def task_key_ids(self, task_id: int) -> List[str]:
        """Absolute task keys belonging to one task, live ones only."""
        return [
            key_id
            for key_id in self.key_ids(task_id=int(task_id), key_type=KEY_TYPE_TASK_ALIAS)
            if self.key_records[key_id]["lifecycle"] != LIFECYCLE_PRUNED
        ]

    def freeze_historical(self, current_task: Optional[int] = None) -> List[str]:
        """Freeze every key that does not belong to ``current_task``.

        The V8 rule already does the right thing for V9-S -- a task key lives on
        the current task id, so it survives; every base key and every earlier
        task key is frozen.
        """
        return super().freeze_historical(current_task=current_task)

    def reset_task_key(self, expert_id: int, task_id: int) -> None:
        """Return a historical expert to its base identity for this task.

        Used when a task-end audit rejects the expert's current-task key: the
        key is restored to the base key it started from, so the expert is
        recalled on later tasks exactly as it was before this task ran.  The
        base identity was never touched, so nothing has to be undone.
        """
        expert_id = int(expert_id)
        key_id = self.task_key_id(expert_id, task_id)
        if key_id not in self.key_records:
            raise V9KeyPoolError(f"no task key {key_id}")
        with torch.no_grad():
            self.keys[key_id].copy_(
                self.keys[self.base_key_id(expert_id)].detach().reshape(-1)
            )

    # ------------------------------------------------------------------
    # integrity
    # ------------------------------------------------------------------
    def validate(self) -> Dict[str, Any]:
        audit = super().validate()
        for key_id, record in self.key_records.items():
            if record["key_type"] != KEY_TYPE_TASK_ALIAS:
                continue
            expert = self.expert_records[record["expert_id"]]
            if record["task_id"] <= int(expert["origin_task"]):
                raise V9KeyPoolError(
                    f"task key {key_id} sits on or before its expert's "
                    f"origin task {expert['origin_task']}"
                )
        for expert_id in self.expert_records:
            aliases = [
                key_id for key_id in self.key_ids(expert_id=int(expert_id))
                if self.key_records[key_id]["key_type"] == KEY_TYPE_TASK_ALIAS
            ]
            tasks = [self.key_records[key_id]["task_id"] for key_id in aliases]
            if len(tasks) != len(set(tasks)):
                raise V9KeyPoolError(
                    f"expert {expert_id} has two task keys on one task"
                )
        return audit

    def audit(self) -> Dict[str, Any]:
        payload = super().audit()
        aliases = self.key_ids(key_type=KEY_TYPE_TASK_ALIAS)
        payload.update(
            {
                "pool_kind": "v9s_multi_key",
                "num_task_keys": len(aliases),
                "task_key_task_ids": sorted(
                    {int(self.key_records[key_id]["task_id"]) for key_id in aliases}
                ),
                #: ``legacy_v9_only``: must be 0 for any pool a V9-S run built.
                "num_legacy_residual_keys": 0,
            }
        )
        return payload

    def historical_checksums(self) -> Dict[str, str]:
        from compose.v8.pool import tensor_checksum

        return {
            key_id: tensor_checksum(self.keys[key_id])
            for key_id in self.historical_key_ids()
        }

    def historical_key_ids(self) -> List[str]:
        """Keys that must never change: committed base keys and earlier keys.

        A candidate's own origin key is *not* in this set while the candidate is
        being formed (its lifecycle is ``candidate``), and neither is a
        historical expert's current-task key (same reason): both are trainable
        by design, and both are keys the responsibility supervision shapes.
        Including them would make the end-of-task freeze audit fail against the
        training it just ran.
        """
        return super().historical_key_ids()

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def export_state(self) -> Dict[str, Any]:
        state = super().export_state()
        state["pool_kind"] = "v9s_multi_key"
        return state

    @classmethod
    def from_state(
        cls, state: Mapping[str, Any], current_task: Optional[int] = None
    ) -> "V9KeyPool":
        pool_kind = state.get("pool_kind")
        if pool_kind == LEGACY_KEY_TYPE_TASK_RESIDUAL or pool_kind == "v9_task_residual":
            return cls._from_legacy_v9_state(state, current_task=current_task)
        if pool_kind != "v8_multi_key" and pool_kind != "v9s_multi_key":
            raise V9KeyPoolError(
                "not a V9-S/V8 key state; migrate V7 pools with load_v7_pool"
            )
        base = MultiKeyExpertPool.from_state(state, current_task=current_task)
        pool = cls(query_dim=base.query_dim)
        # Re-home the already-validated parameters rather than re-creating them,
        # so a round-trip preserves every byte and every `requires_grad` flag.
        pool.keys = base.keys
        pool.expert_records = base.expert_records
        pool.key_records = base.key_records
        pool.validate()
        return pool

    @classmethod
    def _from_legacy_v9_state(
        cls, state: Mapping[str, Any], current_task: Optional[int] = None
    ) -> "V9KeyPool":
        """``legacy_v9_only``: read a V9 v1 state written before V9-S.

        V9 v1 stored a historical expert's current-task routing key as a delta
        against its base key, resolved at routing time by
        ``normalize(base + gamma * delta)``.  V9-S stores absolute keys, so each
        legacy residual is converted to **the absolute key it resolved to** --
        which is the key that was actually routing, so the converted pool routes
        exactly as the checkpoint did.  The conversion is lossy in one direction
        only: afterwards the key can move freely instead of being tied to the
        base.

        This exists so an interrupted V9 v1 run can be resumed rather than
        restarted.  It is not reachable from any V9-S code path: nothing in this
        package writes ``pool_kind="v9_task_residual"``.
        """
        legacy = dict(state)
        gamma = float(legacy.get("gamma", 1.0))
        key_records = legacy.get("key_records") or {}
        keys = legacy.get("keys") or {}
        # Resolve every residual to its effective absolute key before the V8
        # loader sees the state, so the V8 key store is never asked to hold a
        # key role it does not know.
        for key_id, record in list(key_records.items()):
            if record.get("key_type") != LEGACY_KEY_TYPE_TASK_RESIDUAL:
                continue
            if key_id not in keys:
                raise V9KeyPoolError(
                    f"legacy residual {key_id} has no stored tensor"
                )
            expert_id = int(record["expert_id"])
            base_id = "expert{}:task{}:{}".format(
                expert_id, int(record["task_id"]), KEY_TYPE_ORIGIN
            )
            origin_id = None
            for candidate_id, candidate in key_records.items():
                if (
                    int(candidate["expert_id"]) == expert_id
                    and candidate.get("key_type") == KEY_TYPE_ORIGIN
                ):
                    origin_id = candidate_id
                    break
            if origin_id is None:
                raise V9KeyPoolError(
                    f"legacy residual {key_id} has no origin key to resolve against "
                    f"(expected {base_id!r})"
                )
            base = keys[origin_id].detach().float().reshape(-1)
            delta = keys[key_id].detach().float().reshape(-1)
            resolved = F.normalize(base + gamma * delta, dim=-1)
            keys[key_id] = resolved
            record["key_type"] = KEY_TYPE_TASK_ALIAS
            record["extra"] = dict(record.get("extra") or {})
            record["extra"]["legacy_v9_only"] = {
                "converted_from": LEGACY_KEY_TYPE_TASK_RESIDUAL,
                "gamma": gamma,
                "note": "absolute key = normalize(base + gamma*delta) at migration",
            }
        legacy.pop("gamma", None)
        legacy["pool_kind"] = "v8_multi_key"
        base = MultiKeyExpertPool.from_state(legacy, current_task=current_task)
        pool = cls(query_dim=base.query_dim)
        pool.keys = base.keys
        pool.expert_records = base.expert_records
        pool.key_records = base.key_records
        pool.validate()
        return pool


def _tangent_noise(centres: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Project each ``noise`` row onto the tangent plane of its centre.

    ``centres`` is ``[D]`` (broadcast over every noise row) or ``[N, D]``.  The
    returned rows are unit vectors: a perturbation that is not tangent to the
    sphere would be removed by the re-normalisation anyway, so it is projected
    explicitly rather than silently discarded.
    """
    if centres.ndim == 1:
        centres = centres.unsqueeze(0).expand(noise.shape[0], -1)
    projection = (noise * centres).sum(dim=-1, keepdim=True)
    return F.normalize(noise - projection * centres, dim=-1)


def spherical_kmeans_keys(
    queries: torch.Tensor,
    count: int,
    max_samples: int = 4096,
    seed: int = 42,
    iterations: int = 25,
    perturbation: float = 0.0,
) -> torch.Tensor:
    """Deterministic spherical k-means centroids used to *seed* candidate keys.

    The returned centres are initialisation only.  They do not define which
    samples a candidate trains on: every candidate competes for every sample,
    and the final boundary is whatever the answer's responsibility produces.  A
    cheap seeded k-means on a fixed subsample is enough to break the symmetry
    between candidates that a shared task mean would not.
    """
    if count < 1:
        raise ValueError("count must be positive")
    matrix = queries.detach().float().reshape(-1, queries.shape[-1])
    if matrix.shape[0] == 0:
        raise ValueError("candidate key initialisation requires at least one query")
    matrix = F.normalize(matrix, dim=-1)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if matrix.shape[0] > max_samples:
        index = torch.randperm(matrix.shape[0], generator=generator)[:max_samples]
        sample = matrix.index_select(0, index.to(matrix.device))
    else:
        sample = matrix
    # k-means++ seeding keeps the initial centres far apart, which is the whole
    # point of using clustering here instead of a shared mean.
    centres = [sample[int(torch.randint(sample.shape[0], (1,), generator=generator).item())]]
    for _ in range(count - 1):
        similarity = torch.stack([sample @ centre for centre in centres], dim=1).max(dim=1).values
        weight = (1.0 - similarity).clamp_min(0.0) ** 2
        total = float(weight.sum())
        if total <= 0:
            pick = int(torch.randint(sample.shape[0], (1,), generator=generator).item())
        else:
            pick = int(torch.multinomial(weight / total, 1, generator=generator).item())
        centres.append(sample[pick])
    centres_tensor = torch.stack(centres, dim=0)
    for _ in range(int(iterations)):
        assignment = (sample @ centres_tensor.T).argmax(dim=1)
        updated = centres_tensor.clone()
        for index in range(count):
            members = sample[assignment == index]
            if members.shape[0] == 0:
                continue
            updated[index] = F.normalize(members.mean(dim=0), dim=-1)
        if torch.allclose(updated, centres_tensor, atol=1e-6):
            centres_tensor = updated
            break
        centres_tensor = updated
    if perturbation > 0:
        noise = torch.randn(centres_tensor.shape, generator=generator)
        centres_tensor = F.normalize(
            centres_tensor + float(perturbation) * _tangent_noise(centres_tensor, noise),
            dim=-1,
        )
    return centres_tensor.contiguous()


def initialize_candidate_keys(
    queries: torch.Tensor,
    count: int,
    strategy: str = "spherical_kmeans",
    perturbation: float = 0.01,
    max_samples: int = 4096,
    seed: int = 42,
) -> torch.Tensor:
    """Seed ``count`` candidate keys.

    ``spherical_kmeans`` is preferred: the task mean plus small perturbations
    leaves the candidates nearly collinear, and a tiny initial advantage then
    compounds into a single winner.  Well-separated centres keep every
    candidate competitive long enough for the answer signal to decide.
    """
    if strategy == "spherical_kmeans":
        return spherical_kmeans_keys(
            queries,
            count,
            max_samples=max_samples,
            seed=seed,
            perturbation=perturbation,
        )
    if strategy == "task_mean_perturbed":
        mean = F.normalize(queries.detach().float().mean(dim=0).reshape(-1), dim=-1)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(count, mean.numel(), generator=generator)
        noise = _tangent_noise(mean, noise)
        return F.normalize(
            mean.unsqueeze(0) + float(perturbation) * noise, dim=-1
        ).contiguous()
    raise ValueError(f"unknown candidate init strategy {strategy!r}")


__all__ = [
    "LEGACY_KEY_TYPE_TASK_RESIDUAL",
    "V9KeyPool",
    "V9KeyPoolError",
    "initialize_candidate_keys",
    "spherical_kmeans_keys",
]
