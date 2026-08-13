"""Compose router: frozen query encoder + historical expert keys + cosine
matching.

The Compose router is purely functional at inference: it owns the frozen
query encoder, the historical (frozen) expert keys and the learned new
keys, and matches queries with plain cosine similarity. There is no
bias/temperature head, no global calibration and no training-time BCE
distillation.

Two consumption modes:

- ``training_retrieval``: per-sample Top-M retrieval over historical
  experts. Every sample in the batch receives its OWN Top-M expert id row
  (``[batch, M]``), never a broadcast of batch row 0.
- ``inference_selection``: per-sample empty/single/pair decision using the
  fixed ``tau_none`` / ``tau_second`` thresholds from the configuration.

New expert keys are added with ``add_expert``; keys from later tasks are
learned by ``compose.router.key_learning`` and remain frozen afterwards.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .expert_keys import ExpertKeyMetadata, ExpertKeyStore
from .inference import predict_sets
from .set_router import ExpertSetRouter

COMPOSE_ROUTER_CHECKPOINT_VERSION = 1
ROUTER_MODES = ("training_retrieval", "inference_selection")


@dataclass(frozen=True)
class ComposeRetrievalResult:
    """Per-sample Top-M retrieval (training_retrieval mode).

    ``expert_ids`` is a ``[batch, M]`` LongTensor: row ``i`` holds the
    Top-M expert ids for sample ``i`` (padded with ``PAD_EXPERT_ID`` when
    the visible pool is smaller than ``top_m``).
    """

    expert_ids: Tensor  # [batch, M] LongTensor, per-sample rows
    similarities: Tensor  # [batch, M] cosine similarities
    visible_ids: Tuple[int, ...]  # the expert ids the router could see


@dataclass(frozen=True)
class ComposeRouterSelection:
    """Per-sample empty/single/pair decision (inference_selection mode)."""

    sets: Tuple[Tuple[int, ...], ...]  # one set per sample, at most 2 experts
    probabilities: Tensor  # [batch, K] cosine similarities over visible keys
    expert_ids: Tuple[int, ...]  # visible expert ids aligned to probabilities
    answer_features_used: bool = False
    oracle_used: bool = False
    task_id_lookup_used: bool = False
    clustering_used_at_test: bool = False


PAD_EXPERT_ID = -1


class ComposeRouter(nn.Module):
    """Cosine-matching router over an ``ExpertKeyStore``."""

    def __init__(
        self,
        query_encoder: Optional[nn.Module] = None,
        top_m: int = 8,
        max_active_experts: int = 2,
        tau_none: float = 0.5,
        tau_second: float = 0.5,
        feature_extractor_version: str = "frozen_clip_l14_336_v1",
        seed: int = 42,
    ) -> None:
        super().__init__()
        if top_m <= 0 or max_active_experts <= 0:
            raise ValueError("top_m and max_active_experts must be positive")
        if not 0.0 <= tau_none <= 1.0 or not 0.0 <= tau_second <= 1.0:
            raise ValueError("router thresholds must lie in [0, 1]")
        if query_encoder is None:
            # Checkpoint-load callers (eval_task, snapshot.load_router)
            # construct an empty router and restore the frozen encoder from
            # the checkpoint via load_state_dict; give them a deterministically
            # seeded placeholder of the same architecture.
            from compose.router.functional_query import ComposeQueryEncoder

            query_encoder = ComposeQueryEncoder(seed=seed)
        self.query_encoder = query_encoder
        self.query_dim = int(query_encoder.query_dim)
        self.top_m = int(top_m)
        self.max_active_experts = int(max_active_experts)
        self.tau_none = float(tau_none)
        self.tau_second = float(tau_second)
        self.feature_extractor_version = str(feature_extractor_version)
        self.key_store = ExpertKeyStore(metadata=[], query_dim=self.query_dim, seed=seed)
        self.pool_version = None  # type: Optional[int]
        self.config_hash = None  # type: Optional[str]
        self.set_router = ExpertSetRouter(self.query_dim, self.top_m)
        self.set_router_enabled = False
        self.pair_threshold = 0.65
        self.route_anchors = []  # train-only compressed query/decision anchors

    # ------------------------------------------------------------------
    # Key management
    # ------------------------------------------------------------------

    @property
    def expert_ids(self) -> Tuple[int, ...]:
        return self.key_store.expert_ids

    def add_expert(
        self,
        expert_id: int,
        creation_task: int,
        checkpoint_sha256: str,
        key: Optional[Tensor] = None,
        key_initialization: str = "seeded_random",
    ) -> None:
        """Register an expert key. A provided ``key`` is used verbatim
        (normalized); otherwise the store seeds deterministically."""
        expert_id = int(expert_id)
        if str(expert_id) in self.key_store.keys:
            raise ValueError("expert {} already present in router".format(expert_id))
        if key is not None:
            if key.ndim != 1 or key.shape[0] != self.query_dim:
                raise ValueError("key must have shape [{}]".format(self.query_dim))
            with torch.no_grad():
                self.key_store.keys[str(expert_id)] = nn.Parameter(
                    F.normalize(key.float().detach(), dim=0)
                )
        else:
            self.key_store.keys[str(expert_id)] = nn.Parameter(
                F.normalize(torch.randn(self.query_dim), dim=0)
            )
        self.key_store.metadata[expert_id] = ExpertKeyMetadata(
            expert_id=expert_id,
            creation_task=int(creation_task),
            checkpoint_sha256=str(checkpoint_sha256),
            initialization=str(key_initialization),
        )

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _similarities(
        self, query: Tensor, visible_ids: Sequence[int]
    ) -> Tuple[Tensor, Tuple[int, ...]]:
        ids = tuple(int(value) for value in visible_ids)
        if not ids:
            batch = query.shape[0]
            device = query.device
            return torch.empty(batch, 0, device=device), ()
        keys = self.key_store.normalized(ids)
        similarities = query @ keys.T  # [batch, K]
        return similarities, ids

    def retrieve(
        self, query: Tensor, visible_ids: Sequence[int]
    ) -> ComposeRetrievalResult:
        """Per-sample Top-M over the visible historical experts.

        Every batch sample gets its own Top-M row (no broadcast of row 0).
        Rows are padded with ``PAD_EXPERT_ID`` when the visible pool is
        smaller than ``top_m``.
        """
        if query.ndim != 2:
            raise ValueError("query must be rank-2 [batch, query_dim]")
        similarities, ids = self._similarities(query, visible_ids)
        count = len(ids)
        batch = query.shape[0]
        if count == 0:
            empty_ids = torch.full(
                (batch, self.top_m),
                PAD_EXPERT_ID,
                dtype=torch.long,
                device=query.device,
            )
            return ComposeRetrievalResult(
                expert_ids=empty_ids,
                similarities=torch.empty(batch, 0, device=query.device),
                visible_ids=(),
            )
        top_k = min(self.top_m, count)
        top_similarities, top_indices = torch.topk(
            similarities, k=top_k, dim=1, sorted=True
        )
        # Per-sample ids: top_indices[i] maps to ids[i] for EACH sample i.
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=query.device)
        top_ids = ids_tensor[top_indices]  # [batch, top_k]
        if top_k < self.top_m:
            padding = torch.full(
                (batch, self.top_m - top_k),
                PAD_EXPERT_ID,
                dtype=torch.long,
                device=query.device,
            )
            top_ids = torch.cat([top_ids, padding], dim=1)
            padded_similarities = torch.cat(
                [
                    top_similarities,
                    torch.zeros(batch, self.top_m - top_k, device=query.device),
                ],
                dim=1,
            )
        else:
            padded_similarities = top_similarities
        return ComposeRetrievalResult(
            expert_ids=top_ids,
            similarities=padded_similarities,
            visible_ids=ids,
        )

    def select(
        self, query: Tensor, visible_ids: Sequence[int]
    ) -> ComposeRouterSelection:
        """Inference selection: empty/single/pair with fixed thresholds."""
        if query.ndim != 2:
            raise ValueError("query must be rank-2 [batch, query_dim]")
        similarities, ids = self._similarities(query, visible_ids)
        count = len(ids)
        sets = []
        if count == 0:
            for _ in range(query.shape[0]):
                sets.append(())
            return ComposeRouterSelection(
                sets=tuple(sets), probabilities=similarities, expert_ids=()
            )
        if self.set_router_enabled:
            keys = self.key_store.normalized(ids)
            visible_mask = torch.ones(
                query.shape[0], len(ids), dtype=torch.bool, device=query.device
            )
            output = self.set_router(query, keys, ids, visible_mask)
            learned_sets = tuple(predict_sets(output, self.pair_threshold))
            return ComposeRouterSelection(
                sets=learned_sets, probabilities=similarities, expert_ids=ids
            )
        for row in similarities:
            values, indices = torch.topk(
                row, k=min(self.max_active_experts, count), sorted=True
            )
            if float(values[0]) < self.tau_none:
                sets.append(())
            elif len(values) == 1 or float(values[1]) < self.tau_second:
                sets.append((int(ids[int(indices[0])]),))
            else:
                sets.append(
                    (int(ids[int(indices[0])]), int(ids[int(indices[1])]))
                )
        return ComposeRouterSelection(
            sets=tuple(sets), probabilities=similarities, expert_ids=ids
        )

    def run(self, query: Tensor, mode: str, visible_ids: Sequence[int]):
        if mode == "training_retrieval":
            return self.retrieve(query, visible_ids)
        if mode == "inference_selection":
            return self.select(query, visible_ids)
        raise ValueError(
            "mode must be one of {}; got {!r}".format(ROUTER_MODES, mode)
        )

    def set_thresholds(self, tau_none: float, tau_second: float) -> None:
        if not 0.0 <= tau_none <= 1.0 or not 0.0 <= tau_second <= 1.0:
            raise ValueError("router thresholds must lie in [0, 1]")
        self.tau_none = float(tau_none)
        self.tau_second = float(tau_second)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def state_dict_extra(self, pool_version: int, config_hash: str) -> Dict[str, Any]:
        return {
            "checkpoint_version": COMPOSE_ROUTER_CHECKPOINT_VERSION,
            "kind": "compose_router",
            "query_encoder": self.query_encoder.state_dict(),
            "query_encoder_provenance": self.query_encoder.provenance().to_dict(),
            "expert_keys": self.key_store.state_dict(),
            "key_metadata": self.key_store.metadata_state(),
            "tau_none": self.tau_none,
            "tau_second": self.tau_second,
            "top_m": self.top_m,
            "max_active_experts": self.max_active_experts,
            "query_dim": self.query_dim,
            "feature_extractor_version": self.feature_extractor_version,
            "pool_version": int(pool_version),
            "config_hash": str(config_hash),
            "set_router_enabled": bool(self.set_router_enabled),
            "set_router": self.set_router.state_dict(),
            "pair_threshold": float(self.pair_threshold),
            "route_anchors": list(self.route_anchors),
        }

    def load_state_dict_extra(self, state: Dict[str, Any]) -> None:
        if int(state.get("checkpoint_version", -1)) != COMPOSE_ROUTER_CHECKPOINT_VERSION:
            raise ValueError(
                "unsupported compose router checkpoint_version: {}".format(
                    state.get("checkpoint_version")
                )
            )
        if state.get("kind") != "compose_router":
            raise ValueError(
                "checkpoint is not a compose_router: {!r}".format(state.get("kind"))
            )
        self.query_encoder.load_state_dict(state["query_encoder"])
        # Rebuild the key ParameterDict (empty store cannot load_state_dict).
        self.key_store.keys = nn.ParameterDict(
            {
                key[len("keys."):]: nn.Parameter(value.detach().clone())
                for key, value in state["expert_keys"].items()
            }
        )
        for entry in state["key_metadata"].get("experts", []):
            metadata = ExpertKeyMetadata(
                expert_id=int(entry["expert_id"]),
                creation_task=int(entry["creation_task"]),
                checkpoint_sha256=str(entry["checkpoint_sha256"]),
                key_version=int(entry.get("key_version", 1)),
                archived=bool(entry.get("archived", False)),
                initialization=str(entry.get("initialization", "seeded_random")),
            )
            self.key_store.metadata[metadata.expert_id] = metadata
        self.tau_none = float(state["tau_none"])
        self.tau_second = float(state["tau_second"])
        self.top_m = int(state["top_m"])
        self.max_active_experts = int(state["max_active_experts"])
        self.feature_extractor_version = str(state["feature_extractor_version"])
        self.pool_version = int(state["pool_version"])
        self.config_hash = str(state["config_hash"])
        if "set_router" in state:
            self.set_router = ExpertSetRouter(self.query_dim, self.top_m)
            self.set_router.load_state_dict(state["set_router"])
            self.set_router_enabled = bool(state.get("set_router_enabled", False))
            self.pair_threshold = float(state.get("pair_threshold", 0.65))
            self.route_anchors = [dict(value) for value in state.get("route_anchors", [])]

    def validate_pool_version(self, pool_version: int) -> None:
        if self.pool_version is None:
            raise ValueError("router has no pool_version; load a checkpoint first")
        if self.pool_version != int(pool_version):
            raise ValueError(
                "router pool_version {} does not match registry pool_version {}".format(
                    self.pool_version, pool_version
                )
            )


def save_compose_router_checkpoint(
    path: str,
    router: ComposeRouter,
    pool_version: int,
    config_hash: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomically persist the router checkpoint (mkstemp + fsync + replace)."""
    import os
    import tempfile
    from pathlib import Path

    payload = router.state_dict_extra(pool_version, config_hash)
    payload["extra"] = extra or {}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_compose_router_checkpoint(
    path: str, router: ComposeRouter
) -> Dict[str, Any]:
    """Load a router checkpoint into ``router``; returns the ``extra`` block."""
    from pathlib import Path

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError("router checkpoint does not exist: {}".format(target))
    state = torch.load(target, map_location="cpu", weights_only=False)
    router.load_state_dict_extra(state)
    return dict(state.get("extra") or {})
