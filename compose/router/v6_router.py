"""V6 dual-mode Query-Key Router (Stage E3).

Query (task-book formula):

    z_v = frozen_visual_feature(image)
    z_s = frozen_instruction_feature(instruction)
    q   = normalize(W_q(concat(layer_norm(z_v), layer_norm(z_s))))

Expert keys:  e_k = normalize(raw_key_k)

Matching:     score_ik = cosine(q_i, e_k)
              prob_ik  = sigmoid((score_ik - bias_k) / temperature)

Two modes share the same encoder/keys and differ only in how scores are
consumed:

- ``training_retrieval``: returns the historical-expert Top-M (the whole
  pool when K <= M); it never decides residuals directly.
- ``inference_selection``: empty when all probabilities are below
  ``tau_none``; single when only the first passes; pair when the top two
  pass; at most ``max_active_experts`` experts.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .expert_keys import ExpertKeyMetadata, ExpertKeyStore

V6_ROUTER_CHECKPOINT_VERSION = 1
ROUTER_MODES = ("training_retrieval", "inference_selection")


@dataclass(frozen=True)
class V6RetrievalResult:
    """Top-M retrieval result (training_retrieval mode)."""

    expert_ids: Tuple[int, ...]  # ranked candidate expert ids
    similarities: Tensor  # [batch, M] cosine similarities
    probabilities: Tensor  # [batch, M] sigmoid probabilities
    all_visible_ids: Tuple[int, ...]


@dataclass(frozen=True)
class V6SelectionResult:
    """empty/single/pair decision per sample (inference_selection mode)."""

    sets: Tuple[Tuple[int, ...], ...]  # one set per sample, at most 2 experts
    probabilities: Tensor  # [batch, K] sigmoid probabilities over visible keys
    expert_ids: Tuple[int, ...]  # visible expert ids aligned to probabilities


class V6QueryEncoder(nn.Module):
    """Task-book dual-modal query encoder.

    ``z_v`` / ``z_s`` are frozen features (typically CLIP image/text
    embeddings); each is LayerNorm'ed, concatenated, projected and L2
    normalized. No answer or task-id inputs exist by design.
    """

    def __init__(self, visual_dim: int = 768, text_dim: int = 768, query_dim: int = 128) -> None:
        super().__init__()
        if visual_dim <= 0 or text_dim <= 0 or query_dim <= 0:
            raise ValueError("query encoder dimensions must be positive")
        self.visual_dim = int(visual_dim)
        self.text_dim = int(text_dim)
        self.query_dim = int(query_dim)
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.query_projection = nn.Linear(visual_dim + text_dim, query_dim)

    def forward(self, z_v: Tensor, z_s: Tensor) -> Tensor:
        if z_v.ndim != 2 or z_s.ndim != 2:
            raise ValueError("frozen features must be rank-2 [batch, dim]")
        if z_v.shape[0] != z_s.shape[0]:
            raise ValueError("visual and text features must share the batch size")
        if z_v.shape[1] != self.visual_dim or z_s.shape[1] != self.text_dim:
            raise ValueError(
                "feature dims mismatch: expected ({}, {}) got ({}, {})".format(
                    self.visual_dim, self.text_dim, z_v.shape[1], z_s.shape[1]
                )
            )
        fused = torch.cat([self.visual_norm(z_v), self.text_norm(z_s)], dim=-1)
        return F.normalize(self.query_projection(fused), dim=-1)


class V6Router(nn.Module):
    """Dual-mode router over an ``ExpertKeyStore`` with per-expert bias and
    temperature-scaled sigmoid matching."""

    def __init__(
        self,
        query_encoder: V6QueryEncoder,
        top_m: int = 8,
        max_active_experts: int = 2,
        temperature: float = 1.0,
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
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.query_encoder = query_encoder
        self.query_dim = query_encoder.query_dim
        self.top_m = int(top_m)
        self.max_active_experts = int(max_active_experts)
        self.tau_none = float(tau_none)
        self.tau_second = float(tau_second)
        self.feature_extractor_version = str(feature_extractor_version)
        self.key_store = ExpertKeyStore(metadata=[], query_dim=self.query_dim, seed=seed)
        self.expert_bias = nn.ParameterDict()
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())
        self.pool_version = None  # type: Optional[int]
        self.config_hash = None  # type: Optional[str]

    # ------------------------------------------------------------------
    # Key and bias management
    # ------------------------------------------------------------------

    @property
    def expert_ids(self) -> Tuple[int, ...]:
        return self.key_store.expert_ids

    @property
    def temperature(self) -> float:
        return float(self.log_temperature.exp().clamp(0.05, 20.0))

    def add_expert(
        self,
        expert_id: int,
        creation_task: int,
        checkpoint_sha256: str,
        key: Optional[Tensor] = None,
        key_initialization: str = "seeded_random",
        bias: float = 0.0,
    ) -> None:
        """Register a new expert key (and its bias). A provided ``key`` is
        used verbatim (normalized); otherwise the store seeds randomly."""
        expert_id = int(expert_id)
        if str(expert_id) in self.expert_bias:
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
        self.expert_bias[str(expert_id)] = nn.Parameter(torch.tensor(float(bias)))

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _scores(self, query: Tensor, visible_ids: Sequence[int]) -> Tuple[Tensor, Tuple[int, ...]]:
        ids = tuple(int(value) for value in visible_ids)
        if not ids:
            batch = query.shape[0]
            device = query.device
            empty = torch.empty(batch, 0, device=device)
            return empty, ()
        keys = self.key_store.normalized(ids)
        similarities = query @ keys.T  # [batch, K]
        temperature = self.temperature
        bias = torch.stack([self.expert_bias[str(value)] for value in ids])
        probabilities = torch.sigmoid((similarities - bias) / temperature)
        return probabilities, ids

    def retrieve(self, query: Tensor, visible_ids: Sequence[int]) -> V6RetrievalResult:
        """training_retrieval: historical Top-M (whole pool when K <= M)."""
        probabilities, ids = self._scores(query, visible_ids)
        count = len(ids)
        if count == 0:
            batch = query.shape[0]
            device = query.device
            return V6RetrievalResult(
                expert_ids=(),
                similarities=torch.empty(batch, 0, device=device),
                probabilities=torch.empty(batch, 0, device=device),
                all_visible_ids=(),
            )
        top_k = min(self.top_m, count)
        top_probabilities, top_indices = torch.topk(
            probabilities, k=top_k, dim=1, sorted=True
        )
        top_ids = tuple(ids[int(index)] for index in top_indices[0])
        return V6RetrievalResult(
            expert_ids=top_ids,
            similarities=top_probabilities,  # monotonic transform of cosine
            probabilities=top_probabilities,
            all_visible_ids=ids,
        )

    def select(self, query: Tensor, visible_ids: Sequence[int]) -> V6SelectionResult:
        """inference_selection: empty/single/pair with tau thresholds."""
        probabilities, ids = self._scores(query, visible_ids)
        count = len(ids)
        sets = []
        if count == 0:
            for _ in range(query.shape[0]):
                sets.append(())
            return V6SelectionResult(
                sets=tuple(sets), probabilities=probabilities, expert_ids=()
            )
        for row in probabilities:
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
        return V6SelectionResult(
            sets=tuple(sets), probabilities=probabilities, expert_ids=ids
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
            "checkpoint_version": V6_ROUTER_CHECKPOINT_VERSION,
            "kind": "v6_router",
            "query_encoder": self.query_encoder.state_dict(),
            "expert_keys": self.key_store.state_dict(),
            "key_metadata": self.key_store.metadata_state(),
            "expert_bias": {key: value.detach().cpu().item() for key, value in self.expert_bias.items()},
            "log_temperature": float(self.log_temperature.detach().cpu()),
            "temperature": self.temperature,
            "tau_none": self.tau_none,
            "tau_second": self.tau_second,
            "top_m": self.top_m,
            "max_active_experts": self.max_active_experts,
            "query_dim": self.query_dim,
            "feature_extractor_version": self.feature_extractor_version,
            "pool_version": int(pool_version),
            "config_hash": str(config_hash),
        }

    def load_state_dict_extra(self, state: Dict[str, Any]) -> None:
        if int(state.get("checkpoint_version", -1)) != V6_ROUTER_CHECKPOINT_VERSION:
            raise ValueError(
                "unsupported v6 router checkpoint_version: {}".format(
                    state.get("checkpoint_version")
                )
            )
        if state.get("kind") != "v6_router":
            raise ValueError("checkpoint is not a v6_router: {!r}".format(state.get("kind")))
        self.query_encoder.load_state_dict(state["query_encoder"])
        # Rebuild the key ParameterDict (empty store cannot load_state_dict).
        self.key_store.keys = nn.ParameterDict(
            {
                key[len("keys."):]: nn.Parameter(value.detach().clone())
                for key, value in state["expert_keys"].items()
            }
        )
        # Restore key metadata (not part of the ParameterDict state dict).
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
        self.log_temperature.data.fill_(float(state["log_temperature"]))
        for key, bias in sorted(state["expert_bias"].items()):
            self.expert_bias[key] = nn.Parameter(torch.tensor(float(bias)))
        self.pool_version = int(state["pool_version"])
        self.config_hash = str(state["config_hash"])

    def validate_pool_version(self, pool_version: int) -> None:
        if self.pool_version is None:
            raise ValueError("router has no pool_version; load a checkpoint first")
        if self.pool_version != int(pool_version):
            raise ValueError(
                "router pool_version {} does not match registry pool_version {}".format(
                    self.pool_version, pool_version
                )
            )


def save_v6_router_checkpoint(path: str, router: V6Router,
                              pool_version: int, config_hash: str,
                              extra: Optional[Dict[str, Any]] = None) -> None:
    """Atomically persist the router checkpoint (mkstemp + fsync + replace).

    The checkpoint binds: query encoder, expert keys + metadata, per-expert
    bias, temperature, tau thresholds, pool_version, feature extractor
    version and the config hash. Stored as a torch binary for exact tensor
    state; metadata stays JSON-safe inside the payload.
    """
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


def load_v6_router_checkpoint(path: str, router: V6Router) -> Dict[str, Any]:
    """Load a router checkpoint into ``router``; returns the ``extra`` block."""
    from pathlib import Path

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError("router checkpoint does not exist: {}".format(target))
    state = torch.load(target, map_location="cpu", weights_only=False)
    router.load_state_dict_extra(state)
    return dict(state.get("extra") or {})
