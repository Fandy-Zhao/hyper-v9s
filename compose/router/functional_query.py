"""Stable functional query encoder (Compose UCIT pipeline).

The query encoder maps frozen multimodal features to a 128-D L2-normalized
query shared by every downstream stage (historical key retrieval, residual
clustering, cluster key learning, inference matching):

    z_v = frozen visual feature (CLIP image embedding)
    z_s = frozen instruction feature (CLIP text embedding)
    q   = L2Normalize(MLP([LayerNorm(z_v); LayerNorm(z_s)]))

Stability contract:

- the encoder is initialized once with a fixed seed (deterministic
  parameters) and frozen for the whole continual run;
- cluster formation never updates the encoder;
- later tasks load the existing encoder checkpoint instead of creating a
  new random one;
- the initialization provenance (seed, dimensions, module hash) is
  recorded and persisted in every snapshot.

``compose/router/query_encoder.py`` is a separate subsystem (the legacy
Set-Router MultimodalQueryEncoder); do not merge the two.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

QUERY_ENCODER_VERSION = 1


def _seeded_module_state(module: nn.Module, seed: int) -> None:
    """Deterministic parameter initialization from a fixed seed.

    Uses a local torch.Generator so the result is independent of global
    RNG state at creation time.
    """
    generator = torch.Generator().manual_seed(int(seed))
    for parameter in module.parameters():
        nn.init.normal_(parameter, std=0.02, generator=generator)
    for name, parameter in module.named_parameters():
        if "weight" in name and parameter.ndim >= 2:
            nn.init.xavier_uniform_(parameter, generator=generator)
    for name, parameter in module.named_parameters():
        if "bias" in name:
            # zeros_ does not accept a generator (and needs none: the zero
            # value is deterministic by definition).
            nn.init.zeros_(parameter)


@dataclass(frozen=True)
class QueryEncoderProvenance:
    """Deterministic initialization record for the functional query encoder."""

    seed: int
    visual_dim: int
    text_dim: int
    query_dim: int
    module_hash: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "visual_dim": self.visual_dim,
            "text_dim": self.text_dim,
            "query_dim": self.query_dim,
            "module_hash": self.module_hash,
        }


def _stable_hash(value) -> str:
    import hashlib
    import json

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ComposeQueryEncoder(nn.Module):
    """Task-book functional query encoder (frozen after initialization)."""

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 768,
        query_dim: int = 128,
        seed: int = 42,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        if visual_dim <= 0 or text_dim <= 0 or query_dim <= 0:
            raise ValueError("query encoder dimensions must be positive")
        self.visual_dim = int(visual_dim)
        self.text_dim = int(text_dim)
        self.query_dim = int(query_dim)
        self.init_seed = int(seed)
        self.visual_norm = nn.LayerNorm(self.visual_dim)
        self.text_norm = nn.LayerNorm(self.text_dim)
        self.query_projection = nn.Sequential(
            nn.Linear(self.visual_dim + self.text_dim, self.query_dim),
            nn.GELU(),
            nn.Linear(self.query_dim, self.query_dim),
        )
        if initialize:
            _seeded_module_state(self, seed)

    # ------------------------------------------------------------------
    # Provenance / stability
    # ------------------------------------------------------------------

    def provenance(self) -> QueryEncoderProvenance:
        state = self.state_dict()
        signature = {
            name: value.detach().cpu().tolist()
            for name, value in sorted(state.items())
        }
        signature["dimensions"] = {
            "visual_dim": self.visual_dim,
            "text_dim": self.text_dim,
            "query_dim": self.query_dim,
        }
        signature["seed"] = self.init_seed
        return QueryEncoderProvenance(
            seed=self.init_seed,
            visual_dim=self.visual_dim,
            text_dim=self.text_dim,
            query_dim=self.query_dim,
            module_hash=_stable_hash(signature),
        )

    def freeze(self) -> None:
        """Freeze every parameter of the encoder (the default for the run)."""
        for parameter in self.parameters():
            parameter.requires_grad = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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


def query_encoder_state_dict(encoder: ComposeQueryEncoder) -> Dict[str, object]:
    return {
        "checkpoint_version": QUERY_ENCODER_VERSION,
        "kind": "compose_query_encoder",
        "visual_dim": encoder.visual_dim,
        "text_dim": encoder.text_dim,
        "query_dim": encoder.query_dim,
        "init_seed": encoder.init_seed,
        "state_dict": encoder.state_dict(),
        "provenance": encoder.provenance().to_dict(),
    }


def save_query_encoder_checkpoint(path: str, encoder: ComposeQueryEncoder) -> None:
    """Atomically persist the query encoder (mkstemp + fsync + replace)."""
    import os
    import tempfile
    from pathlib import Path

    payload = query_encoder_state_dict(encoder)
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


def load_query_encoder_checkpoint(
    path: str, encoder: Optional[ComposeQueryEncoder] = None
) -> Dict[str, object]:
    """Load a query encoder checkpoint.

    When ``encoder`` is None a new encoder is created from the checkpoint's
    recorded seed (deterministic init), so repeated loads of the same
    checkpoint always produce identical parameters.
    """
    from pathlib import Path

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(
            "query encoder checkpoint does not exist: {}".format(target)
        )
    state = torch.load(target, map_location="cpu", weights_only=False)
    if int(state.get("checkpoint_version", -1)) != QUERY_ENCODER_VERSION:
        raise ValueError(
            "unsupported query encoder checkpoint_version: {}".format(
                state.get("checkpoint_version")
            )
        )
    if state.get("kind") != "compose_query_encoder":
        raise ValueError(
            "checkpoint is not a compose_query_encoder: {!r}".format(state.get("kind"))
        )
    if encoder is None:
        encoder = ComposeQueryEncoder(
            visual_dim=int(state["visual_dim"]),
            text_dim=int(state["text_dim"]),
            query_dim=int(state["query_dim"]),
            seed=int(state["init_seed"]),
            initialize=True,
        )
    encoder.load_state_dict(state["state_dict"])
    return {
        "provenance": state["provenance"],
        "visual_dim": int(state["visual_dim"]),
        "text_dim": int(state["text_dim"]),
        "query_dim": int(state["query_dim"]),
        "init_seed": int(state["init_seed"]),
    }
