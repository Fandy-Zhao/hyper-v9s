"""Sparse differentiable routing over a per-sample candidate set (spec §7).

One sample's routing row is

    [ historical recall (Top-C, wide on some steps) | all current candidates ]

so the row is *dense*: every sample carries the same number of slots and only
the historical ids differ.  That is what lets V9-S express ``C + M`` active
experts through the existing ``ComposeSelection``/``ComposeLinear`` path without
per-row padding and without a second forward pass per expert.

Routing is an independent sigmoid per expert::

    s_ik = cos(q_i, e[k]) / tau
    p_ik = sigmoid(s_ik - b_k)

``b_k`` is a learnable per-expert bias.  It is not cosmetic -- with all queries
of a task living in a narrow cone, a shared cosine scale cannot pull a
non-contributing expert down to ``p ~ 0.02`` on its own, and the bias is the
term that lets each expert's *propensity* be learned separately from its
*address*.  A softmax is deliberately not used: experts are not mutually
exclusive classes, and a sample may need none, one, or two.

**The gate that drives the forward is detached from the key graph.**  ``p`` is
still differentiable w.r.t. the keys -- that is what makes the contribution
measurable -- but the copy consumed by the composition is ``p.detach()``.  So
``L_ans`` reaches a key by exactly one route and it is not this one: the answer
influences keys only through contribution -> responsibility -> ``L_key``.  See
``V9RoutingConfig.direct_answer_gradient_to_key`` for why the two must not run
at once.

One consequence is worth stating plainly, because it removes machinery rather
than adding it: the straight-through estimator in the discretisation stage
exists to pass an answer gradient *to the gates*.  When the forward gate carries
no key gradient by construction, ``hard - p.detach() + p`` with ``p`` already
detached collapses to ``hard``, which is exactly the deployed Top-2 rule.  The
final stage is therefore literally the inference rule, not a surrogate for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from compose.adapters.types import PAD_EXPERT_ID, ComposeSelection

from .config import V9_METHOD_TITLE, V9Config
from .keys import V9KeyPool
from .retrieval import compose_candidate_rows
from .schedule import STAGE_BOOTSTRAP, STAGE_HARD


class V9RouterError(RuntimeError):
    """Raised when the router cannot build a legal routing row."""


@dataclass
class V9RouteOutput:
    """One micro-step's routing decision, with the graph still attached."""

    expert_ids: torch.Tensor      # [B, S] long; PAD_EXPERT_ID in empty slots
    probabilities: torch.Tensor   # [B, S] float; differentiable sigmoid gates
    forward_gates: torch.Tensor   # [B, S] float; what the composition consumes
    slot_mask: torch.Tensor       # [B, S] bool; a real expert sits here
    hard_gates: Optional[torch.Tensor]  # [B, S] 0/1, only in the hard stage
    temperature: float
    stage: str
    #: Whether this step used the periodic wide recall.
    wide: bool = False

    @property
    def answer_gates(self) -> torch.Tensor:
        """The gate tensor the answer loss is a function of.

        This is what the contribution gradient is taken against, and it is the
        *forward* gate, not the key-differentiable one.  The two hold the same
        value, but only this one was consumed by the composition -- a derivative
        taken against the other is a derivative against a tensor the answer
        never saw, and it is identically zero.

        The name says whose gate it is rather than how it was built: the
        bootstrap stage floors it and the hard stage drives the forward with the
        Top-2 indicator, so "soft" describes it on one stage out of three.
        """
        return self.forward_gates

    @property
    def batch_size(self) -> int:
        return int(self.expert_ids.shape[0])

    @property
    def slots(self) -> int:
        return int(self.expert_ids.shape[1])

    def active_counts(self) -> torch.Tensor:
        """Expected number of active experts per row, from the forward gates."""
        return (self.forward_gates.detach() * self.slot_mask).sum(dim=1)

    def selection(self) -> ComposeSelection:
        """The ``ComposeSelection`` this route drives.

        Rows always keep their full slot width.  In the hard stage the
        non-selected experts carry a gate of exactly zero, which the composition
        masks out -- so the forward result is the deployed Top-2 while the
        straight-through surrogate keeps every slot in the backward graph.
        """
        return ComposeSelection(
            expert_ids=self.expert_ids,
            gates=self.forward_gates,
            max_slots=self.slots,
            allow_zero_gates=True,
        )

    def metadata(self) -> Dict[str, object]:
        with torch.no_grad():
            return {
                "temperature": float(self.temperature),
                "stage": self.stage,
                "wide_recall": bool(self.wide),
                "live_slots": int(self.slot_mask.sum().item()),
                "mean_active_experts": float(
                    self.active_counts().mean().item()
                ),
                "mean_probability": float(
                    (self.probabilities.detach() * self.slot_mask)
                    .sum()
                    .item()
                    / max(int(self.slot_mask.sum().item()), 1)
                ),
            }


class V9Router(nn.Module):
    """Holds the learnable routing parameters and builds the selection.

    The key pool is a submodule, so every routing key is reachable from
    ``model.parameters()`` (which is what makes the trainable-parameter audit
    and the optimizer allowlist meaningful).  The pool's ordering is fixed for
    the whole task, so the routing path never synchronises to discover which
    experts exist.
    """

    def __init__(
        self,
        config: V9Config,
        key_pool: V9KeyPool,
        candidate_ids: Sequence[int],
        historical_ids: Sequence[int],
        task_index: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.key_pool = key_pool
        self.task_index = int(task_index)
        self.candidate_ids = [int(value) for value in candidate_ids]
        self.historical_ids = [int(value) for value in historical_ids]
        order = self.historical_ids + self.candidate_ids
        if len(set(order)) != len(order):
            raise V9RouterError("an expert id appears in two routing blocks")
        self._order = order
        self._position = {expert_id: index for index, expert_id in enumerate(order)}
        self.register_buffer(
            "_lookup",
            torch.full((max(order) + 1 if order else 1,), -1, dtype=torch.long),
            persistent=False,
        )
        for expert_id, index in self._position.items():
            self._lookup[expert_id] = index
        # One dense bias vector in ``_order`` order rather than one parameter
        # per expert: the gather is a single op, the parameter is always in the
        # graph (so DDP never sees it as unused), and the optimizer group is one
        # tensor instead of one entry per expert.
        bias = torch.full((len(order),), float(config.routing.bias_init))
        if config.routing.learnable_bias:
            self.bias = nn.Parameter(bias)
        else:
            self.register_buffer("bias", bias, persistent=False)
        self.last_route: Optional[V9RouteOutput] = None
        #: Actual routing-row width, fixed the first time a micro-batch is
        #: routed.  It equals the configured width whenever the pool has enough
        #: historical experts; on task 0 (empty history) it is just ``M``.
        self._route_slots: Optional[int] = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @property
    def candidate_count(self) -> int:
        return len(self.candidate_ids)

    @property
    def slots(self) -> int:
        """Configured routing-row width; the realised width may be smaller.

        ``_route_slots`` is preferred wherever the real selection matters.
        """
        return int(self._route_slots or self.config.selection_slots)

    def candidate_key_ids(self) -> List[str]:
        return [self.key_pool.origin_key_id(e) for e in self.candidate_ids]

    def task_key_ids(self) -> List[str]:
        """Current-task keys held by historical experts -- all trained by L_key."""
        return [
            self.key_pool.task_key_id(expert_id, self.task_index)
            for expert_id in self.historical_ids
            if self.key_pool.has_task_key(expert_id, self.task_index)
        ]

    def routing_key_ids(self) -> List[str]:
        """Key ids in ``self._order`` order -- the router's parameter ordering."""
        return [
            self.key_pool.routing_key_id(expert_id, self.task_index)
            for expert_id in self._order
        ]

    def _bias_vector(self, device: torch.device) -> torch.Tensor:
        return self.bias.to(device=device, dtype=torch.float32)

    def bias_state(self) -> Dict[str, float]:
        """Per-expert routing bias, by expert id -- for logs and checkpoints."""
        with torch.no_grad():
            return {
                str(expert_id): float(self.bias[index].item())
                for index, expert_id in enumerate(self._order)
            }

    def _expert_columns(self, expert_ids: torch.Tensor) -> torch.Tensor:
        """Map expert ids to router columns, with PAD collapsing to column 0."""
        position = self._lookup.to(expert_ids.device)
        safe = expert_ids.clamp_min(0)
        if int(safe.max().item()) >= int(position.numel()):
            raise V9RouterError(
                "expert id {} is outside the router's fixed ordering".format(
                    int(safe.max().item())
                )
            )
        return position.index_select(0, safe.reshape(-1)).reshape(expert_ids.shape)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def route(
        self,
        queries: torch.Tensor,
        historical_rows: torch.Tensor,
        temperature: float,
        stage: str,
        wide: bool = False,
    ) -> V9RouteOutput:
        """Build one micro-step's routing row and its gate tensor.

        ``wide`` selects the periodic wide recall: the row keeps its full cached
        width, but only the leading ``wide_top_c`` historical columns are live.
        Every rank computes ``wide`` from ``(seed, step)`` alone, so the ranks
        agree on the row without a collective.
        """
        if queries.ndim != 2:
            raise V9RouterError("queries must be [batch, query_dim]")
        rows = int(queries.shape[0])
        expert_ids = compose_candidate_rows(
            historical_rows.to(queries.device), self.candidate_ids
        )
        if int(expert_ids.shape[0]) != rows:
            raise V9RouterError(
                "historical recall rows ({}) do not match the query batch ({})".format(
                    int(expert_ids.shape[0]), rows
                )
            )
        width = int(expert_ids.shape[1])
        if self._route_slots is None:
            self._route_slots = width
        elif self._route_slots != width:
            raise V9RouterError(
                "routing row width changed mid-task ({} -> {}); the historical "
                "recall block must be fixed for the whole task".format(
                    self._route_slots, width
                )
            )
        slot_mask = expert_ids.ne(PAD_EXPERT_ID)
        historical_width = max(width - len(self.candidate_ids), 0)
        active_history = min(
            int(self.config.active_historical_slots(wide)), historical_width
        )
        if active_history < historical_width:
            # The masked tail is empty by construction -- it holds ids the recall
            # did compute -- so masking it is a compute decision, not a data one.
            live = torch.zeros(width, dtype=torch.bool, device=expert_ids.device)
            live[:active_history] = True
            live[historical_width:] = True
            slot_mask = slot_mask & live.unsqueeze(0)
        duplicate = (
            expert_ids.unsqueeze(2) == expert_ids.unsqueeze(1)
        ) & slot_mask.unsqueeze(2) & slot_mask.unsqueeze(1)
        if bool(torch.triu(duplicate, diagonal=1).any().item()):
            raise V9RouterError(
                "a routing row contains the same expert twice; the historical "
                "recall and the candidate block must be disjoint"
            )

        key_matrix = self.key_pool.effective_key_matrix(
            self.routing_key_ids(), detach=False
        ).to(queries.device)
        query_matrix = F.normalize(queries.float(), dim=-1)
        columns = self._expert_columns(expert_ids)    # [B, S]
        bias_vector = self._bias_vector(queries.device)
        tau = max(float(temperature), 1e-6)

        def _gate(keys: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            slot_cosine = (query_matrix @ keys.T).gather(1, columns)
            slot_bias = bias.gather(0, columns.reshape(-1)).reshape(expert_ids.shape)
            gate = torch.sigmoid(slot_cosine / tau - slot_bias)
            return torch.where(slot_mask, gate, torch.zeros_like(gate))

        # Two gates, one value.  ``probabilities`` is differentiable w.r.t. the
        # key vectors, and it is the tensor ``L_key`` trains.  The composition
        # consumes the *other* one, built from the same cosine against detached
        # keys and a detached bias: identical numbers, but no parameter sits
        # behind it.
        #
        # The two must be built this way round and not by detaching a shared
        # result.  ``p.detach()`` severs the loss from ``p`` entirely -- the
        # answer would then be a function of no gate at all, ``dL_ans/da`` would
        # be identically zero, and the responsibility would have no signal to
        # distil.  Building the forward gate from detached *inputs* keeps the
        # answer differentiable w.r.t. the gate while leaving the keys outside
        # that graph, which is the whole contract: ``L_ans`` judges the gate, the
        # gate judges the keys, and the two never touch directly.
        probabilities = _gate(key_matrix, bias_vector)
        forward_probabilities = _gate(key_matrix.detach(), bias_vector.detach())
        # Built from constants, the gate is a *leaf* with no graph, so autograd
        # would report no gradient for it -- a constant has none.  Marking it as
        # an input variable is what makes the answer differentiable w.r.t. the
        # gate while leaving the keys outside: the derivative is taken against
        # this leaf and stops at it.  Without this line the contribution is
        # identically zero and every key stays where k-means left it.
        forward_probabilities.requires_grad_(True)

        candidate_slots = torch.isin(
            expert_ids, torch.tensor(self.candidate_ids, device=expert_ids.device)
        )
        forward_gates, hard = self._forward_gates(
            forward_probabilities, slot_mask, stage, differentiable=self._detach_keys(),
            candidate_slot_mask=candidate_slots,
        )
        self.last_route = V9RouteOutput(
            expert_ids=expert_ids,
            probabilities=probabilities,
            forward_gates=forward_gates,
            slot_mask=slot_mask,
            hard_gates=hard,
            temperature=float(tau),
            stage=str(stage),
            wide=bool(wide),
        )
        return self.last_route

    def _detach_keys(self) -> bool:
        """Whether the forward gate must be cut loose from the key graph."""
        return not bool(self.config.routing.direct_answer_gradient_to_key)

    def _forward_gates(
        self,
        probabilities: torch.Tensor,
        slot_mask: torch.Tensor,
        stage: str,
        differentiable: bool = True,
        candidate_slot_mask: Optional[torch.Tensor] = None,
    ) -> "tuple[torch.Tensor, Optional[torch.Tensor]]":
        """Stage-dependent forward gate (spec §15, §16).

        ``bootstrap`` raises every valid gate to the exposure floor.  A newly
        created expert has ``Delta_theta ~ 0``, hence ``dL/da_new ~ 0``: without
        the floor it can never earn the answer gradient that would give it
        capability.  The floor is an exposure device only -- it breaks symmetry,
        it does not assign a permanent expert boundary.

        ``hard`` is the deployed Top-2 indicator.  In the hard stage the
        composition must *forward* that indicator while remaining differentiable
        w.r.t. the gate, so the straight-through form
        ``hard - p.detach() + p`` is used: its value is the deployed rule, its
        derivative w.r.t. ``p`` is the identity, and ``p`` here is already built
        from detached keys.  Feeding the sigmoid itself would serve a mixture
        the deployment never serves; feeding ``hard`` alone would leave the last
        stage of training with ``dL_ans/da = 0`` and therefore with no
        responsibility to distil at all.

        ``probabilities`` is the gate the composition will consume, and it is
        parameter-free by construction when ``direct_answer_gradient_to_key`` is
        false -- which the config refuses to set otherwise.  ``differentiable``
        is the same flag read at this level: with the contract in force there is
        nothing to detach, because nothing was ever attached.
        """
        if stage == STAGE_HARD:
            k = min(
                int(self.config.routing.max_inference_experts),
                int(probabilities.shape[1]),
            )
            top = torch.topk(probabilities.detach(), k, dim=1).indices
            hard = torch.zeros_like(probabilities)
            hard.scatter_(1, top, 1.0)
            hard = torch.where(slot_mask, hard, torch.zeros_like(hard))
            if not differentiable:
                return hard, hard
            return hard - probabilities.detach() + probabilities, hard
        gates = probabilities
        if stage == STAGE_BOOTSTRAP:
            floor = float(self.config.bootstrap.gate_floor)
            if floor > 0:
                candidate_mask = candidate_slot_mask if candidate_slot_mask is not None else slot_mask
                gates = torch.where(
                    slot_mask,
                    torch.where(candidate_mask, gates.clamp_min(floor), gates),
                    torch.zeros_like(gates),
                )
        return gates, None

    def deployed_gates(
        self, probabilities: torch.Tensor, slot_mask: torch.Tensor
    ) -> torch.Tensor:
        """The gates the deployed rule would apply to an already-scored row.

        Deployment is Top-``max_inference_experts`` over the gate probability,
        with no answer, no task id and no training recall -- the same rule
        ``_forward_gates`` applies in the hard stage.  Exposing it separately
        lets the calibration measure the *deployed* answer loss on held-out data
        and compare it with the soft-gate loss the training objective actually
        minimises (spec §34); a gap that grows without bound means the two have
        come apart and the soft gates no longer predict what will be served.
        """
        hard, _ = self._forward_gates(
            probabilities.detach(), slot_mask, STAGE_HARD, differentiable=False
        )
        return hard

    # ------------------------------------------------------------------
    # trainability
    # ------------------------------------------------------------------
    def trainable_parameters(self) -> Dict[str, List[nn.Parameter]]:
        """The router's own trainable parameters, grouped for the optimizer."""
        bias_parameters = (
            [self.bias] if self.bias.requires_grad and self.bias in self.parameters() else []
        )
        key_parameters = [
            self.key_pool.keys[key_id]
            for key_id in self.key_pool.trainable_key_ids()
        ]
        return {"bias": bias_parameters, "key": key_parameters}

    def audit(self) -> Dict[str, object]:
        route = self.last_route
        return {
            "task_index": self.task_index,
            "routing_type": self.config.routing.type,
            "method_title": V9_METHOD_TITLE,
            "slots": self.slots,
            "candidate_ids": list(self.candidate_ids),
            "historical_ids": list(self.historical_ids),
            "trainable_task_key_ids": self.task_key_ids(),
            "historical_top_c": int(self.config.historical_retrieval.top_c),
            "wide_retrieval": {
                "enabled": bool(self.config.wide_retrieval.enabled),
                "ratio": float(self.config.wide_retrieval.ratio),
                "top_c": int(self.config.wide_retrieval.top_c),
            },
            "learnable_bias": bool(self.config.routing.learnable_bias),
            "direct_answer_gradient_to_key": bool(
                self.config.routing.direct_answer_gradient_to_key
            ),
            "forward_gates_detached_from_keys": self._detach_keys(),
            "last_temperature": None if route is None else route.temperature,
            "last_stage": None if route is None else route.stage,
            "last_wide_recall": None if route is None else bool(route.wide),
        }


__all__ = ["V9RouteOutput", "V9Router", "V9RouterError"]
