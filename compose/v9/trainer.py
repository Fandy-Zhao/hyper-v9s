"""V9 training step: one forward, one gate gradient, one backward.

The whole method is this loop, once per micro-batch:

    route  ->  build selection  ->  ONE backbone forward with every candidate
    branch  ->  ground-truth answer NLL  ->  ``autograd.grad(L_ans, gates)``
    ->  ``G = -a * dL/da`` (stop-gradient)  ->  responsibility  ->  ``L_key``
    ->  ONE backward on ``L_total``

There is no per-expert answer enumeration and no pair enumeration anywhere in
this path.  The exact remove-and-reroute quantity exists only in
:meth:`V9ComposeTrainer.calibration_pass` -- a validation-side diagnostic over a
handful of held-out samples (spec §30) that no gradient ever flows through.

The gate gradient is taken with ``create_graph=False``: it is a teacher target,
so differentiating through it would buy nothing and cost a full second-order
graph (spec §11).  The routing graph is a *subgraph* of the answer graph, so the
single subsequent ``backward`` credits the answer to the expert LoRA (Path A)
and to the routing keys (Path B) together (spec §9).

Per-slot task statistics (usage / support / contribution) are accumulated on
device with ``index_add_`` and reduced once at task end: no per-slot ``.item()``
and no CPU round-trip sits in the micro-batch path (spec §25, §34).
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from compose.adapters.types import ComposeSelection
from compose.lora.rms import runtime_kappa_calibration
from compose.train.profiler import TrainingProfiler, wrap_optimizer_step
from compose.train.trainer import ComposeTrainer
from compose.v7.hf_trainer import (
    _distributed,
    candidate_lora_state,
    load_candidate_lora_state,
    mean_sync_accumulated_gradients,
    ranked_metrics_paths,
)
from compose.v7.training import (
    V7JsonlLogger,
    adapter_checksums,
    assert_historical_lora_frozen,
    full_data_coverage_audit,
)
from compose.v8.pool import LIFECYCLE_PRUNED, tensor_checksum

from .checkpoint import load_v9_checkpoint, save_v9_checkpoint
from .config import V9Config
from .contribution import (
    V9Contribution,
    answer_derived_responsibility,
    calibration_report,
    contribution_statistics,
    exact_removal_contribution,
    gate_gradient,
    local_conditional_contribution,
    pair_rerank_report,
)
from .keys import V9KeyPool
from .losses import compose_total_loss
from .retrieval import is_wide_step
from .router import PAD_EXPERT_ID, V9RouteOutput, V9Router
from .schedule import V9StageScheduler, V9StageState


def assert_no_historical_key_gradients(pool: V9KeyPool) -> None:
    """Fail fast if a Base Key or an earlier task's key carries a gradient.

    The check is ordered so the common case is free: a frozen key has
    ``requires_grad=False`` and therefore never has a ``.grad`` at all, so the
    device-synchronising ``.any()`` only runs once something has already gone
    wrong.
    """
    offenders = []
    for key_id in pool.historical_key_ids():
        parameter = pool.keys[key_id]
        if parameter.requires_grad:
            raise AssertionError(
                "historical routing key {} is trainable".format(key_id)
            )
        if parameter.grad is not None and bool(parameter.grad.detach().ne(0).any()):
            offenders.append(key_id)
    if offenders:
        raise AssertionError(
            "historical routing keys received gradient: {}".format(sorted(offenders)[:10])
        )


class V9TrainerError(RuntimeError):
    """Raised when the V9-S training loop cannot honour its own contract."""


def answer_loss_key_gradient(
    answer_loss: torch.Tensor, key_parameters: Mapping[str, torch.nn.Parameter]
) -> Dict[str, float]:
    """``max |dL_ans/dkey|`` per key id -- the isolation evidence.

    A correct V9-S graph returns zeros everywhere: the forward gate is detached
    from the key graph, so nothing the answer loss is computed from depends on a
    key.  Used once per task as a live assertion on the real training graph, and
    in the gradient unit tests, so the claim "``L_ans`` does not update any Key"
    is *checked* rather than asserted in prose.

    A frozen key is reported as ``0.0`` **without being differentiated**, and it
    is worth being exact about what that number then means.  ``torch.autograd.grad``
    raises ``RuntimeError: One of the differentiated Tensors does not require
    grad`` for such an input, because autograd cannot deliver a gradient to a
    tensor that is not a differentiable leaf -- there is no edge to carry one.
    Differentiating the whole key set naively therefore crashed the first task
    that had a historical expert to freeze, on its first training step: task 0
    has no frozen key at all, so no task-0 preflight could reach it.  What is
    differentiable is measured here; what is frozen is zero because the graph
    has no edge to it, and the complementary half of that claim -- that the key
    really is frozen -- is checked independently by the freeze audit
    (``assert_historical_keys_frozen``: ``requires_grad`` False and a null
    ``.grad``), which is also what makes the two halves cover the whole key set.
    """
    ids = list(key_parameters)
    if not ids:
        return {}
    trainable = [key_id for key_id in ids if key_parameters[key_id].requires_grad]
    report = {key_id: 0.0 for key_id in ids}
    if not trainable:
        return report
    grads = torch.autograd.grad(
        outputs=answer_loss,
        inputs=[key_parameters[key_id] for key_id in trainable],
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    for key_id, gradient in zip(trainable, grads):
        report[key_id] = (
            0.0 if gradient is None
            else float(gradient.detach().abs().max().item())
        )
    return report


class V9ComposeTrainer(ComposeTrainer):
    """Answer-guided responsibility distillation under the DDP contract."""

    def __init__(
        self,
        *args,
        v9_config: V9Config,
        v9_router: V9Router,
        v9_task_index: int,
        v9_metrics_path: str,
        v9_total_steps: int,
        v9_require_full_coverage: bool = False,
        v9_profiler=None,
        **kwargs,
    ) -> None:
        self.profiler = v9_profiler if v9_profiler is not None else TrainingProfiler(None)
        self.v9_config = v9_config
        self.v9_router = v9_router
        self.v9_key_pool: V9KeyPool = v9_router.key_pool
        self.v9_task_index = int(v9_task_index)
        self.v9_require_full_coverage = bool(v9_require_full_coverage)
        # The stage schedule is a set of *ratios* of the run, so it cannot be
        # built until the run's length is known -- and the length depends on the
        # dataloader the trainer only builds later.  A non-positive total here
        # therefore means "not resolved yet", not "zero steps": the entry point
        # calls ``set_total_steps`` before the first step.  Building a scheduler
        # with a guessed total would silently move every stage boundary, which
        # is the failure mode the ratios exist to prevent.
        self.v9_total_steps = int(v9_total_steps)
        self.v9_scheduler: Optional[V9StageScheduler] = None
        if self.v9_total_steps >= 1:
            self.v9_scheduler = V9StageScheduler(
                v9_config.schedule, v9_config.routing, self.v9_total_steps
            )
        world_size = torch.distributed.get_world_size() if _distributed() else 1
        rank = torch.distributed.get_rank() if _distributed() else 0
        self.v9_metrics_paths = ranked_metrics_paths(Path(v9_metrics_path), world_size)
        self.v9_logger = V7JsonlLogger(str(self.v9_metrics_paths[rank]))

        # ---- per-expert accumulators (device tensors, spec §23/§25) ----
        self.v9_order = list(v9_router._order)
        self.v9_index = {
            expert_id: index for index, expert_id in enumerate(self.v9_order)
        }
        self.v9_micro_steps = 0
        self.v9_noop_steps = 0
        #: Gradient-window bookkeeping.  ``_v9_last_micro_was_boundary`` and
        #: ``_v9_global_step_seen`` exist so that a boundary this trainer fails
        #: to notice raises instead of showing up as two ranks that quietly stop
        #: agreeing; see ``_at_sync_boundary``.
        self._v9_last_micro_was_boundary = False
        self._v9_global_step_seen: Optional[int] = None
        self.v9_unsynced_optimizer_steps = 0
        self.v9_observed_sample_count = 0
        self.v9_unique_sample_ids = set()
        self.v9_stage_steps: Dict[str, int] = {}
        self.v9_extra_forwards = 0
        #: Main-backbone forwards, counted where they actually happen.  §41 (D)
        #: asks for the forwards *per training sample*, which is the number that
        #: says whether the offered experts really do ride on one pass.
        self.v9_backbone_forwards = 0
        #: Backbone traversals counted by a forward hook on the model module
        #: itself, independent of this trainer's own bookkeeping.
        #: ``v9_backbone_forwards`` records what the trainer *believes* it did;
        #: these record what the model was actually asked to do, and §41 (D)'s
        #: invariant -- one traversal per micro-step, however many experts a row
        #: offers -- is precisely the claim that the two agree.  A future
        #: ``for expert: model(...)`` loop would leave the first counter at one
        #: and multiply the second, which is the regression the pair exists to
        #: catch.  The per-scope counters separate the training pass (the
        #: invariant) from the bounded held-out calibration, whose extra
        #: forwards are legitimate *measurement* and are counted apart rather
        #: than folded in.
        self.v9_model_forwards = 0
        self.v9_train_model_forwards = 0
        self.v9_calibration_model_forwards = 0
        #: Traversals observed on a *wide* recall step, and the number of such
        #: micro-steps.  Wide retrieval widens the routing row; it must widen
        #: the LoRA branch set and nothing else, so the ratio below is checked
        #: against the narrow one rather than assumed equal.
        self.v9_wide_model_forwards = 0
        self.v9_wide_micro_steps = 0
        self._v9_forward_handle = None
        self._v9_forward_scope: Optional[str] = None
        #: Soft-gate vs deployed-Top-2 answer NLL on held-out data (spec §34).
        self.v9_validation_scores: Optional[Dict[str, float]] = None
        #: Optimizer steps that used the periodic wide recall.
        self.v9_wide_steps = 0
        #: Pair set measured by the spec §35 ablation, fixed on the first
        #: calibration batch so a column means the same pair throughout.
        self._v9_pair_probe: Optional[List[Tuple[int, int]]] = None
        self.v9_calibration: Optional[Dict[str, float]] = None
        self.v9_validation_gain: Dict[int, float] = {}
        self._v9_accumulators: Optional[Dict[str, torch.Tensor]] = None
        self._v9_active = None
        self._v9_active_ids: Optional[torch.Tensor] = None
        self._v9_losses = None
        self._v9_last_sample_ids: Sequence[str] = ()
        self._v9_step_timing = None
        self._v9_previous_step_end = None
        self._v9_audited_first_step = False
        self._v9_pending_optimizer_state = None
        self._v9_pending_scheduler_state = None
        self._gate_grad_abs_sum: Optional[torch.Tensor] = None
        self._gate_grad_count = 0
        self._v9_checked_answer_key_isolation = False
        self.v9_answer_key_isolation: Optional[Dict[str, Any]] = None
        super().__init__(*args, **kwargs)
        self._assert_router_registered()
        self._historical_key_before = self.v9_key_pool.historical_checksums()
        self._historical_lora_before = adapter_checksums(
            self.expert_pool.manager, self.v9_key_pool.historical_ids
        )
        self._historical_rms_before = runtime_kappa_calibration(
            self.model, self.v9_key_pool.historical_ids
        )

    # ------------------------------------------------------------------
    # registration / optimizer
    # ------------------------------------------------------------------
    def set_total_steps(self, total_steps: int) -> None:
        """Fix the schedule once the real dataloader length is known.

        Called by the entry point after the trainer exists, because the number
        of optimizer steps depends on the sampler the trainer actually builds.
        Safe to call before training only: it re-derives every stage boundary.
        """
        if int(self.state.global_step) not in (0,):
            raise RuntimeError(
                "the V9-S stage schedule cannot be re-derived after training has "
                "started (global_step={})".format(int(self.state.global_step))
            )
        self.v9_total_steps = max(int(total_steps), 1)
        self.v9_scheduler = V9StageScheduler(
            self.v9_config.schedule, self.v9_config.routing, self.v9_total_steps
        )

    def _assert_answer_loss_isolated_from_keys(
        self, answer_loss: torch.Tensor, route: V9RouteOutput
    ) -> None:
        """Prove on the live graph that ``L_ans`` touches no key parameter.

        Every key the router can route by is covered -- candidates, historical
        current-task keys, and the frozen base keys of the historical experts --
        so the report is about the whole key set, not just the trainable subset.
        The trainable keys are *measured*; a frozen key is reported as zero
        because it is not a differentiable leaf and no edge can reach it, and
        ``frozen_keys_not_differentiated`` names those, so the report never
        presents a structural zero as a measurement.  A nonzero entry means the
        forward gate was not detached, or something else reintroduced a key into
        the answer path; either way the run must stop before it trains a key with
        a second, conflicting signal.
        """
        parameters: Dict[str, torch.nn.Parameter] = {}
        for key_id in self.v9_key_pool.key_ids():
            if self.v9_key_pool.key_records[key_id]["lifecycle"] == LIFECYCLE_PRUNED:
                continue
            parameter = self.v9_key_pool.keys[key_id]
            if parameter.requires_grad or parameter.is_leaf:
                parameters[key_id] = parameter
        report = answer_loss_key_gradient(answer_loss, parameters)
        offenders = {key_id: value for key_id, value in report.items() if value > 0.0}
        frozen = sorted(
            key_id for key_id, parameter in parameters.items() if not parameter.requires_grad
        )
        self.v9_answer_key_isolation = {
            "keys_checked": len(report),
            "measured_keys": len(report) - len(frozen),
            "frozen_keys_not_differentiated": frozen,
            "max_abs_gradient": max(report.values()) if report else 0.0,
            "nonzero_keys": sorted(offenders)[:10],
            "gate_detached": self.v9_router._detach_keys(),
            # The forward gate *must* carry a graph -- the answer has to be
            # differentiable w.r.t. it or there is no contribution to measure.
            # What must be absent is a key behind that graph, which is what
            # ``max_abs_gradient`` above reports.
            "answer_gate_requires_grad": bool(route.answer_gates.requires_grad),
        }
        if offenders:
            raise AssertionError(
                "L_ans reaches {} key parameter(s); the forward gate must be "
                "detached so that responsibility is the only answer-side "
                "supervision of a key. First offenders: {}".format(
                    len(offenders), dict(list(sorted(offenders.items()))[:5])
                )
            )

    def _stage_state(self, global_step: int) -> V9StageState:
        """The stage/temperature for a step, refusing to guess the schedule."""
        if self.v9_scheduler is None:
            raise V9TrainerError(
                "the stage schedule has no total step count; the entry point "
                "must call set_total_steps() before training begins"
            )
        return self.v9_scheduler.state(int(global_step))

    def _assert_router_registered(self) -> None:
        """The router must be a submodule of the model, not a free object.

        DDP only synchronizes parameters reachable from the wrapped module, and
        the trainable-parameter audit reads ``model.named_parameters()``; a
        router held off to the side would train unsynchronised.
        """
        names = {name for name, _ in self.model.named_parameters()}
        missing = [
            name
            for name, _ in self.v9_router.named_parameters()
            if "v9_router.{}".format(name) not in names
        ]
        if missing:
            raise AssertionError(
                "the V9 router is not registered on the model; missing {}".format(
                    missing[:5]
                )
            )

    def v9_training_config(self) -> Dict[str, float]:
        """Resolved HF-side hyperparameters, so nothing is hard-coded here.

        The two learning rates live on ``TrainingArguments`` because that is how
        the rest of the repository configures warmup, decay and clipping.
        """
        return {
            "lora_learning_rate": float(getattr(self.args, "learning_rate", 2e-4)),
            "key_learning_rate": float(getattr(self.args, "v9_key_learning_rate", 3e-4)),
        }

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        groups = self.v9_router.trainable_parameters()
        key_parameters = list(groups["key"])
        bias_parameters = list(groups["bias"])
        lora_parameters = [
            parameter
            for expert_id in self.v9_key_pool.current_ids
            for layer in self.expert_pool.manager.layers.values()
            for parameter in layer.experts[str(int(expert_id))].parameters()
        ]
        allowed = {
            id(value) for value in key_parameters + bias_parameters + lora_parameters
        }
        unexpected = [
            name
            for name, value in self.model.named_parameters()
            if value.requires_grad and id(value) not in allowed
        ]
        if unexpected:
            raise AssertionError(
                "V9 optimizer refuses trainable parameters outside the candidate "
                "LoRA, the candidates' keys, the historical experts' current-task keys "
                "and the "
                "routing bias: {}".format(unexpected[:10])
            )
        training = self.v9_training_config()
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": lora_parameters,
                    "lr": training["lora_learning_rate"],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": key_parameters + bias_parameters,
                    "lr": training["key_learning_rate"],
                    "weight_decay": 0.0,
                },
            ]
        )
        if self._v9_pending_optimizer_state is not None:
            self.optimizer.load_state_dict(self._v9_pending_optimizer_state)
            self._v9_pending_optimizer_state = None
        wrap_optimizer_step(self.optimizer, self.profiler)
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        scheduler = super().create_scheduler(num_training_steps, optimizer=optimizer)
        if self._v9_pending_scheduler_state is not None:
            scheduler.load_state_dict(self._v9_pending_scheduler_state)
            self._v9_pending_scheduler_state = None
        return scheduler

    # ------------------------------------------------------------------
    # training step
    # ------------------------------------------------------------------
    def _at_sync_boundary(self) -> bool:
        """True on the micro-step that closes an optimizer step.

        The boundary is ``accelerator.sync_gradients``, which HF's loop really
        does drive: ``create_accelerator_and_postprocess`` builds the
        ``GradientAccumulationPlugin`` with ``num_steps =
        gradient_accumulation_steps``, so accelerate sets the flag once per
        window.  V8 reads this same flag.

        The V9-S v1 draft replaced it with ``(global_step + 1) % accumulation ==
        0``, on the theory that the flag was stuck True.  It is not, and the
        replacement reads a counter that **only advances at the boundary it is
        trying to detect**: ``global_step`` is constant across a window, so the
        expression was constant too -- it fired on all eight micro-steps of one
        window in every eight, and on none of the other seven.  The optimizer
        therefore stepped 44 times out of 50 on each rank's own local gradients.
        The task-end audit caught the result as differing candidate-LoRA
        checksums across ranks, which is what that audit is for.

        The second half of this method is the reason the mistake cannot recur
        silently: a boundary the trainer fails to notice shows up as an
        optimizer step that happened on a micro-step we did not treat as a
        window end, and that raises.
        """
        boundary = bool(getattr(self.accelerator, "sync_gradients", True))
        step = int(self.state.global_step)
        if (
            self._v9_global_step_seen is not None
            and step != self._v9_global_step_seen
            and not self._v9_last_micro_was_boundary
        ):
            self.v9_unsynced_optimizer_steps += 1
            raise V9TrainerError(
                "optimizer step {} happened on a micro-step this trainer did not "
                "treat as a gradient window boundary; the ranks would take that "
                "step on unsynchronised local gradients".format(step)
            )
        self._v9_global_step_seen = step
        self._v9_last_micro_was_boundary = boundary
        return boundary

    # ------------------------------------------------------------------
    # one-backbone-forward invariant (§41 D)
    # ------------------------------------------------------------------
    def _count_backbone_forward(self, _module, _inputs, _output) -> None:
        """Forward hook on the model module; counts every traversal."""
        self.v9_model_forwards += 1
        scope = self._v9_forward_scope
        if scope == "train":
            self.v9_train_model_forwards += 1
        elif scope == "calibration":
            self.v9_calibration_model_forwards += 1

    def _attach_backbone_counter(self, model) -> None:
        """Hook the *unwrapped* module, which every path forwards through.

        The training step receives the DDP wrapper and the calibration reaches
        for ``self.model`` directly; DDP's own forward calls ``self.module``, so
        one hook on the unwrapped module sees both and neither path can acquire
        an uncounted traversal by choosing a different handle.
        """
        if self._v9_forward_handle is not None:
            return
        target = getattr(model, "module", model)
        self._v9_forward_handle = target.register_forward_hook(
            self._count_backbone_forward
        )

    @contextlib.contextmanager
    def _backbone_scope(self, scope: str):
        """Attribute the traversals inside this block to ``scope``."""
        previous = self._v9_forward_scope
        self._v9_forward_scope = scope
        try:
            yield
        finally:
            self._v9_forward_scope = previous

    def training_step(self, model, inputs):
        step_started = time.perf_counter()
        inter_step_wait = (
            None
            if self._v9_previous_step_end is None
            else step_started - self._v9_previous_step_end
        )
        self.profiler.micro_step()
        sample_ids = tuple(str(value) for value in inputs.get("sample_ids", ()))
        self._v9_last_sample_ids = sample_ids
        self.v9_unique_sample_ids.update(sample_ids)
        self.v9_observed_sample_count += len(sample_ids)
        self.v9_micro_steps += 1

        queries = inputs.get("fixed_queries")
        if queries is None:
            raise ValueError("a V9 batch must carry fixed_queries")
        device = next(self.v9_key_pool.parameters()).device
        queries = queries.to(device)
        inputs["fixed_queries"] = queries
        historical_rows = inputs.get("historical_topc")
        if historical_rows is None:
            raise ValueError("a V9 batch must carry historical_topc")
        token_stats = inputs.get("token_stats")
        if token_stats is not None:
            self.profiler.observe_batch(*token_stats)

        stage_state = self._stage_state(int(self.state.global_step))
        self.v9_stage_steps[stage_state.stage] = (
            self.v9_stage_steps.get(stage_state.stage, 0) + 1
        )
        # Wide recall is decided from (seed, global_step) alone, so every rank
        # agrees on the row width without a collective: a broadcast here would
        # be a per-step synchronisation to compute something each rank can
        # derive for itself.
        wide = is_wide_step(
            int(self.state.global_step),
            self.v9_config.wide_retrieval.ratio,
            seed=self.v9_config.seed,
        )
        if wide:
            self.v9_wide_steps += 1
            self.v9_wide_micro_steps += 1
        with self.profiler.timed("routing_time"):
            route = self.v9_router.route(
                queries,
                historical_rows,
                stage_state.temperature,
                stage_state.stage,
                wide=bool(wide),
            )
            selection = route.selection()
            self._v9_active = (queries, route, stage_state)
            self._accumulate_route_statistics(route)

        # ``no_sync`` suppresses DDP's per-micro-step all-reduce; the window is
        # reduced exactly once at the accumulation boundary below.  This is the
        # established V7 protocol: one flat mean-reduce per optimizer step with
        # an explicit presence mask, so ``grad=None`` is a global decision.
        no_sync = (
            model.no_sync()
            if _distributed() and hasattr(model, "no_sync")
            else contextlib.nullcontext()
        )
        # §41 (D), enforced rather than reported: whatever a row offers --
        # the narrow Top-C, the wide recall, every current candidate -- the
        # experts ride on ONE traversal.  The hook counts what the model was
        # asked to do, so a per-expert enumeration cannot pass by leaving the
        # trainer's own counter at one.
        self._attach_backbone_counter(model)
        traversals_before = self.v9_train_model_forwards
        with self._backbone_scope("train"):
            with no_sync:
                with self.profiler.timed("step_body_time"):
                    loss = self._ddp_training_step(model, inputs, selection)
        traversals = self.v9_train_model_forwards - traversals_before
        if traversals != 1:
            raise V9TrainerError(
                "one-backbone-forward invariant violated at micro-step {}: the "
                "model was traversed {} times for one micro-batch (expected 1). "
                "V9 requires one backbone pass carrying every offered expert; "
                "a per-expert full-model forward is the exact cost this method "
                "exists to avoid.".format(self.v9_micro_steps, traversals)
            )
        if wide:
            self.v9_wide_model_forwards += traversals

        boundary = self._at_sync_boundary()
        if boundary or not self._v9_audited_first_step:
            with self.profiler.timed("audit_time"):
                assert_historical_lora_frozen(
                    self.expert_pool.manager, self.v9_key_pool.historical_ids
                )
                assert_no_historical_key_gradients(self.v9_key_pool)
            self._v9_audited_first_step = True

        gradient_window_synced = False
        if boundary:
            if self.optimizer is None:
                raise RuntimeError("the V9 optimizer must exist before synchronization")
            with self.profiler.timed("allreduce_time"):
                gradient_window_synced = mean_sync_accumulated_gradients(
                    parameter
                    for group in self.optimizer.param_groups
                    for parameter in group["params"]
                )
        # The timing is assigned before the metrics are written, not after: the
        # metrics read this dict, so writing first logged the *previous*
        # boundary's timing -- one window late, and with the previous window's
        # ``gradient_window_synced`` in it.
        self._v9_step_timing = {
            "wall_time": time.time(),
            "training_step_sec": time.perf_counter() - step_started,
            "inter_step_wait_sec": inter_step_wait,
            "local_batch_size": len(sample_ids),
            "gradient_window_synced": gradient_window_synced,
        }
        if boundary:
            self._write_step_metrics()
        self._v9_previous_step_end = time.perf_counter()
        return loss

    # ------------------------------------------------------------------
    # on-device statistics
    # ------------------------------------------------------------------
    def _accumulators(self, device: torch.device) -> Dict[str, torch.Tensor]:
        if self._v9_accumulators is None or self._v9_accumulators["usage"].device != device:
            size = len(self.v9_order)
            self._v9_accumulators = {
                name: torch.zeros(size, dtype=torch.float32, device=device)
                for name in (
                    "usage",
                    "selected",
                    "support",
                    "contribution_sum",
                    "positive_contribution",
                    "positive_count",
                    "responsibility_sum",
                )
            }
        return self._v9_accumulators

    def _slot_index(self, expert_ids: torch.Tensor) -> torch.Tensor:
        """Map expert ids to accumulator rows, with PAD collapsing to row 0."""
        width = max(self.v9_order) + 1 if self.v9_order else 1
        lookup = torch.full(
            (width,), -1, dtype=torch.long, device=expert_ids.device
        )
        for expert_id, index in self.v9_index.items():
            lookup[expert_id] = index
        safe = expert_ids.clamp_min(0)
        if int(safe.max().item()) >= width:
            raise AssertionError(
                "a routed expert id is outside the router's fixed ordering"
            )
        return lookup.index_select(0, safe.reshape(-1)).reshape(expert_ids.shape)

    def _accumulate_route_statistics(self, route: V9RouteOutput) -> None:
        """Exposure counters.  Local only -- ``all_reduce`` happens at task end."""
        with torch.no_grad():
            accumulators = self._accumulators(route.expert_ids.device)
            self._v9_active_ids = route.expert_ids
            ids = self._slot_index(route.expert_ids)[route.slot_mask]
            gates = route.forward_gates.detach()[route.slot_mask]
            accumulators["usage"].index_add_(0, ids, torch.ones_like(gates))
            accumulators["selected"].index_add_(
                0, ids, (gates > 0.5).to(gates.dtype)
            )
            accumulators["support"].index_add_(0, ids, gates)

    def _accumulate_contribution_statistics(
        self, route: V9RouteOutput, contribution: V9Contribution
    ) -> None:
        with torch.no_grad():
            accumulators = self._accumulators(route.expert_ids.device)
            ids = self._slot_index(route.expert_ids)[route.slot_mask]
            raw = contribution.raw.detach()[route.slot_mask]
            positive = contribution.positive.detach()[route.slot_mask]
            accumulators["contribution_sum"].index_add_(0, ids, raw)
            accumulators["positive_contribution"].index_add_(0, ids, positive)
            accumulators["positive_count"].index_add_(
                0, ids, (positive > 0).to(positive.dtype)
            )
            # The target itself, not just its sign: this is what ``L_key``
            # pushes the gate towards, so its mean is the quantity that says
            # whether the responsibility is concentrating or flattening out.
            accumulators["responsibility_sum"].index_add_(
                0,
                ids,
                contribution.responsibility.detach()[route.slot_mask],
            )

    def _usage_snapshot(self) -> Dict[str, Dict[str, float]]:
        """One device->host transfer per accumulator, at boundaries only."""
        if self._v9_accumulators is None:
            return {}
        snapshot = {
            name: values.detach().cpu().tolist()
            for name, values in self._v9_accumulators.items()
        }
        payload = {}
        for index, expert_id in enumerate(self.v9_order):
            usage = int(snapshot["usage"][index])
            payload[str(expert_id)] = {
                "usage": usage,
                "selected": int(snapshot["selected"][index]),
                "support": float(snapshot["support"][index]),
                "contribution_sum": float(snapshot["contribution_sum"][index]),
                "positive_contribution": float(
                    snapshot["positive_contribution"][index]
                ),
                "positive_count": int(snapshot["positive_count"][index]),
                "responsibility_sum": float(snapshot["responsibility_sum"][index]),
            }
        return payload

    # ------------------------------------------------------------------
    # objective
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False):
        from transformers import Trainer

        queries = inputs.pop("fixed_queries")
        inputs.pop("sample_ids", None)
        inputs.pop("historical_topc", None)
        inputs["v7_sum_per_sample_loss"] = True
        result = Trainer.compute_loss(self, model, inputs, return_outputs=return_outputs)
        # One pass carries every offered expert: the selection is applied inside
        # the model's forward, not by looping over experts around it.
        self.v9_backbone_forwards += 1
        answer_loss, outputs = result if return_outputs else (result, None)
        if self._v9_active is None:
            raise RuntimeError("V9 routing must run before compute_loss")
        _, route, _stage_state = self._v9_active

        # The model returns a per-sample *sum* once the micro-batch exceeds one;
        # every V9 term is a per-micro-batch mean, so normalise once here.  A
        # summed answer term against a meaned key term would scale the effective
        # key weight by the micro-batch width.
        answer_loss = answer_loss / queries.shape[0]

        with self.profiler.timed("contribution_time"):
            # The derivative is taken against the gate the answer loss is a
            # function of -- the one the composition consumed.  Taking it
            # against the key-differentiable copy would be taking it against a
            # tensor the answer never saw, which is identically zero.
            contribution_raw = local_conditional_contribution(
                answer_loss,
                route.forward_gates,
                retain_graph=True,
                values=route.forward_gates,
            )
            contribution = answer_derived_responsibility(
                contribution_raw,
                valid=route.slot_mask,
                epsilon=self.v9_config.loss.responsibility_epsilon,
            )
        self._accumulate_contribution_statistics(route, contribution)
        with torch.no_grad():
            total = contribution_raw.detach().abs().sum()
            if self._gate_grad_abs_sum is None:
                self._gate_grad_abs_sum = total
            else:
                self._gate_grad_abs_sum = self._gate_grad_abs_sum + total
            self._gate_grad_count += int(contribution_raw.numel())

        if not self._v9_checked_answer_key_isolation:
            # The method's central claim, checked against the graph that is
            # about to be differentiated -- not against a reimplementation of it
            # in a unit test.  ``L_ans`` must have zero derivative w.r.t. every
            # key parameter; the only answer-side route into a key is
            # contribution -> responsibility -> L_key, which is a *teacher*
            # (detached) and therefore does not appear here at all.
            self._assert_answer_loss_isolated_from_keys(answer_loss, route)
            self._v9_checked_answer_key_isolation = True

        terms = compose_total_loss(
            answer_loss=answer_loss,
            # ``L_key`` must reach the keys, so it is the differentiable copy of
            # the gate that enters the objective, not the parameter-free one the
            # composition consumed.  The two carry the same value; only one of
            # them can carry a gradient home.
            probabilities=route.probabilities,
            responsibility=contribution.responsibility,
            valid_rows=contribution.valid,
            slot_mask=route.slot_mask,
            config=self.v9_config.loss,
        )
        if not bool(route.slot_mask.any().item()):
            # Nothing trainable was routed to.  The zero-valued anchor keeps the
            # backward pass well-formed and keeps every trainable parameter in
            # the graph, so DDP never sees one as unused because of this batch.
            terms.total = terms.total + 0.0 * sum(
                parameter.sum()
                for group in self.optimizer.param_groups
                for parameter in group["params"]
            )
            self.v9_noop_steps += 1
        self._v9_losses = (terms, contribution, _stage_state, route)
        return (terms.total, outputs) if return_outputs else terms.total

    # ------------------------------------------------------------------
    # logging (spec §38: aggregate, at an interval, never per-sample)
    # ------------------------------------------------------------------
    def _write_step_metrics(self) -> None:
        if self._v9_losses is None:
            return
        terms, contribution, stage_state, route = self._v9_losses
        statistics = contribution_statistics(
            contribution.raw, contribution.responsibility, route.slot_mask
        )
        # Asked once per interval rather than per expert: ``_keys_per_expert``
        # walks the pool, and the metrics path must stay off the critical path.
        keys_per_expert = self._keys_per_expert()
        gate_grad_abs_mean = 0.0
        if self._gate_grad_abs_sum is not None and self._gate_grad_count:
            gate_grad_abs_mean = float(
                (self._gate_grad_abs_sum / self._gate_grad_count).item()
            )
        payload = {
            "step": int(self.state.global_step),
            "task_index": self.v9_task_index,
            "stage": stage_state.stage,
            "stage_progress": round(stage_state.stage_progress, 6),
            "temperature": round(stage_state.temperature, 6),
            "loss_answer": float(terms.answer.detach().item()),
            "loss_key": float(terms.key.detach().item()),
            "loss_sparse": float(terms.sparse.detach().item()),
            "loss_budget": float(terms.budget.detach().item()),
            "loss_total": float(terms.total.detach().item()),
            "mean_active_experts": float(terms.active_mass.mean().item()),
            "responsibility_valid_rate": round(contribution.valid_rate, 6),
            "responsibility_row_sum": round(
                float(contribution.responsibility.sum(dim=1).mean().item()), 6
            ),
            "contribution_positive_rate": round(statistics["positive_rate"], 6),
            "contribution_mean": round(statistics["mean"], 6),
            "gate_grad_abs_mean": gate_grad_abs_mean,
            "responsibility_mean": round(statistics["responsibility_mean"], 6),
            "responsibility_max": round(statistics["responsibility_max"], 6),
            "keys_per_expert": keys_per_expert,
            "trainable_key_max_cosine": self._trainable_key_redundancy(),
            "peak_memory_bytes": self._peak_memory_bytes(),
            "backbone_forwards": int(self.v9_backbone_forwards),
            # §41 (D) is a claim about the *micro-step*, not about the sample:
            # every offered expert rides on the one forward, so this is 1.0 and
            # stays 1.0 as the number of candidates grows.  The per-sample figure
            # is below it purely because a micro-batch shares the pass, which is
            # why both are reported rather than the more flattering one.
            "backbone_forwards_per_micro_step": (
                self.v9_backbone_forwards / max(self.v9_micro_steps, 1)
            ),
            "backbone_forwards_per_sample": (
                self.v9_backbone_forwards / max(self.v9_observed_sample_count, 1)
            ),
            # The same ratio taken from the model's own forward hook rather than
            # from this trainer's bookkeeping, so the two lines are independent
            # measurements of one claim and a disagreement means one of them is
            # wrong.  ``calibration_model_forwards`` is the bounded held-out
            # measurement and is deliberately outside the ratio above.
            "model_forwards": int(self.v9_model_forwards),
            "train_model_forwards": int(self.v9_train_model_forwards),
            "calibration_model_forwards": int(self.v9_calibration_model_forwards),
            "model_forwards_per_micro_step": (
                self.v9_train_model_forwards / max(self.v9_micro_steps, 1)
            ),
            # Wide retrieval widens the routing row; it must widen the LoRA
            # branch set and nothing else.  Reported per wide micro-step so the
            # claim is checked on the steps that actually exercise it, rather
            # than diluted by the 95% of steps where the wide path is dormant.
            "wide_micro_steps": int(self.v9_wide_micro_steps),
            "wide_model_forwards_per_micro_step": (
                self.v9_wide_model_forwards / max(self.v9_wide_micro_steps, 1)
            ),
            "extra_forwards": int(self.v9_extra_forwards),
            "per_expert": self._usage_snapshot(),
        }
        if self._v9_step_timing is not None:
            payload.update(self._v9_step_timing)
            # Throughput of the micro-step just finished, this rank alone.  The
            # world size multiplies it into the run's real rate, which is why the
            # local batch size travels beside it rather than being folded in.
            elapsed = float(self._v9_step_timing.get("training_step_sec") or 0.0)
            payload["samples_per_second"] = (
                float(self._v9_step_timing.get("local_batch_size", 0)) / elapsed
                if elapsed > 0.0
                else 0.0
            )
        with self.profiler.timed("metrics_time"):
            self.v9_logger.write(payload)
        self.profiler.bump("optimizer_steps_logged")

    def _keys_per_expert(self) -> Dict[str, int]:
        """How many routing keys each expert currently carries (spec §34).

        A historical expert that earned a key this task carries two -- a frozen
        base and this task's own -- while a candidate carries one.  The count is
        the observable that says whether the multi-key pool is growing the way
        the method intends or is quietly accumulating aliases.
        """
        records = self.v9_key_pool.key_records
        counts: Dict[str, int] = {}
        for expert_id in self.v9_order:
            counts[str(int(expert_id))] = sum(
                1
                for key_id in self.v9_key_pool.key_ids(expert_id=int(expert_id))
                if records[key_id]["lifecycle"] != LIFECYCLE_PRUNED
            )
        return counts

    def _trainable_key_redundancy(self) -> float:
        """Largest cosine between two keys this task may still move (spec §34).

        Two candidates whose keys have collapsed onto each other cannot be told
        apart at inference, and the failure is silent: the gates look healthy
        while the pool holds one expert twice.  Reported every logging interval
        because it is the cheapest early warning the method has.
        """
        trainable = [
            key_id
            for key_id in self.v9_key_pool.trainable_key_ids()
            if key_id in self.v9_key_pool.keys
        ]
        if len(trainable) < 2:
            return 0.0
        matrix = F.normalize(
            torch.stack([self.v9_key_pool.keys[key_id].detach() for key_id in trainable])
            .float()
            .cpu(),
            dim=-1,
        )
        cosine = matrix @ matrix.T
        cosine.fill_diagonal_(-2.0)
        return float(cosine.max().item())

    @staticmethod
    def _peak_memory_bytes() -> int:
        """Peak allocator usage on this rank; 0 on a CPU-only run."""
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.max_memory_allocated())

    # ------------------------------------------------------------------
    # task-end aggregation (spec §23)
    # ------------------------------------------------------------------
    def _local_usage_counters(self) -> Dict[str, Any]:
        return {
            "rank": torch.distributed.get_rank() if _distributed() else 0,
            "per_expert": self._usage_snapshot(),
            "micro_steps": int(self.v9_micro_steps),
            "noop_micro_steps": int(self.v9_noop_steps),
            # Serialised with ``micro_steps`` because the two are only
            # meaningful as a ratio: restoring the denominator without the
            # numerator made a resumed task report fewer traversals than
            # micro-steps and fail §41 (D) for having been restarted.
            "backbone_forwards": int(self.v9_backbone_forwards),
            # Always zero in a healthy run: a non-zero value would have raised
            # inside ``_at_sync_boundary`` before reaching a checkpoint.
            "unsynced_optimizer_steps": int(self.v9_unsynced_optimizer_steps),
            "observed_sample_count": int(self.v9_observed_sample_count),
            "unique_sample_ids": sorted(self.v9_unique_sample_ids),
            "stage_steps": dict(self.v9_stage_steps),
            "order": list(self.v9_order),
            # §41 (D) evidence.  These survive a resume for the same reason the
            # usage counters do: a task that is interrupted and resumed must
            # still report one traversal per micro-step over the micro-steps it
            # actually took, not over the ones since the last checkpoint.
            "model_forwards": int(self.v9_model_forwards),
            "train_model_forwards": int(self.v9_train_model_forwards),
            "calibration_model_forwards": int(self.v9_calibration_model_forwards),
            "wide_micro_steps": int(self.v9_wide_micro_steps),
            "wide_model_forwards": int(self.v9_wide_model_forwards),
            "wide_steps": int(self.v9_wide_steps),
            "extra_forwards": int(self.v9_extra_forwards),
        }

    def global_task_statistics(self) -> Dict[str, Any]:
        """All-reduce the task statistics and return the global view.

        Every rank computes the same numbers from the same gathered rows, so the
        prune/retain decision taken from them is a pure function of the data and
        cannot diverge between ranks (spec §23).
        """
        local = self._local_usage_counters()
        gathered = [local]
        if _distributed():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local)
        micro_steps = max(sum(int(row["micro_steps"]) for row in gathered), 1)
        observed_samples = max(sum(int(row["observed_sample_count"]) for row in gathered), 1)
        order = sorted(
            {int(expert_id) for row in gathered for expert_id in row.get("order", ())}
        ) or list(self.v9_order)
        fields = (
            "usage",
            "selected",
            "support",
            "contribution_sum",
            "positive_contribution",
            "positive_count",
            "responsibility_sum",
        )
        per_expert = {}
        for expert_id in order:
            key = str(expert_id)
            sums = {field: 0.0 for field in fields}
            for row in gathered:
                entry = row.get("per_expert", {}).get(key)
                if entry is None:
                    continue
                for field in fields:
                    sums[field] += float(entry[field])
            usage = int(sums["usage"])
            per_expert[key] = {
                "offered_count": usage,
                "usage": usage,
                # Usage is accumulated per sample x offered slot.  Dividing by
                # micro-steps changes its meaning with batch size/world size.
                "offered_rate": usage / observed_samples,
                "usage_rate": usage / observed_samples,
                "selected": int(sums["selected"]),
                "selected_rate": (sums["selected"] / usage) if usage else 0.0,
                "effective_support": sums["support"],
                "mean_contribution": (sums["contribution_sum"] / usage) if usage else 0.0,
                "mean_positive_contribution": (
                    (sums["positive_contribution"] / usage) if usage else 0.0
                ),
                "positive_contribution_rate": (
                    (sums["positive_count"] / usage) if usage else 0.0
                ),
                "mean_gate": (sums["support"] / usage) if usage else 0.0,
                "mean_responsibility": (
                    (sums["responsibility_sum"] / usage) if usage else 0.0
                ),
            }
        stage_steps: Dict[str, int] = {}
        for row in gathered:
            for stage, count in row.get("stage_steps", {}).items():
                stage_steps[stage] = stage_steps.get(stage, 0) + int(count)
        covered = {
            sample_id for row in gathered for sample_id in row.get("unique_sample_ids", ())
        }
        keys_per_expert = self._keys_per_expert()
        return {
            "world_size": len(gathered),
            "micro_steps": micro_steps,
            "stage_steps": stage_steps,
            "noop_micro_steps": sum(int(row["noop_micro_steps"]) for row in gathered),
            "observed_sample_count": observed_samples,
            "unique_sample_count": len(covered),
            "per_expert": per_expert,
            # Spec §34: an expert's usage is only interpretable against how many
            # keys it holds, so the two are reported together rather than left
            # for whoever reads the log to join by hand.
            "keys_per_expert": keys_per_expert,
            "expert_usage_vs_key_count": [
                {
                    "expert_id": int(expert_id),
                    "keys": keys_per_expert.get(str(expert_id), 0),
                    "usage_rate": per_expert.get(str(expert_id), {}).get(
                        "usage_rate", 0.0
                    ),
                    "historical": int(expert_id) in self.v9_key_pool.historical_ids,
                }
                for expert_id in order
            ],
            "ranks": gathered,
        }

    # ------------------------------------------------------------------
    # distributed integrity
    # ------------------------------------------------------------------
    def distributed_barrier(self) -> None:
        if _distributed():
            torch.distributed.barrier()

    def expert_pool_metadata_hash(self) -> str:
        """Stable digest of everything the ranks must agree on (spec §23)."""
        digest = hashlib.sha256()
        for expert_id in self.v9_key_pool.expert_ids():
            digest.update(str(int(expert_id)).encode("utf-8"))
        for key_id in self.v9_key_pool.key_ids():
            digest.update(key_id.encode("utf-8"))
            digest.update(tensor_checksum(self.v9_key_pool.keys[key_id]).encode("utf-8"))
        for layer_name in sorted(self.expert_pool.manager.layers):
            layer = self.expert_pool.manager.layers[layer_name]
            digest.update(layer_name.encode("utf-8"))
            for expert_id in sorted(self.v9_key_pool.current_ids):
                expert = layer.experts[str(int(expert_id))]
                for name, value in sorted(expert.state_dict().items()):
                    digest.update(name.encode("utf-8"))
                    digest.update(tensor_checksum(value).encode("utf-8"))
        return digest.hexdigest()

    def distributed_state_audit(self, stage: str) -> Dict[str, Any]:
        local = {
            "rank": torch.distributed.get_rank() if _distributed() else 0,
            "key_checksums": {
                key_id: tensor_checksum(self.v9_key_pool.keys[key_id])
                for key_id in self.v9_key_pool.key_ids()
            },
            "candidate_lora_checksums": adapter_checksums(
                self.expert_pool.manager, self.v9_key_pool.current_ids
            ),
            "router_order": list(self.v9_router._order),
            "bias": self.v9_router.bias_state(),
            "pool_metadata_hash": self.expert_pool_metadata_hash(),
        }
        gathered = [local]
        if _distributed():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local)
        comparable = [
            {key: value for key, value in row.items() if key != "rank"}
            for row in gathered
        ]
        if any(row != comparable[0] for row in comparable[1:]):
            offending = sorted(
                key
                for key in comparable[0]
                if any(row[key] != comparable[0][key] for row in comparable[1:])
            )
            raise AssertionError(
                "V9 distributed state differs across ranks at {}: {}".format(
                    stage, offending
                )
            )
        return {"stage": stage, "world_size": len(gathered), "ranks": gathered}

    # ------------------------------------------------------------------
    # freeze integrity
    # ------------------------------------------------------------------
    def assert_task_freeze_integrity(self) -> Dict[str, Any]:
        after_keys = self.v9_key_pool.historical_checksums()
        if after_keys != self._historical_key_before:
            changed = sorted(
                key_id
                for key_id in set(after_keys) | set(self._historical_key_before)
                if after_keys.get(key_id) != self._historical_key_before.get(key_id)
            )
            raise AssertionError(
                "a Base Key or an earlier task's key changed during this task: "
                "{}".format(changed[:10])
            )
        after_lora = adapter_checksums(
            self.expert_pool.manager, self.v9_key_pool.historical_ids
        )
        if after_lora != self._historical_lora_before:
            raise AssertionError("historical LoRA checksum changed during this task")
        after_rms = runtime_kappa_calibration(self.model, self.v9_key_pool.historical_ids)
        if after_rms != self._historical_rms_before:
            raise AssertionError("historical RMS calibration changed during this task")
        return {
            "historical_key_unchanged": True,
            "historical_lora_unchanged": True,
            "historical_rms_unchanged": True,
        }

    def full_data_coverage_audit(self, num_train_samples: int) -> Dict[str, Any]:
        """Did every training sample actually reach the loss (spec §34)?

        V9 must consume the full task data: a schedule that quietly drops
        samples would change what the answer supervises without changing any
        visible hyperparameter.
        """
        local = {
            "rank": torch.distributed.get_rank() if _distributed() else 0,
            "unique_sample_ids": sorted(self.v9_unique_sample_ids),
            "observed_rows": int(self.v9_observed_sample_count),
            "micro_steps": int(self.v9_micro_steps),
        }
        gathered = [local]
        if _distributed():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local)
        unique_ids = sorted(
            {value for row in gathered for value in row["unique_sample_ids"]}
        )
        observed = sum(int(row["observed_rows"]) for row in gathered)
        audit = full_data_coverage_audit(
            num_train_samples=int(num_train_samples),
            unique_sample_ids=unique_ids,
            optimizer_micro_steps=max(int(row["micro_steps"]) for row in gathered),
            optimizer_steps=int(self.state.global_step),
            observed_sample_count=observed,
            require_full=bool(self.v9_require_full_coverage),
        )
        audit.update(
            {
                "world_size": len(gathered),
                "rank_unique_sample_ids_seen": {
                    str(row["rank"]): len(row["unique_sample_ids"]) for row in gathered
                },
                "rank_observed_rows": {
                    str(row["rank"]): row["observed_rows"] for row in gathered
                },
                "distributed_padding_count": max(0, observed - int(num_train_samples)),
            }
        )
        return audit

    def trainable_parameter_audit(self, print_census: bool = True) -> Dict[str, Any]:
        """Full parameter census (spec §28), printed once per task."""
        rows: List[Dict[str, Any]] = []
        totals: Dict[str, int] = {}
        for name, parameter in self.model.named_parameters():
            trainable = bool(parameter.requires_grad)
            rows.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "requires_grad": trainable,
                    "numel": int(parameter.numel()),
                }
            )
            totals["total_parameters"] = totals.get("total_parameters", 0) + int(
                parameter.numel()
            )
            if trainable:
                totals["trainable_parameters"] = totals.get(
                    "trainable_parameters", 0
                ) + int(parameter.numel())
                if ".experts." in name:
                    totals["candidate_lora_parameters"] = totals.get(
                        "candidate_lora_parameters", 0
                    ) + int(parameter.numel())
                elif "v9_router" in name:
                    totals["router_parameters"] = totals.get(
                        "router_parameters", 0
                    ) + int(parameter.numel())
        # Fail fast: a frozen module that somehow requires grad would silently
        # train the backbone.
        forbidden = [
            row["name"]
            for row in rows
            if row["requires_grad"]
            and ".experts." not in row["name"]
            and "v9_router" not in row["name"]
        ]
        if forbidden:
            raise AssertionError(
                "parameters outside the candidate LoRA / router are trainable: "
                "{}".format(forbidden[:10])
            )
        candidate_keys = [
            self.v9_key_pool.origin_key_id(expert_id)
            for expert_id in self.v9_key_pool.current_ids
        ]
        trainable_keys = self.v9_key_pool.trainable_key_ids()
        #: Historical experts' *current-task* keys: independent, absolute, and
        #: the only part of a historical expert that trains.
        task_keys = [key for key in trainable_keys if key not in set(candidate_keys)]
        totals["candidate_key_parameters"] = sum(
            int(self.v9_key_pool.keys[key_id].numel())
            for key_id in candidate_keys
            if key_id in self.v9_key_pool.key_records
        )
        totals["historical_task_key_parameters"] = sum(
            int(self.v9_key_pool.keys[key_id].numel()) for key_id in task_keys
        )
        payload = {
            "parameters": rows,
            "totals": totals,
            "trainable_key_ids": list(trainable_keys),
            "candidate_key_ids": candidate_keys,
            "historical_task_key_ids": task_keys,
        }
        if print_census and (not _distributed() or torch.distributed.get_rank() == 0):
            for row in rows:
                print(
                    "V9 param {name:<90} {shape!s:<20} requires_grad={requires_grad}".format(
                        name=row["name"],
                        shape=tuple(row["shape"]),
                        requires_grad=row["requires_grad"],
                    )
                )
            print("V9 parameter census: {}".format(totals))
        return payload

    # ------------------------------------------------------------------
    # exact-contribution calibration (spec §30)
    # ------------------------------------------------------------------
    def calibration_pass(
        self,
        loader,
        sample_budget: int,
        top_k: int = 2,
    ) -> Optional[Dict[str, float]]:
        """Compare ``G_grad`` against the exact removal effect, once.

        This is the only place in the codebase that runs the exact oracle.  It
        runs on at most ``sample_budget`` held-out samples, it is a diagnostic,
        and no gradient from it ever reaches the optimizer.  If the two
        quantities come out uncorrelated, the gate-gradient proxy is not
        measuring what the method assumes and the run should stop rather than
        train for days on a signal that does not mean anything.
        """
        if not self.v9_config.validation.contribution_calibration:
            return None
        # The calibration is a single-process diagnostic: it runs on one rank
        # and its forwards must not enter DDP's forward/backward bookkeeping,
        # which would leave the reducer waiting for a backward that never comes.
        model = getattr(self.model, "module", self.model)
        self._attach_backbone_counter(model)
        # Everything forwarded inside this method is held-out *measurement*, not
        # the one pass a training sample rides on.  Attributing it to its own
        # scope is what keeps §41 (D) checkable: the training invariant is
        # "one traversal per micro-step", and folding the calibration's
        # legitimate extra forwards into that count would make the invariant
        # unfalsifiable exactly where it matters most -- on the runs that use
        # the exact oracle.
        self._v9_forward_scope = "calibration"
        device = next(self.v9_key_pool.parameters()).device
        was_training = model.training
        grad_values: List[torch.Tensor] = []
        exact_values: List[torch.Tensor] = []
        gain_sum: Dict[int, float] = {}
        gain_count: Dict[int, int] = {}
        pair_deployed: List[torch.Tensor] = []
        pair_alternatives: List[torch.Tensor] = []
        pair_is_deployed: List[torch.Tensor] = []
        consumed = 0
        try:
            for inputs in loader:
                if consumed >= int(sample_budget):
                    break
                with torch.no_grad():
                    prepared = self._prepare_inputs(inputs)
                queries = prepared.pop("fixed_queries").to(device)
                prepared.pop("sample_ids", None)
                historical_rows = prepared.pop("historical_topc").to(device)
                prepared["v7_sum_per_sample_loss"] = True
                route, local, base = self._calibration_gradients(
                    model, queries, historical_rows, prepared
                )
                exact = self._exact_removal_matrix(model, prepared, route, base)
                mask = route.slot_mask
                grad_values.append(local[mask].detach().cpu())
                exact_values.append(exact[mask].detach().cpu())
                self._accumulate_validation_gain(
                    route.expert_ids[mask], exact[mask], gain_sum, gain_count
                )
                scores = self._validation_score_gap(
                    model, route, prepared, self.v9_validation_scores
                )
                self.v9_validation_scores = scores
                if self.v9_config.inference.pair_rerank:
                    deployed, alternatives, flag = self._exact_pair_losses(
                        model, prepared, route
                    )
                    pair_deployed.append(deployed.detach().cpu())
                    pair_alternatives.append(alternatives.detach().cpu())
                    pair_is_deployed.append(flag.detach().cpu())
                consumed += int(base.shape[0])
                del route, local, base, exact, prepared
        finally:
            self._v9_forward_scope = None
            if was_training:
                model.train()
        if not grad_values:
            return None
        self.v9_calibration = calibration_report(
            torch.cat(grad_values), torch.cat(exact_values), top_k=top_k
        )
        self.v9_calibration["samples"] = int(consumed)
        if pair_deployed:
            deployed = torch.cat(pair_deployed)
            # "Which challenger was best" is the argmin over the columns, and
            # ``pair_is_deployed`` says per sample whether the best one was the
            # pair the gate itself serves.
            self.v9_calibration["pair_rerank"] = pair_rerank_report(
                deployed,
                torch.cat(pair_alternatives),
                torch.cat(pair_is_deployed),
            )
        #: Mean exact answer-loss reduction each expert buys on held-out data,
        #: under the deployed routing rule.  Consumed by the candidate commit
        #: audit (spec §19); it is empty when the calibration did not run, which
        #: the audit reports rather than guessing.
        self.v9_validation_gain = {
            int(expert_id): gain_sum[int(expert_id)] / max(gain_count[int(expert_id)], 1)
            for expert_id in gain_sum
        }
        if self.v9_validation_scores is not None:
            # Same artifact as the calibration: both answer "does the training
            # objective predict what deployment will score", and reading them
            # apart invites exactly the confusion they exist to prevent.
            self.v9_calibration["validation_scores"] = dict(self.v9_validation_scores)
        return self.v9_calibration

    def _validation_score_gap(
        self,
        model,
        route: V9RouteOutput,
        prepared: Dict[str, Any],
        running: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """Soft-gate vs deployed-Top-2 answer NLL on held-out data (spec §34).

        Both scores are measured on the same batch, in the same mode, against
        the same cached expert outputs: the only thing that changes between them
        is the gate tensor, so the difference is the price of serving the hard
        rule instead of the mixture the objective trains.  Accumulated as a
        running mean so the number does not depend on the size of the last
        calibration batch.

        ``gap = hard - soft``, so it is the price *paid*, not the headroom
        available: positive means deployment scores worse than the objective the
        run minimised, and it is that direction the report needs to be readable
        in.
        """
        # "soft" here means the sigmoid mixture the objective optimises, as
        # against the Top-2 indicator deployment serves -- a different
        # distinction from the one ``answer_gates`` draws.
        mixture_gates = route.probabilities.detach()
        hard_gates = self.v9_router.deployed_gates(route.probabilities, route.slot_mask)
        scores: Dict[str, float] = {}
        was_training = model.training
        model.eval()
        try:
            for name, gates in (("soft", mixture_gates), ("hard_top2", hard_gates)):
                selection = ComposeSelection(
                    expert_ids=route.expert_ids,
                    gates=gates,
                    max_slots=int(route.expert_ids.shape[1]),
                    allow_zero_gates=True,
                )
                with torch.no_grad(), self.expert_pool.manager.selection_context(
                    selection
                ):
                    model(**prepared)
                    per_sample = getattr(model, "v7_per_sample_answer_nll", None)
                    if per_sample is None:
                        raise RuntimeError(
                            "the soft/hard validation gap requires the per-sample "
                            "answer NLL"
                        )
                    scores[name] = float(per_sample.mean().item())
                # Counted as an extra forward: it is held-out *measurement*, not
                # part of the one pass a training sample rides on, and folding it
                # into the backbone-forward count would understate §41 (D).
                self.v9_extra_forwards += 1
        finally:
            if was_training:
                model.train()
        scores["gap"] = scores["hard_top2"] - scores["soft"]
        if running:
            previous = float(running.get("batches", 0.0))
            for name in ("soft", "hard_top2", "gap"):
                scores[name] = (
                    float(running.get(name, 0.0)) * previous + scores[name]
                ) / (previous + 1.0)
            scores["batches"] = previous + 1.0
        else:
            scores["batches"] = 1.0
        return scores

    @staticmethod
    def _accumulate_validation_gain(
        slot_expert_ids: torch.Tensor,
        slot_gains: torch.Tensor,
        gain_sum: Dict[int, float],
        gain_count: Dict[int, int],
    ) -> None:
        """Fold one batch's per-slot removal deltas into per-expert means."""
        ids = slot_expert_ids.detach().cpu().reshape(-1).tolist()
        gains = slot_gains.detach().cpu().reshape(-1).tolist()
        for expert_id, gain in zip(ids, gains):
            expert_id = int(expert_id)
            gain_sum[expert_id] = gain_sum.get(expert_id, 0.0) + float(gain)
            gain_count[expert_id] = gain_count.get(expert_id, 0) + 1

    def _calibration_gradients(self, model, queries, historical_rows, prepared):
        """One grad-enabled answer forward; return the route, ``G_grad``, NLL.

        The graph is created inside this call and dropped when it returns, so at
        most one calibration batch is ever resident at a time.

        The derivative is taken against the batch's *sum* of per-sample losses,
        not their mean.  Either answers §30's ranking questions identically -- a
        uniform factor cancels out of every correlation, sign and top-k test --
        but the report also prints ``grad_mean`` beside ``exact_mean``, and a
        ``1/batch`` factor there would make the proxy look weaker than it is by
        exactly the batch width.  The quantity being calibrated is one sample's
        removal effect, so the estimate has to be one sample's derivative.

        The backward runs *inside* the selection context.  Activation
        checkpointing recomputes each block's forward during ``backward()``, and
        a recomputation that cannot see the routing rebuilds a different graph
        than the forward did -- PyTorch then reports the two having saved a
        different number of tensors and the calibration dies on its first batch.
        The training loop already holds the context across forward *and*
        backward for exactly this reason (``_ddp_training_step``); this is the
        same contract, not a second one.
        """
        with torch.enable_grad():
            route = self.v9_router.route(
                queries,
                historical_rows,
                float(self.v9_config.routing.temperature_start),
                "soft",
            )
            with self.expert_pool.manager.selection_context(route.selection()):
                model.train()
                model(**prepared)
                per_sample = getattr(model, "v7_per_sample_answer_nll", None)
                if per_sample is None:
                    raise RuntimeError(
                        "calibration requires the per-sample routed answer NLL"
                    )
                gradient = gate_gradient(
                    per_sample.sum(), route.answer_gates, retain_graph=False
                )
            local = -(route.answer_gates.detach() * gradient.detach())
            return route, local.detach(), per_sample.detach().clone()

    def _exact_removal_matrix(
        self,
        model,
        prepared: Dict[str, Any],
        route: V9RouteOutput,
        base_losses: torch.Tensor,
    ) -> torch.Tensor:
        """``G_exact(k) = L(S \\ E_k) - L(S)`` per sample and candidate slot.

        One backbone forward per distinct expert, which is exactly why this is
        bounded to a handful of held-out samples and is unreachable from the
        training path.

        Two details the composition insists on.  The removal is expressed by
        replacing the expert's ids with ``PAD_EXPERT_ID``, and a padded slot must
        carry a zero gate -- the selection validates that, so the gate is zeroed
        with the id.  And the answer loss is per *sample* while the matrix is per
        *slot*: the delta is broadcast down the slot axis explicitly rather than
        left to ``torch.where``, which would align it against the wrong axis (and
        only complain when the micro-batch width happens to differ from the slot
        count).
        """
        expert_ids = route.expert_ids
        gate_template = route.forward_gates
        distinct = sorted({int(value) for value in expert_ids[route.slot_mask].tolist()})
        # On the routing device: the accumulator is compared against ``delta``
        # by ``torch.where``, and a shape-only allocation defaults to CPU, where
        # it agrees with a CPU fixture and with nothing else.  This is the
        # post-training calibration -- the stage that runs after the training
        # loop has already committed its checkpoints -- so the failure it used
        # to produce was an abort with no report at all.
        matrix = torch.zeros(
            expert_ids.shape, dtype=torch.float32, device=expert_ids.device
        )
        for expert_id in distinct:
            removed_slot = expert_ids.eq(expert_id)
            kept = torch.where(
                removed_slot, torch.full_like(expert_ids, PAD_EXPERT_ID), expert_ids
            )
            kept_gates = torch.where(
                removed_slot, torch.zeros_like(gate_template), gate_template
            )
            alternative = ComposeSelection(
                expert_ids=kept,
                gates=kept_gates,
                max_slots=int(kept.shape[1]),
                allow_zero_gates=True,
            )
            with torch.no_grad(), self.expert_pool.manager.selection_context(alternative):
                model(**prepared)
                removed = getattr(model, "v7_per_sample_answer_nll", None)
                if removed is None:
                    raise RuntimeError(
                        "exact removal requires the per-sample answer NLL"
                    )
                delta = exact_removal_contribution(base_losses, removed.detach())
            self.v9_extra_forwards += 1
            matrix = torch.where(removed_slot, delta.reshape(-1, 1), matrix)
        return matrix

    def _exact_pair_losses(
        self,
        model,
        prepared: Dict[str, Any],
        route: V9RouteOutput,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """Exact answer NLL of the deployed pair versus its challengers (§35).

        Returns ``(column_0, all_columns, is_deployed_flag)``.  Column 0 is the
        pair with the largest summed gate mass among the candidates, which is the
        pair the gate ranking deploys when it has to name one; the remaining
        columns are the next-best pairs by the same score.  One backbone forward
        per pair, on the calibration sample only; the main loop never reaches
        this function.

        ``is_deployed_flag`` marks, per sample, whether column 0 *is* the pair
        that sample's deployed rule selects.  Without it the report would be
        comparing every sample against a pair it may never have served, and
        ``best_pair_is_deployed_rate`` would be a statement about column 0 rather
        than about the gate.  The probe is a global (per-batch, per-task) set, so
        a sample whose deployed pair contains a recalled historical expert is
        flagged 0 -- honestly, since that pair is not among the challengers.
        """
        pairs = self._pair_probe(route)
        rows = int(route.expert_ids.shape[0])
        alternatives = torch.zeros((rows, len(pairs)), dtype=torch.float32)
        deployed_flag = torch.zeros(rows, dtype=torch.float32)
        deployed_gates = self.v9_router.deployed_gates(
            route.probabilities.detach(), route.slot_mask
        )
        for column, pair in enumerate(pairs):
            wanted = torch.zeros_like(route.expert_ids, dtype=torch.bool)
            for expert_id in pair:
                wanted |= route.expert_ids.eq(int(expert_id))
            wanted &= route.slot_mask
            kept = torch.where(
                wanted, route.expert_ids, torch.full_like(route.expert_ids, PAD_EXPERT_ID)
            )
            kept_gates = torch.where(
                wanted, route.forward_gates, torch.zeros_like(route.forward_gates)
            )
            alternative = ComposeSelection(
                expert_ids=kept,
                gates=kept_gates,
                max_slots=int(kept.shape[1]),
                allow_zero_gates=True,
            )
            with torch.no_grad(), self.expert_pool.manager.selection_context(alternative):
                model(**prepared)
                losses = getattr(model, "v7_per_sample_answer_nll", None)
                if losses is None:
                    raise RuntimeError("pair rerank requires the per-sample answer NLL")
            self.v9_extra_forwards += 1
            alternatives[:, column] = losses.detach().float().cpu()
            if column == 0:
                chosen = torch.where(
                    deployed_gates.detach().gt(0),
                    route.expert_ids,
                    torch.full_like(route.expert_ids, PAD_EXPERT_ID),
                )
                target = set(int(value) for value in pair)
                for row in range(rows):
                    served = {
                        int(value)
                        for value in chosen[row].tolist()
                        if int(value) != PAD_EXPERT_ID
                    }
                    deployed_flag[row] = 1.0 if served == target else 0.0
        return alternatives[:, 0].clone(), alternatives, deployed_flag

    def _pair_probe(self, route: V9RouteOutput) -> "list[tuple[int, int]]":
        """Candidate pairs to measure, ordered by summed gate mass.

        Restricted to the current candidates, and computed once per task from
        the first calibration batch.  Candidates are the only experts every
        routing row is guaranteed to contain; a historical expert appears only
        when it is recalled, so a pair containing one is not measurable across
        the whole sample.  The pair set must also be identical from batch to
        batch, or column ``p`` would mean a different pair per batch.
        """
        if self._v9_pair_probe is not None:
            return self._v9_pair_probe
        candidates = sorted(int(value) for value in self.v9_router.candidate_ids)
        if len(candidates) < 2:
            raise RuntimeError(
                "pair rerank needs at least two current candidates; this task "
                "has {}".format(len(candidates))
            )
        mass = {expert_id: 0.0 for expert_id in candidates}
        ids = route.expert_ids.detach().cpu()
        gates = route.forward_gates.detach().float().cpu()
        mask = route.slot_mask.detach().cpu()
        for row in range(int(ids.shape[0])):
            for slot in range(int(ids.shape[1])):
                if not bool(mask[row, slot]):
                    continue
                expert_id = int(ids[row, slot])
                if expert_id in mass:
                    mass[expert_id] += float(gates[row, slot])
        ranked = sorted(candidates, key=lambda expert_id: (-mass[expert_id], expert_id))
        pairs = [
            (ranked[first], ranked[second])
            for first in range(len(ranked))
            for second in range(first + 1, len(ranked))
        ]
        limit = int(self.v9_config.validation.pair_rerank_pairs)
        self._v9_pair_probe = pairs[: max(limit, 1)]
        return self._v9_pair_probe

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------
    def _save_checkpoint(self, model, trial, metrics=None):
        self.distributed_barrier()
        super()._save_checkpoint(model, trial, metrics)
        self.distributed_barrier()
        counters = [self._local_usage_counters()]
        if _distributed():
            counters = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(counters, self._local_usage_counters())
        checkpoint_dir = os.path.join(
            self._get_output_dir(trial=trial),
            "checkpoint-{}".format(self.state.global_step),
        )
        if self.args.should_save:
            stage_state = self._stage_state(int(self.state.global_step))
            save_v9_checkpoint(
                os.path.join(checkpoint_dir, "v9_state.pt"),
                task_index=self.v9_task_index,
                training_step=self.state.global_step,
                stage=stage_state.stage,
                stage_progress=stage_state.stage_progress,
                key_pool=self.v9_key_pool,
                router_state={
                    "order": list(self.v9_router._order),
                    "candidate_ids": list(self.v9_router.candidate_ids),
                    "historical_ids": list(self.v9_router.historical_ids),
                    "bias": self.v9_router.bias_state(),
                },
                candidate_lora_state=candidate_lora_state(
                    self.expert_pool.manager, self.v9_key_pool.current_ids
                ),
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
                usage_counters={"per_rank": counters},
                config=self.v9_config,
                rms_state={},
            )
        self.distributed_barrier()

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        path = os.path.join(str(resume_from_checkpoint), "v9_state.pt")
        payload, loaded_pool, loaded_config = load_v9_checkpoint(
            path, current_task=self.v9_task_index
        )
        if loaded_config != self.v9_config:
            raise ValueError("V9 resume config mismatch")
        if loaded_pool.expert_ids() != self.v9_key_pool.expert_ids():
            raise ValueError(
                "V9 resume expert registry mismatch: {} != {}".format(
                    loaded_pool.expert_ids(), self.v9_key_pool.expert_ids()
                )
            )
        if loaded_pool.key_ids() != self.v9_key_pool.key_ids():
            raise ValueError(
                "V9 resume key registry mismatch: {} != {}".format(
                    loaded_pool.key_ids(), self.v9_key_pool.key_ids()
                )
            )
        for key_id in loaded_pool.key_ids():
            self.v9_key_pool.keys[key_id].data.copy_(loaded_pool.keys[key_id])
        router_state = payload.get("router_state") or {}
        if list(router_state.get("order", [])) != list(self.v9_router._order):
            raise ValueError("V9 resume router ordering mismatch")
        bias = router_state.get("bias") or {}
        if bias:
            with torch.no_grad():
                self.v9_router.bias.copy_(
                    torch.tensor([float(bias[str(e)]) for e in self.v9_router._order])
                )
        load_candidate_lora_state(
            self.expert_pool.manager, payload["candidate_lora_state"]
        )
        # Trainability follows the *current* task, never the file: only this
        # task's candidates and this task's historical task keys stay trainable.
        self.v9_key_pool.freeze_historical(current_task=self.v9_task_index)
        counters = dict(payload["candidate_usage_counters"])
        if "per_rank" in counters:
            rank = torch.distributed.get_rank() if _distributed() else 0
            per_rank = {int(row["rank"]): row for row in counters["per_rank"]}
            counters = dict(per_rank[rank])
        self._restore_usage_counters(counters)
        optimizer_state = payload.get("optimizer")
        scheduler_state = payload.get("scheduler")
        if self.optimizer is not None and optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        else:
            self._v9_pending_optimizer_state = optimizer_state
        if self.lr_scheduler is not None and scheduler_state is not None:
            self.lr_scheduler.load_state_dict(scheduler_state)
        else:
            self._v9_pending_scheduler_state = scheduler_state

    def _restore_usage_counters(self, counters) -> None:
        self.v9_micro_steps = int(counters.get("micro_steps", 0))
        self.v9_noop_steps = int(counters.get("noop_micro_steps", 0))
        # Resume starts a fresh observation of ``global_step``: the counter is
        # restored, so the *change* it would otherwise see on the first
        # micro-step after loading is not a missed boundary.
        self._v9_global_step_seen = None
        self._v9_last_micro_was_boundary = False
        self.v9_unsynced_optimizer_steps = int(
            counters.get("unsynced_optimizer_steps", 0)
        )
        self.v9_observed_sample_count = int(counters.get("observed_sample_count", 0))
        self.v9_unique_sample_ids = set(counters.get("unique_sample_ids", ()))
        self.v9_stage_steps = dict(counters.get("stage_steps", {}))
        # Restored before the early return below: a resumed run whose per-expert
        # block is empty still has to carry the traversals it already made, or
        # the §41 (D) ratio would be computed over a partial denominator.
        self.v9_backbone_forwards = int(counters.get("backbone_forwards", 0))
        self.v9_model_forwards = int(counters.get("model_forwards", 0))
        self.v9_train_model_forwards = int(counters.get("train_model_forwards", 0))
        self.v9_calibration_model_forwards = int(
            counters.get("calibration_model_forwards", 0)
        )
        self.v9_wide_micro_steps = int(counters.get("wide_micro_steps", 0))
        self.v9_wide_model_forwards = int(counters.get("wide_model_forwards", 0))
        self.v9_wide_steps = int(counters.get("wide_steps", 0))
        self.v9_extra_forwards = int(counters.get("extra_forwards", 0))
        per_expert = counters.get("per_expert") or {}
        if not per_expert:
            return
        device = next(self.v9_key_pool.parameters()).device
        accumulators = self._accumulators(device)
        for key, entry in per_expert.items():
            index = self.v9_index.get(int(key))
            if index is None:
                continue
            with torch.no_grad():
                accumulators["usage"][index] = float(entry.get("usage", 0))
                accumulators["selected"][index] = float(entry.get("selected", 0))
                accumulators["support"][index] = float(entry.get("support", 0.0))
                accumulators["contribution_sum"][index] = float(
                    entry.get("contribution_sum", 0.0)
                )
                accumulators["positive_contribution"][index] = float(
                    entry.get("positive_contribution", 0.0)
                )
                accumulators["positive_count"][index] = float(
                    entry.get("positive_count", 0)
                )


__all__ = ["V9ComposeTrainer", "assert_no_historical_key_gradients"]
