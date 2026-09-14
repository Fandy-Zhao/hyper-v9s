"""Transformers integration for V7 dynamic global Top-2 training."""

import os
import json
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import List

import torch

from compose.adapters.types import pad_selection
from compose.train.profiler import TrainingProfiler, wrap_optimizer_step
from compose.train.trainer import ComposeTrainer
from llava.train.llava_trainer import LengthGroupedSampler

from .checkpoint import load_v7_checkpoint, save_v7_checkpoint
from .pool import V7ExpertKeyPool, tensor_checksum
from .routing import GlobalTop2Router
from .training import (
    V7JsonlLogger,
    adapter_checksums,
    assert_historical_key_gradients_frozen,
    assert_historical_lora_frozen,
    selected_current_key_loss,
    trainable_selection,
    full_data_coverage_audit,
)


def _distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def ranked_metrics_paths(base: Path, world_size: int) -> List[Path]:
    """The metrics files a run of ``world_size`` ranks writes.

    One definition, used by both the writer and the reader in
    :class:`V7ComposeTrainer`.  They used to disagree: the writer kept the
    un-suffixed name when ``world_size == 1`` while the reader checked
    ``torch.distributed.is_initialized()``, which ``torchrun --nproc_per_node 1``
    also makes true -- so a single-process run trained to completion and then
    raised ``FileNotFoundError`` inside ``final_diagnostics``, leaving
    ``v7_training_diagnostics.json`` empty and the job marked failed.
    """
    if world_size <= 1:
        return [base]
    return [
        base.with_name("{}.rank{}{}".format(base.stem, rank, base.suffix))
        for rank in range(world_size)
    ]


def attach_v7_ddp_key_anchor(model, key_pool):
    """Expose current keys to DDP graph discovery without dense execution.

    V7 computes its routing loss outside the wrapped LLaVA forward.  The
    zero-valued dependency added here makes the registered current keys visible
    in DDP's forward output graph, while selected keys still receive the real
    routing-loss gradient and unselected keys remain exact no-ops locally.
    """
    if not _distributed() or torch.distributed.get_world_size() == 1:
        return None
    if getattr(model, "_v7_ddp_key_anchor_handle", None) is not None:
        return model._v7_ddp_key_anchor_handle

    def hook(_module, _inputs, output):
        anchor = sum(
            parameter.sum() * 0.0
            for parameter in key_pool.keys.values()
            if parameter.requires_grad
        )
        if isinstance(output, dict):
            field = "loss" if output.get("loss") is not None else "logits"
            if output.get(field) is None:
                raise RuntimeError("V7 DDP anchor requires loss or logits output")
            output[field] = output[field] + anchor
            return output
        if isinstance(output, tuple) and output:
            return (output[0] + anchor,) + output[1:]
        field = "loss" if getattr(output, "loss", None) is not None else "logits"
        value = getattr(output, field, None)
        if value is None:
            raise RuntimeError("V7 DDP anchor requires loss or logits output")
        setattr(output, field, value + anchor)
        return output

    handle = model.register_forward_hook(hook)
    model._v7_ddp_key_anchor_handle = handle
    return handle


def candidate_lora_state(manager, expert_ids):
    state = {}
    for layer_name, layer in manager.layers.items():
        for expert_id in expert_ids:
            expert = layer.experts[str(int(expert_id))]
            for name, value in expert.state_dict().items():
                state["{}.experts.{}.{}".format(layer_name, expert_id, name)] = value.detach().cpu()
    return state


def load_candidate_lora_state(manager, state):
    expected = set()
    for layer_name, layer in manager.layers.items():
        for expert_id in manager.expert_ids():
            expert = layer.experts[str(int(expert_id))]
            payload = {}
            prefix = "{}.experts.{}.".format(layer_name, expert_id)
            for name in expert.state_dict():
                key = prefix + name
                if key in state:
                    payload[name] = state[key]
                    expected.add(key)
            if payload:
                expert.load_state_dict(payload, strict=True)
    unknown = set(state) - expected
    if unknown:
        raise ValueError("unknown V7 candidate LoRA tensors: {}".format(sorted(unknown)[:5]))


def padded_compose_selections(selection):
    """Adapt V7 Top-2 rows to the unified four-slot execution contract."""
    return [
        pad_selection(
            tuple(record["expert_ids"]),
            tuple(record["gates"]),
        )
        for record in selection.per_sample_sets()
    ]


def resolve_v8_selectable_experts(
    historical_ids, current_ids, reusable_historical_ids, task_index
):
    """Return (reusable old, excluded old, selectable) with fail-closed checks."""
    historical = {int(value) for value in historical_ids}
    current = {int(value) for value in current_ids}
    reusable = (
        historical
        if reusable_historical_ids is None
        else {int(value) for value in reusable_historical_ids}
    )
    if not reusable.issubset(historical):
        raise ValueError("reusable historical experts must belong to the frozen pool")
    if int(task_index) == 0 and reusable:
        raise ValueError("Task0 reusable historical pool must be empty")
    excluded = historical - reusable
    selectable = reusable | current
    if len(selectable) < 2:
        raise ValueError("Global Top-2 needs at least two reusable-old/current experts")
    return tuple(sorted(reusable)), tuple(sorted(excluded)), tuple(sorted(selectable))


def mean_sync_accumulated_gradients(parameters):
    """Mean-reduce a sparse-routing accumulation window exactly once.

    This runs at the accumulation boundary *before* Trainer gradient clipping.
    Doing the reduction inside ``optimizer.step`` would incorrectly clip each
    rank's local gradient before synchronization.  A separate presence reduce
    also makes ``grad=None`` global: if any rank used a parameter, every rank
    takes the same Adam step even when the reduced numerical gradient is zero.
    """
    if not _distributed() or torch.distributed.get_world_size() == 1:
        return False
    grouped = {}
    for parameter in parameters:
        grouped.setdefault((parameter.dtype, parameter.device), []).append(parameter)
    world_size = float(torch.distributed.get_world_size())
    for values in grouped.values():
        present = torch.tensor(
            [parameter.grad is not None for parameter in values],
            dtype=torch.int32,
            device=values[0].device,
        )
        torch.distributed.all_reduce(present, op=torch.distributed.ReduceOp.MAX)
        globally_present = present.cpu().tolist()
        flat = torch.zeros(
            sum(parameter.numel() for parameter in values),
            dtype=values[0].dtype,
            device=values[0].device,
        )
        offset = 0
        for parameter in values:
            size = parameter.numel()
            if parameter.grad is not None:
                if parameter.grad.is_sparse:
                    raise TypeError("V7 mean synchronization requires dense gradients")
                flat[offset:offset + size].copy_(parameter.grad.detach().reshape(-1))
            offset += size
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat.div_(world_size)
        offset = 0
        for index, parameter in enumerate(values):
            size = parameter.numel()
            reduced = flat[offset:offset + size].reshape(parameter.shape)
            if globally_present[index]:
                if parameter.grad is None:
                    parameter.grad = reduced.clone()
                else:
                    parameter.grad.detach().copy_(reduced)
            else:
                parameter.grad = None
            offset += size
    return True


class V7ComposeTrainer(ComposeTrainer):
    def _get_train_sampler(self):
        sampler = super()._get_train_sampler()
        if self.v7_require_full_coverage and isinstance(
            sampler, LengthGroupedSampler
        ):
            # HF Trainer derives optimizer steps from each rank's dataloader
            # length.  Pad the global sampler to a complete global optimizer
            # window so a short final accumulation window cannot be dropped.
            sampler.pad_to_multiple = sampler.batch_size * sampler.world_size
        return sampler

    def __init__(
        self,
        *args,
        v7_key_pool: V7ExpertKeyPool,
        v7_config,
        v7_task_index: int,
        v7_metrics_path: str,
        v7_require_full_coverage: bool = False,
        v7_reusable_historical_ids=None,
        v7_profiler=None,
        **kwargs
    ) -> None:
        self.profiler = v7_profiler if v7_profiler is not None else TrainingProfiler(None)
        self.v7_key_pool = v7_key_pool
        self.v7_config = v7_config
        self.v7_task_index = int(v7_task_index)
        self.v7_router = GlobalTop2Router(v7_key_pool)
        (self.v7_reusable_historical_ids,
         self.v7_excluded_historical_ids,
         self.v7_selectable_expert_ids) = resolve_v8_selectable_experts(
            v7_key_pool.historical_ids,
            v7_key_pool.current_ids,
            v7_reusable_historical_ids,
            self.v7_task_index,
        )
        rank = torch.distributed.get_rank() if _distributed() else 0
        world_size = torch.distributed.get_world_size() if _distributed() else 1
        self.v7_metrics_base_path = Path(v7_metrics_path)
        self.v7_metrics_world_size = world_size
        metrics_path = ranked_metrics_paths(self.v7_metrics_base_path, world_size)[rank]
        self.v7_logger = V7JsonlLogger(str(metrics_path))
        self.v7_usage = {str(value): 0 for value in v7_key_pool.current_ids}
        # Per-route-key counters for the current task's reuse keys: the task-end
        # retention decision reads them, exactly like the candidate counters.
        self.v7_reuse_usage = {
            key_id: 0
            for key_id in sorted(v7_key_pool.trainable_key_ids)
            if v7_key_pool.route_keys[key_id].key_type == "reuse"
        }
        self.v7_reuse_gradient_ids = set()
        self.v7_noop_steps = 0
        self.v7_require_full_coverage = bool(v7_require_full_coverage)
        self.v7_unique_sample_ids = set()
        self.v7_observed_sample_count = 0
        self.v7_micro_steps = 0
        self._v7_key_gradient_ids = set()
        self._v7_lora_gradient_ids = set()
        self._v7_key_gradient_sq = 0.0
        self._v7_lora_gradient_sq = 0.0
        self._v7_gradient_hook_handles = []
        self._v7_active = None
        self._v7_previous_step_end = None
        self._v7_step_timing = None
        super().__init__(*args, **kwargs)
        self._historical_key_before = v7_key_pool.historical_checksums()
        self._historical_lora_before = adapter_checksums(
            self.expert_pool.manager, v7_key_pool.historical_ids
        )
        from compose.lora.rms import runtime_kappa_calibration

        self._historical_rms_before = runtime_kappa_calibration(
            self.model, v7_key_pool.historical_ids
        )
        for key_id in sorted(v7_key_pool.trainable_key_ids):
            self._v7_gradient_hook_handles.append(
                v7_key_pool.keys[key_id].register_hook(
                    lambda gradient, value=key_id: self._record_v7_gradient(
                        "key", value, gradient
                    )
                )
            )
        for expert_id in v7_key_pool.current_ids:
            for layer in self.expert_pool.manager.layers.values():
                for parameter in layer.experts[str(expert_id)].parameters():
                    self._v7_gradient_hook_handles.append(
                        parameter.register_hook(
                            lambda gradient, value=expert_id: self._record_v7_gradient(
                                "lora", value, gradient
                            )
                        )
                    )

    def _record_v7_gradient(self, kind, value, gradient):
        """Record which key id / expert id produced a non-zero gradient.

        ``key`` records route-key ids (strings), ``lora`` records expert ids
        (ints); the two registries are deliberately different objects.
        """
        finite = gradient.detach().float()
        if not bool(torch.isfinite(finite).all()):
            raise FloatingPointError("non-finite V7 {} gradient".format(kind))
        squared = float(finite.square().sum())
        if squared > 0.0:
            getattr(self, "_v7_{}_gradient_ids".format(kind)).add(value)
            attribute = "_v7_{}_gradient_sq".format(kind)
            setattr(self, attribute, getattr(self, attribute) + squared)
            if kind == "key" and value in getattr(self, "v7_reuse_usage", {}):
                self.v7_reuse_gradient_ids.add(value)
        return gradient

    def route_full_data_queries(self, queries):
        """Query-key routing inside reusable-old plus current candidates only.

        This method is intentionally incapable of accepting answers, NLLs, or
        teacher assignments.  Keeping the Stage-B boundary explicit makes the
        zero-oracle contract mechanically testable.
        """
        routed = self.v7_router(
            queries, excluded=self.v7_excluded_historical_ids
        )
        routed_ids = set(routed.expert_ids.detach().cpu().view(-1).tolist())
        forbidden = routed_ids & set(self.v7_excluded_historical_ids)
        if forbidden:
            raise AssertionError(
                "non-reusable historical expert reached full-data routing: {}".format(
                    sorted(forbidden)
                )
            )
        return routed

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        # Exactly three things are trainable: the candidate LoRAs, the candidate
        # keys, and the current task's reuse keys for reusable historical
        # experts.  Historical LoRAs and canonical historical keys are absent
        # by construction: the key list is the explicit trainability registry,
        # never ``expert.lifecycle == "current"``.
        key_parameters = [
            self.v7_key_pool.keys[key_id]
            for key_id in sorted(self.v7_key_pool.trainable_key_ids)
        ]
        lora_parameters = [
            parameter
            for expert_id in self.v7_key_pool.current_ids
            for layer in self.expert_pool.manager.layers.values()
            for parameter in layer.experts[str(expert_id)].parameters()
        ]
        allowed = {id(value) for value in key_parameters + lora_parameters}
        unexpected = [
            name for name, value in self.model.named_parameters()
            if value.requires_grad and id(value) not in allowed
        ]
        if unexpected:
            raise AssertionError(
                "V7 optimizer refuses non-current-Key/LoRA parameters: {}".format(
                    unexpected[:10]
                )
            )
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": lora_parameters,
                    "lr": self.v7_config.training.effective_lora_learning_rate,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": key_parameters,
                    "lr": self.v7_config.training.key_learning_rate,
                    "weight_decay": 0.0,
                },
            ]
        )
        pending = getattr(self, "_v7_pending_optimizer_state", None)
        if pending is not None:
            self.optimizer.load_state_dict(pending)
            self._v7_pending_optimizer_state = None
        wrap_optimizer_step(self.optimizer, self.profiler)
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        scheduler = super().create_scheduler(num_training_steps, optimizer=optimizer)
        pending = getattr(self, "_v7_pending_scheduler_state", None)
        if pending is not None:
            scheduler.load_state_dict(pending)
            self._v7_pending_scheduler_state = None
        return scheduler

    def ddp_backward_loss_scale(self, world_size):
        # The V7 3-GPU recipe defines its real global batch as 1 x 21 x 3.
        # Standard DDP averaging is therefore the intended optimization rule.
        return 1.0

    def trainable_parameter_audit(self):
        audit = {
            "current_key_parameters": sum(value.numel() for value in self.v7_key_pool.parameters() if value.requires_grad),
            "reuse_key_parameters": sum(
                self.v7_key_pool.keys[key_id].numel()
                for key_id in self.v7_key_pool.trainable_key_ids
                if self.v7_key_pool.route_keys[key_id].key_type == "reuse"
            ),
            "trainable_key_ids": sorted(self.v7_key_pool.trainable_key_ids),
            "current_lora_parameters": sum(
                parameter.numel()
                for expert_id in self.v7_key_pool.current_ids
                for layer in self.expert_pool.manager.layers.values()
                for parameter in layer.experts[str(expert_id)].parameters()
                if parameter.requires_grad
            ),
            "historical_key_parameters": sum(
                self.v7_key_pool.keys[str(value)].numel() for value in self.v7_key_pool.historical_ids
            ),
            "query_parameters": 0,
        }
        audit["optimizer_parameter_count"] = sum(
            parameter.numel()
            for group in (self.optimizer.param_groups if self.optimizer else ())
            for parameter in group["params"]
        )
        audit["optimizer_group_parameter_counts"] = [
            sum(parameter.numel() for parameter in group["params"])
            for group in (self.optimizer.param_groups if self.optimizer else ())
        ]
        return audit

    def training_step(self, model, inputs):
        step_started = time.perf_counter()
        inter_step_wait = (
            None
            if self._v7_previous_step_end is None
            else step_started - self._v7_previous_step_end
        )
        self.profiler.micro_step()
        sample_ids = tuple(str(value) for value in inputs.get("sample_ids", ()))
        self._v7_last_sample_ids = sample_ids
        self.v7_unique_sample_ids.update(sample_ids)
        self.v7_observed_sample_count += len(sample_ids)
        self.v7_micro_steps += 1
        self._v7_key_gradient_ids.clear()
        self._v7_lora_gradient_ids.clear()
        self._v7_key_gradient_sq = 0.0
        self._v7_lora_gradient_sq = 0.0
        queries = inputs.get("fixed_queries")
        if queries is None:
            raise ValueError("V7 batch is missing fixed_queries")
        key_device = next(self.v7_key_pool.parameters()).device
        queries = queries.to(key_device)
        inputs["fixed_queries"] = queries
        token_stats = inputs.get("token_stats")
        if token_stats is not None:
            # Counted on the CPU by the collator; calling ``.item()`` here would
            # add one device synchronisation per micro-batch, which is exactly
            # the cost the profiler exists to attribute.
            self.profiler.observe_batch(*token_stats)
        with self.profiler.timed("routing_time"):
            # Stage B is query-key only.  The few-shot teacher contributes one
            # immutable task-level set; no answer, NLL, or sample assignment is
            # available in this call path.
            routed = self.route_full_data_queries(queries)
            inputs["compose_selections"] = padded_compose_selections(routed.selection)
            current_selected = set(
                routed.expert_ids.detach().cpu().view(-1).tolist()
            ) & set(self.v7_key_pool.current_ids)
            selected_trainable_keys = set(trainable_selection(routed, self.v7_key_pool))
            selected_experts = set(routed.expert_ids.detach().cpu().view(-1).tolist())
            self._v7_active = (queries, routed, current_selected)
            for expert_id in routed.expert_ids.detach().cpu().view(-1).tolist():
                if str(expert_id) in self.v7_usage:
                    self.v7_usage[str(expert_id)] += 1
                reuse = sorted(
                    self.v7_key_pool.keys_of_expert(
                        int(expert_id), key_type="reuse", lifecycle="current"
                    )
                )
                for key_id in reuse:
                    self.v7_reuse_usage[key_id] = self.v7_reuse_usage.get(key_id, 0) + 1
        no_sync = model.no_sync() if _distributed() and hasattr(model, "no_sync") else nullcontext()
        with no_sync:
            loss = super().training_step(model, inputs)
        with self.profiler.timed("audit_time"):
            assert_historical_key_gradients_frozen(self.v7_key_pool)
            assert_historical_lora_frozen(
                self.expert_pool.manager, self.v7_key_pool.historical_ids
            )
        if not self._v7_key_gradient_ids.issubset(selected_trainable_keys):
            raise AssertionError(
                "a trainable key of an unselected expert received a new micro-batch "
                "gradient: {}".format(sorted(self._v7_key_gradient_ids - selected_trainable_keys))
            )
        if not self._v7_lora_gradient_ids.issubset(selected_experts):
            raise AssertionError(
                "unselected LoRA received a new micro-batch gradient: {}".format(
                    sorted(self._v7_lora_gradient_ids - selected_experts)
                )
            )
        gradient_window_synced = False
        if self.accelerator.sync_gradients:
            if self.optimizer is None:
                raise RuntimeError("V7 optimizer must exist before gradient synchronization")
            with self.profiler.timed("allreduce_time"):
                gradient_window_synced = mean_sync_accumulated_gradients(
                    parameter
                    for group in self.optimizer.param_groups
                    for parameter in group["params"]
                )
        self._v7_step_timing = {
            "wall_time": time.time(),
            "training_step_sec": time.perf_counter() - step_started,
            "inter_step_wait_sec": inter_step_wait,
            "local_batch_size": len(sample_ids),
            "gradient_window_synced": gradient_window_synced,
        }
        self._write_step_metrics()
        self._v7_previous_step_end = time.perf_counter()
        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        queries = inputs.pop("fixed_queries")
        inputs.pop("sample_ids", None)
        inputs["v7_sum_per_sample_loss"] = True
        result = super().compute_loss(model, inputs, return_outputs=return_outputs)
        answer_loss, outputs = result if return_outputs else (result, None)
        if self._v7_active is None:
            raise RuntimeError("V7 routing must run before compute_loss")
        _, routed, current_selected = self._v7_active
        # Both terms enter the objective as per-micro-batch *means*.
        #
        # HF divides every micro-step's loss by ``gradient_accumulation_steps``
        # before backward, and widening the micro-batch shrinks GA by the same
        # factor -- so a per-micro-batch *sum* reaches the optimizer multiplied
        # by the width, while a mean does not.  The model hands back a sum once
        # the batch exceeds one (``sum_of_per_sample_token_means``) and the same
        # number at width one, so dividing by the width here leaves every
        # existing width-1 run untouched and brings every wider split onto its
        # objective.  A summed key term against a meaned answer term was what put
        # the effective key weight at 0.4 rather than 0.1 at micro 4.
        answer_loss = answer_loss / queries.shape[0]
        _, per_sample = selected_current_key_loss(
            queries, routed.expert_ids, self.v7_key_pool
        )
        key_loss = per_sample.mean()
        total = answer_loss + self.v7_config.training.lambda_key * key_loss
        updated_keys = trainable_selection(routed, self.v7_key_pool)
        if not current_selected and not updated_keys:
            # Nothing trainable was selected. A zero-valued trainable-key anchor
            # permits backward while keeping the optimizer step a true no-op.
            total = total + sum(
                parameter.sum() * 0.0
                for parameter in self.v7_key_pool.keys.values()
                if parameter.requires_grad
            )
            self.v7_noop_steps += 1
        self._v7_losses = (answer_loss.detach(), key_loss.detach(), total.detach(), per_sample.detach())
        return (total, outputs) if return_outputs else total

    def _write_step_metrics(self):
        answer, key, total, per_sample = self._v7_losses
        _, routed, current_selected = self._v7_active
        selected_key_grad = self._v7_key_gradient_sq ** 0.5
        selected_lora_grad = self._v7_lora_gradient_sq ** 0.5
        updated_keys = trainable_selection(routed, self.v7_key_pool)
        payload = {
                "step": int(self.state.global_step),
                "answer_loss": float(answer),
                "key_loss": float(key),
                "total_loss": float(total),
                "per_sample_key_loss": per_sample.cpu().tolist(),
                "selected_expert_ids": routed.expert_ids.detach().cpu().tolist(),
                "route_types": list(routed.route_types),
                "selected_current_ids": sorted(int(value) for value in current_selected),
                # The route key that actually made each selected expert
                # reachable, per slot, plus the key types.  ``route_types``
                # keeps its original expert-level meaning (old expert vs new
                # candidate); these two add the key-level view.
                "selected_key_ids": [list(row) for row in routed.key_ids],
                "selected_key_types": [list(row) for row in routed.key_types],
                "updated_trainable_key_ids": list(updated_keys),
                "old_old_noop": not bool(current_selected) and not bool(updated_keys),
                "selected_current_key_grad_norm": selected_key_grad,
                "selected_current_lora_grad_norm": selected_lora_grad,
                "sample_ids": list(
                    str(value) for value in self._v7_last_sample_ids
                ),
        }
        if self._v7_step_timing is not None:
            payload.update(self._v7_step_timing)
        with self.profiler.timed("metrics_time"):
            self.v7_logger.write(payload)
        self.profiler.bump("micro_batches_logged")

    def final_diagnostics(self, num_train_samples):
        route_counts = Counter()
        selection_counts = Counter()
        pair_counts = Counter()
        cross_task_pairs = Counter()
        losses = Counter()
        rows = 0
        paths = ranked_metrics_paths(
            self.v7_metrics_base_path, self.v7_metrics_world_size
        )
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                records = list(handle)
            for line in records:
                record = json.loads(line)
                rows += 1
                route_counts.update(record["route_types"])
                losses["answer_loss"] += float(record["answer_loss"])
                losses["key_loss"] += float(record["key_loss"])
                losses["total_loss"] += float(record["total_loss"])
                for pair in record["selected_expert_ids"]:
                    pair = tuple(sorted(int(value) for value in pair))
                    pair_counts[pair] += 1
                    selection_counts.update(pair)
                    origins = tuple(
                        int(self.v7_key_pool.metadata[value]["origin_task"])
                        for value in pair
                    )
                    if origins[0] != origins[1]:
                        cross_task_pairs[pair] += 1
        routed_samples = sum(route_counts.values())
        historical = set(self.v7_key_pool.historical_ids)
        historical_selected = sum(
            count for expert_id, count in selection_counts.items() if expert_id in historical
        )
        return {
            "num_train_samples": int(num_train_samples),
            "logged_micro_steps": rows,
            "routed_samples": routed_samples,
            "OldOldRate": route_counts["OldOld"] / routed_samples if routed_samples else 0.0,
            "OldNewRate": route_counts["OldNew"] / routed_samples if routed_samples else 0.0,
            "NewNewRate": route_counts["NewNew"] / routed_samples if routed_samples else 0.0,
            "candidate_selection_count": {
                str(value): selection_counts[value] for value in self.v7_key_pool.current_ids
            },
            "historical_expert_usage": {
                str(value): selection_counts[value] for value in self.v7_key_pool.historical_ids
            },
            "selectable_old_experts": list(self.v7_reusable_historical_ids),
            "excluded_old_experts": list(self.v7_excluded_historical_ids),
            # The current task's reuse keys: how often each fired and how often
            # each actually received key-learning gradient.
            "reuse_key_usage": {
                key_id: int(count)
                for key_id, count in sorted(self.v7_reuse_usage.items())
            },
            "reuse_key_gradient_ids": sorted(self.v7_reuse_gradient_ids),
            "selectable_new_candidates": list(self.v7_key_pool.current_ids),
            "oracle_eval_sample_count": 0,
            "CrossTaskReuseRate": historical_selected / max(1, 2 * routed_samples),
            "expert_pair_frequency": {
                "{},{}".format(*pair): count for pair, count in pair_counts.items()
            },
            "cross_task_expert_pair_frequency": {
                "{},{}".format(*pair): count for pair, count in cross_task_pairs.items()
            },
            "mean_losses": {
                key: value / rows if rows else 0.0 for key, value in losses.items()
            },
        }

    def full_data_coverage_audit(self, num_train_samples):
        local = {
            "rank": torch.distributed.get_rank() if _distributed() else 0,
            "unique_sample_ids": sorted(self.v7_unique_sample_ids),
            "observed_rows": int(self.v7_observed_sample_count),
            "micro_steps": int(self.v7_micro_steps),
        }
        gathered = [local]
        if _distributed():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local)
        unique_ids = sorted({
            value for row in gathered for value in row["unique_sample_ids"]
        })
        observed = sum(row["observed_rows"] for row in gathered)
        audit = full_data_coverage_audit(
            num_train_samples=num_train_samples,
            unique_sample_ids=unique_ids,
            optimizer_micro_steps=max(row["micro_steps"] for row in gathered),
            optimizer_steps=self.state.global_step,
            observed_sample_count=observed,
            require_full=self.v7_require_full_coverage,
        )
        audit.update({
            "world_size": len(gathered),
            "rank_unique_sample_ids_seen": {
                str(row["rank"]): len(row["unique_sample_ids"]) for row in gathered
            },
            "rank_observed_rows": {
                str(row["rank"]): row["observed_rows"] for row in gathered
            },
            "distributed_padding_count": max(0, observed - int(num_train_samples)),
        })
        return audit

    def _local_usage_counters(self):
        return {
            "rank": torch.distributed.get_rank() if _distributed() else 0,
            "candidate_usage": dict(self.v7_usage),
            "reuse_key_usage": dict(self.v7_reuse_usage),
            "reuse_key_gradient_ids": sorted(self.v7_reuse_gradient_ids),
            "noop_micro_steps": int(self.v7_noop_steps),
            "micro_steps": int(self.v7_micro_steps),
            "observed_sample_count": int(self.v7_observed_sample_count),
            "unique_sample_ids": sorted(self.v7_unique_sample_ids),
        }

    def distributed_barrier(self):
        if _distributed():
            torch.distributed.barrier()

    def distributed_state_audit(self, stage):
        optimizer_names = []
        if self.optimizer is not None:
            optimizer_ids = {
                id(parameter)
                for group in self.optimizer.param_groups
                for parameter in group["params"]
            }
            optimizer_names = sorted(
                (name, parameter.numel())
                for name, parameter in self.model.named_parameters()
                if id(parameter) in optimizer_ids
            )
        local = {
            "rank": torch.distributed.get_rank() if _distributed() else 0,
            # Every route key, canonical and reuse alike: a rank that diverged
            # on a reuse key is exactly the failure this audit exists to catch.
            "key_checksums": {
                key_id: tensor_checksum(self.v7_key_pool.keys[key_id])
                for key_id in self.v7_key_pool.key_ids
            },
            "route_key_registry": {
                key_id: self.v7_key_pool.route_keys[key_id].to_dict()
                for key_id in self.v7_key_pool.key_ids
            },
            "current_lora_checksums": adapter_checksums(
                self.expert_pool.manager, self.v7_key_pool.current_ids
            ),
            "optimizer_parameters": optimizer_names,
            "model_router_share_key_pool": self.v7_router.key_pool is self.v7_key_pool,
            "model_registers_same_key_pool": getattr(self.model, "v7_key_pool", None) is self.v7_key_pool,
            "reusable_historical_expert_ids": list(self.v7_reusable_historical_ids),
            "excluded_historical_expert_ids": list(self.v7_excluded_historical_ids),
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
            raise AssertionError("V7 distributed state differs across ranks at {}".format(stage))
        if not local["model_router_share_key_pool"] or not local["model_registers_same_key_pool"]:
            raise AssertionError("V7 model and router must share the registered key pool")
        return {"stage": stage, "world_size": len(gathered), "ranks": gathered}

    def _save_checkpoint(self, model, trial, metrics=None):
        self.distributed_barrier()
        super()._save_checkpoint(model, trial, metrics)
        self.distributed_barrier()
        counters = [self._local_usage_counters()]
        if _distributed():
            counters = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(
                counters, self._local_usage_counters()
            )
        checkpoint_dir = os.path.join(
            self._get_output_dir(trial=trial),
            "checkpoint-{}".format(self.state.global_step),
        )
        if self.args.should_save:
            save_v7_checkpoint(
                os.path.join(checkpoint_dir, "v7_state.pt"),
                task_index=self.v7_task_index,
                training_step=self.state.global_step,
                key_pool=self.v7_key_pool,
                candidate_lora_state=candidate_lora_state(
                    self.expert_pool.manager, self.v7_key_pool.current_ids
                ),
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
                usage_counters={"per_rank": counters},
                config=self.v7_config,
                rms_state={},
            )
        self.distributed_barrier()

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        path = Path(resume_from_checkpoint) / "v7_state.pt"
        payload, loaded_pool, loaded_config = load_v7_checkpoint(
            str(path), restore_rng=True
        )
        if loaded_config != self.v7_config:
            raise ValueError("V7 resume config mismatch")
        if loaded_pool.expert_ids != self.v7_key_pool.expert_ids:
            raise ValueError("V7 resume expert registry mismatch")
        if loaded_pool.key_ids != self.v7_key_pool.key_ids:
            raise ValueError(
                "V7 resume route-key registry mismatch: {} != {}".format(
                    loaded_pool.key_ids, self.v7_key_pool.key_ids
                )
            )
        for key_id in loaded_pool.key_ids:
            self.v7_key_pool.keys[key_id].data.copy_(loaded_pool.keys[key_id])
            if loaded_pool.route_keys[key_id] != self.v7_key_pool.route_keys[key_id]:
                raise ValueError("V7 resume route-key record mismatch for {}".format(key_id))
        for expert_id in loaded_pool.expert_ids:
            self.v7_key_pool.metadata[expert_id] = dict(loaded_pool.metadata[expert_id])
        self.v7_key_pool.pool_version = loaded_pool.pool_version
        # Trainability comes from the checkpoint's own registry: the canonical
        # historical keys are re-frozen, the current task's reuse keys stay
        # trainable, exactly as when the checkpoint was written.
        self.v7_key_pool.trainable_key_ids = set(loaded_pool.trainable_key_ids)
        for key_id in self.v7_key_pool.key_ids:
            self.v7_key_pool._apply_trainable(
                key_id, key_id in self.v7_key_pool.trainable_key_ids
            )
        self.v7_key_pool.freeze_historical()
        load_candidate_lora_state(
            self.expert_pool.manager, payload["candidate_lora_state"]
        )
        counters = dict(payload["candidate_usage_counters"])
        if "per_rank" in counters:
            rank = torch.distributed.get_rank() if _distributed() else 0
            per_rank = {int(row["rank"]): row for row in counters["per_rank"]}
            counters = dict(per_rank[rank])
        if "candidate_usage" in counters:
            self.v7_usage = dict(counters["candidate_usage"])
            self.v7_reuse_usage = {
                key_id: 0 for key_id in self.v7_reuse_usage
            } | dict(counters.get("reuse_key_usage", {}))
            self.v7_reuse_gradient_ids = set(counters.get("reuse_key_gradient_ids", ()))
            self.v7_noop_steps = int(counters.get("noop_micro_steps", 0))
            self.v7_micro_steps = int(counters.get("micro_steps", 0))
            self.v7_observed_sample_count = int(counters.get("observed_sample_count", 0))
            self.v7_unique_sample_ids = set(counters.get("unique_sample_ids", ()))
        else:
            # Backward compatibility with initial V7 checkpoints.
            self.v7_usage = counters
        optimizer_state = payload.get("optimizer")
        scheduler_state = payload.get("scheduler")
        if self.optimizer is not None and optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        else:
            self._v7_pending_optimizer_state = optimizer_state
        if self.lr_scheduler is not None and scheduler_state is not None:
            self.lr_scheduler.load_state_dict(scheduler_state)
        else:
            self._v7_pending_scheduler_state = scheduler_state

    def assert_task_freeze_integrity(self):
        after_keys = self.v7_key_pool.historical_checksums()
        after_lora = adapter_checksums(
            self.expert_pool.manager, self.v7_key_pool.historical_ids
        )
        if after_keys != self._historical_key_before:
            raise AssertionError("historical key checksum changed during task")
        if after_lora != self._historical_lora_before:
            raise AssertionError("historical LoRA checksum changed during task")
        from compose.lora.rms import runtime_kappa_calibration

        after_rms = runtime_kappa_calibration(
            self.model, self.v7_key_pool.historical_ids
        )
        if after_rms != self._historical_rms_before:
            raise AssertionError("historical RMS calibration changed during task")
        return {
            "historical_key_unchanged": True,
            "historical_lora_unchanged": True,
            "historical_rms_unchanged": True,
        }
