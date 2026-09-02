"""Transformers integration for V7 dynamic global Top-2 training."""

import os
import json
from collections import Counter
from pathlib import Path

import torch

from compose.adapters.types import pad_selection
from compose.train.trainer import ComposeTrainer

from .checkpoint import load_v7_checkpoint, save_v7_checkpoint
from .pool import V7ExpertKeyPool
from .routing import GlobalTop2Router
from .training import (
    V7JsonlLogger,
    adapter_checksums,
    assert_historical_key_gradients_frozen,
    assert_historical_lora_frozen,
    selected_current_key_loss,
)


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


class V7ComposeTrainer(ComposeTrainer):
    def __init__(
        self,
        *args,
        v7_key_pool: V7ExpertKeyPool,
        v7_config,
        v7_task_index: int,
        v7_metrics_path: str,
        **kwargs
    ) -> None:
        self.v7_key_pool = v7_key_pool
        self.v7_config = v7_config
        self.v7_task_index = int(v7_task_index)
        self.v7_router = GlobalTop2Router(v7_key_pool)
        self.v7_logger = V7JsonlLogger(v7_metrics_path)
        self.v7_usage = {str(value): 0 for value in v7_key_pool.current_ids}
        self.v7_noop_steps = 0
        self._v7_active = None
        super().__init__(*args, **kwargs)
        self._historical_key_before = v7_key_pool.historical_checksums()
        self._historical_lora_before = adapter_checksums(
            self.expert_pool.manager, v7_key_pool.historical_ids
        )

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        key_parameters = [
            self.v7_key_pool.keys[str(value)] for value in self.v7_key_pool.current_ids
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
                    "lr": self.v7_config.training.lora_learning_rate,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": key_parameters,
                    "lr": self.v7_config.training.key_learning_rate,
                    "weight_decay": 0.0,
                },
            ]
        )
        return self.optimizer

    def trainable_parameter_audit(self):
        return {
            "current_key_parameters": sum(value.numel() for value in self.v7_key_pool.parameters() if value.requires_grad),
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

    def training_step(self, model, inputs):
        queries = inputs.get("fixed_queries")
        if queries is None:
            raise ValueError("V7 batch is missing fixed_queries")
        key_device = next(self.v7_key_pool.parameters()).device
        queries = queries.to(key_device)
        inputs["fixed_queries"] = queries
        routed = self.v7_router(queries)
        inputs["compose_selections"] = padded_compose_selections(routed.selection)
        current_selected = set(routed.expert_ids.detach().cpu().view(-1).tolist()) & set(
            self.v7_key_pool.current_ids
        )
        self._v7_active = (queries, routed, current_selected)
        for expert_id in routed.expert_ids.detach().cpu().view(-1).tolist():
            if str(expert_id) in self.v7_usage:
                self.v7_usage[str(expert_id)] += 1
        loss = super().training_step(model, inputs)
        assert_historical_key_gradients_frozen(self.v7_key_pool)
        assert_historical_lora_frozen(
            self.expert_pool.manager, self.v7_key_pool.historical_ids
        )
        for expert_id in set(self.v7_key_pool.current_ids) - current_selected:
            gradient = self.v7_key_pool.keys[str(expert_id)].grad
            if gradient is not None and bool(gradient.detach().ne(0).any()):
                raise AssertionError("unselected current key received gradient")
        self._write_step_metrics()
        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        queries = inputs.pop("fixed_queries")
        inputs.pop("sample_ids", None)
        result = super().compute_loss(model, inputs, return_outputs=return_outputs)
        answer_loss, outputs = result if return_outputs else (result, None)
        if self._v7_active is None:
            raise RuntimeError("V7 routing must run before compute_loss")
        _, routed, current_selected = self._v7_active
        key_loss, per_sample = selected_current_key_loss(
            queries, routed.expert_ids, self.v7_key_pool
        )
        total = answer_loss + self.v7_config.training.lambda_key * key_loss
        if not current_selected:
            # Old+Old has no expert graph. A zero-valued current-key anchor
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
        selected_key_grad = sum(
            float(self.v7_key_pool.keys[str(value)].grad.detach().float().square().sum())
            for value in current_selected
            if self.v7_key_pool.keys[str(value)].grad is not None
        ) ** 0.5
        selected_lora_grad = sum(
            float(parameter.grad.detach().float().square().sum())
            for value in current_selected
            for layer in self.expert_pool.manager.layers.values()
            for parameter in layer.experts[str(value)].parameters()
            if parameter.grad is not None
        ) ** 0.5
        self.v7_logger.write(
            {
                "step": int(self.state.global_step),
                "answer_loss": float(answer),
                "key_loss": float(key),
                "total_loss": float(total),
                "per_sample_key_loss": per_sample.cpu().tolist(),
                "selected_expert_ids": routed.expert_ids.detach().cpu().tolist(),
                "route_types": list(routed.route_types),
                "selected_current_ids": sorted(int(value) for value in current_selected),
                "old_old_noop": not bool(current_selected),
                "selected_current_key_grad_norm": selected_key_grad,
                "selected_current_lora_grad_norm": selected_lora_grad,
            }
        )

    def final_diagnostics(self, num_train_samples):
        route_counts = Counter()
        selection_counts = Counter()
        pair_counts = Counter()
        cross_task_pairs = Counter()
        losses = Counter()
        rows = 0
        with self.v7_logger.path.open("r", encoding="utf-8") as handle:
            for line in handle:
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

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics)
        checkpoint_dir = os.path.join(
            self._get_output_dir(trial=trial),
            "checkpoint-{}".format(self.state.global_step),
        )
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
            usage_counters=self.v7_usage,
            config=self.v7_config,
            rms_state={},
        )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        path = Path(resume_from_checkpoint) / "v7_state.pt"
        payload, loaded_pool, loaded_config = load_v7_checkpoint(
            str(path), restore_rng=True
        )
        if loaded_config != self.v7_config:
            raise ValueError("V7 resume config mismatch")
        if loaded_pool.expert_ids != self.v7_key_pool.expert_ids:
            raise ValueError("V7 resume expert registry mismatch")
        for expert_id in loaded_pool.expert_ids:
            self.v7_key_pool.keys[str(expert_id)].data.copy_(loaded_pool.keys[str(expert_id)])
            self.v7_key_pool.metadata[expert_id] = dict(loaded_pool.metadata[expert_id])
        self.v7_key_pool.pool_version = loaded_pool.pool_version
        self.v7_key_pool.freeze_historical()
        load_candidate_lora_state(
            self.expert_pool.manager, payload["candidate_lora_state"]
        )
        self.v7_usage = dict(payload["candidate_usage_counters"])

    def assert_task_freeze_integrity(self):
        after_keys = self.v7_key_pool.historical_checksums()
        after_lora = adapter_checksums(
            self.expert_pool.manager, self.v7_key_pool.historical_ids
        )
        if after_keys != self._historical_key_before:
            raise AssertionError("historical key checksum changed during task")
        if after_lora != self._historical_lora_before:
            raise AssertionError("historical LoRA checksum changed during task")
        return {"historical_key_unchanged": True, "historical_lora_unchanged": True}
