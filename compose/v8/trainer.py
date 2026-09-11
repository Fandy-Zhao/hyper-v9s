"""The V8 current-task training loop: one task's gated training stage.

Everything that decides *what* may change is already implemented and tested on
its own: the freeze policy and the trainable-parameter whitelist
(``gating.enforce_freeze_policy``, ``gating.audit_trainable_parameters``), the
per-sample gradient gating (``selection.build_selection`` +
``gating.residual_answer_loss``), the alias-key objective
(``key_learning.alias_key_loss``), the frozen-tensor ledger
(``gating.capture_frozen_ledger`` / ``verify_frozen_ledger``), the pruning,
commit and checkpoint contracts.  What did not exist was the loop that runs
them, in order, over one task's samples.  Without it the V8 code was a set of
parts with no assembly and V8-B had no entry point, so this module is assembly
rather than new mechanism.

Three deliberate choices:

* **The dataset is never filtered** (spec PART 46 item 4).  A batch holds the
  current task's samples exactly as the teacher classified them -- BaseOnly,
  Reuse1, Reuse2 and Residual together -- and the covered samples are
  neutralised by their per-sample weight ``r_i = 0``, not by being dropped.
  Dropping them would silently change what "one epoch" means and would hide the
  gating from the leakage probe.
* **The forward is injected.**  ``forward_fn(sample_ids) -> rank-1 NLL tensor``
  is the only model-facing seam.  In a real run it is a thin adapter over the
  frozen backbone and ``compose.v7.training.teacher_forcing_token_nll``; in
  tests it is a tiny ``ComposeLinear`` stack.  Both drive the *same* trainer
  code, so the gating, the whitelist and the ledger are exercised for real
  without a GPU.
* **The freeze is checked, not assumed.**  The ledger is captured before the
  optimizer is built and re-verified in :meth:`V8TaskTrainer.finalize`; a
  changed historical tensor raises rather than warns.

Scope boundary, stated so it is not mistaken for an omission: candidate-expert
*redundancy* pruning is planned and reported (``plan_candidate_pruning``) but
not applied automatically.  Removing a redundant candidate means removing the
expert, its only key is its origin key, and the redundancy threshold
(``candidate_redundancy_cosine``) has not been validated on this pool -- so the
plan is handed to the commit decision instead of being executed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch
from torch import nn

from compose.adapters.runtime import use_selection
from compose.v8.checkpoint import save_checkpoint
from compose.v8.commit import commit_task
from compose.v8.config import V8Config, V8KeyConfig
from compose.v8.gating import (
    FrozenLedger,
    GradientGatingError,
    audit_trainable_parameters,
    capture_frozen_ledger,
    enforce_freeze_policy,
    gradient_leakage_probe,
    residual_answer_loss,
    state_weight_report,
    verify_frozen_ledger,
)
from compose.v8.key_learning import alias_key_loss, build_key_targets
from compose.v8.pruning import (
    apply_pruning,
    assert_pool_not_emptied,
    plan_candidate_pruning,
    plan_key_pruning,
)
from compose.v8.selection import build_selection, residual_weights


class TrainerError(RuntimeError):
    """Raised when a training batch or a training contract is not satisfiable."""


@dataclass
class TrainBatch:
    """One mixed batch: the samples, their teacher states and their routes."""

    sample_ids: List[str]
    state_by_sample: Mapping[str, str]
    experts_by_sample: Mapping[str, Sequence[int]]

    def __post_init__(self) -> None:
        self.sample_ids = [str(value) for value in self.sample_ids]
        if not self.sample_ids:
            raise TrainerError("a training batch needs at least one sample")
        missing = [
            sample_id for sample_id in self.sample_ids
            if sample_id not in self.state_by_sample
            or sample_id not in self.experts_by_sample
        ]
        if missing:
            raise TrainerError(f"batch is missing states/routes for {missing[:5]}")


@dataclass
class TrainReport:
    """What one epoch actually did, in enough detail to audit it."""

    steps: int = 0
    micro_batches: int = 0
    sample_count: int = 0
    residual_samples: int = 0
    covered_samples: int = 0
    answer_loss_sum: float = 0.0
    key_loss_sum: float = 0.0
    key_steps: int = 0
    states: Dict[str, int] = field(default_factory=dict)
    gating: List[Dict[str, Any]] = field(default_factory=list)
    gradient_experts: List[int] = field(default_factory=list)
    gradient_keys: List[str] = field(default_factory=list)
    frozen_gradient_offenders: List[str] = field(default_factory=list)

    @property
    def mean_answer_loss(self) -> float:
        return self.answer_loss_sum / self.micro_batches if self.micro_batches else 0.0

    @property
    def mean_key_loss(self) -> float:
        return self.key_loss_sum / self.key_steps if self.key_steps else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": int(self.steps),
            "micro_batches": int(self.micro_batches),
            "sample_count": int(self.sample_count),
            "residual_samples": int(self.residual_samples),
            "covered_samples": int(self.covered_samples),
            "mean_answer_loss": self.mean_answer_loss,
            "mean_key_loss": self.mean_key_loss,
            "key_steps": int(self.key_steps),
            "states": dict(self.states),
            "gradient_experts": list(self.gradient_experts),
            "gradient_keys": list(self.gradient_keys),
            "frozen_gradient_offenders": list(self.frozen_gradient_offenders),
            "gating": list(self.gating),
        }


class V8TaskTrainer:
    """Trains one task's candidate experts and alias keys, and nothing else."""

    def __init__(
        self,
        *,
        model: nn.Module,
        manager,
        pool,
        config: V8Config,
        current_task: int,
        candidate_expert_ids: Sequence[int],
        forward_fn: Callable[[Sequence[str]], torch.Tensor],
        queries_by_sample: Optional[Mapping[str, torch.Tensor]] = None,
        key_config: Optional[V8KeyConfig] = None,
    ) -> None:
        self.model = model
        self.manager = manager
        self.pool = pool
        self.config = config
        self.current_task = int(current_task)
        self.candidate_expert_ids = sorted({int(value) for value in candidate_expert_ids})
        self.forward_fn = forward_fn
        self.queries_by_sample = dict(queries_by_sample or {})
        self.key_config = key_config or config.key

        if not self.candidate_expert_ids:
            raise TrainerError(
                "a V8 task trains at least one candidate expert; an empty "
                "candidate list would make the whitelist vacuous"
            )

        self.freeze_report = enforce_freeze_policy(
            model, manager, pool,
            current_task=self.current_task,
            candidate_expert_ids=self.candidate_expert_ids,
        )
        if not self.freeze_report["audit"].ok:
            raise GradientGatingError(
                "trainable-parameter audit failed:\n"
                + self.freeze_report["audit"].render()
            )
        self.ledger: FrozenLedger = capture_frozen_ledger(
            pool,
            self.lora_tensors(),
            candidate_expert_ids=self.candidate_expert_ids,
            current_task=self.current_task,
        )
        self.global_step = 0
        self.last_targets = None
        self._gradient_experts: set = set()
        self._gradient_keys: set = set()
        self.whitelist = self.named_whitelist()
        self.optimizer = self.build_optimizer()

    # ------------------------------------------------------------------
    # parameter surfaces
    # ------------------------------------------------------------------
    def lora_tensors(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Every registered expert's LoRA tensors, keyed by expert then name."""
        return {
            int(expert_id): {
                "{}.{}".format(layer_name, name): tensor
                for layer_name, layer in sorted(self.manager.layers.items())
                for name, tensor in layer.experts[str(expert_id)].state_dict().items()
            }
            for expert_id in self.registered_expert_ids()
        }

    def registered_expert_ids(self) -> List[int]:
        return sorted({
            int(key)
            for layer in self.manager.layers.values()
            for key in layer.experts.keys()
        })

    def named_whitelist(self) -> Dict[str, torch.Tensor]:
        """The trainable surface, by name: candidate LoRAs then alias keys.

        This is the *whole* trainable surface.  A parameter that requests
        gradient and is not in here is a freeze violation, which is why
        :meth:`build_optimizer` re-runs ``audit_trainable_parameters`` on the
        real model rather than trusting this list.
        """
        named: Dict[str, torch.Tensor] = {}
        for expert_id in self.candidate_expert_ids:
            for layer_name, layer in sorted(self.manager.layers.items()):
                if str(expert_id) not in layer.experts:
                    raise TrainerError(f"candidate expert {expert_id} is not registered")
                for name, parameter in sorted(
                    layer.experts[str(expert_id)].named_parameters()
                ):
                    named["candidate{}.{}.{}".format(expert_id, layer_name, name)] = parameter
        for key_id, record in sorted(self.pool.key_records.items()):
            if (
                record["task_id"] == self.current_task
                and record["key_type"] == "task_alias"
                and record["lifecycle"] != "pruned"
            ):
                named[key_id] = self.pool.keys[key_id]
        return named

    def build_optimizer(self) -> torch.optim.Optimizer:
        audit = audit_trainable_parameters(
            model=self.model,
            pool=self.pool,
            current_task=self.current_task,
            candidate_expert_ids=self.candidate_expert_ids,
            candidate_parameters=[
                parameter
                for expert_id in self.candidate_expert_ids
                for layer in self.manager.layers.values()
                for parameter in layer.experts[str(expert_id)].parameters()
            ],
        )
        if not audit.ok:
            raise GradientGatingError(audit.render())
        lora_parameters = [
            parameter for name, parameter in self.whitelist.items()
            if name.startswith("candidate")
        ]
        key_parameters = [
            parameter for name, parameter in self.whitelist.items()
            if not name.startswith("candidate")
        ]
        groups = []
        if lora_parameters:
            groups.append({
                "params": lora_parameters,
                "lr": float(self.config.training.learning_rate),
                "weight_decay": float(self.config.training.weight_decay),
                "name": "candidate_lora",
            })
        if key_parameters:
            groups.append({
                "params": key_parameters,
                "lr": float(self.key_config.learning_rate),
                "weight_decay": 0.0,
                "name": "alias_keys",
            })
        if not groups:
            raise TrainerError("the whitelist is empty; nothing would be trained")
        return torch.optim.AdamW(groups)

    # ------------------------------------------------------------------
    # one epoch
    # ------------------------------------------------------------------
    def train_epoch(
        self,
        batches: Sequence[TrainBatch],
        teacher_result=None,
        progress=None,
    ) -> TrainReport:
        """Run one epoch of mixed gated batches.

        ``teacher_result`` is only needed to train alias keys: it supplies the
        three-valued targets.  Without it the epoch trains the candidate expert
        alone, which is the correct behaviour for a task whose teacher found no
        reusable history.
        """
        if not batches:
            raise TrainerError("an epoch needs at least one batch")
        targets = None
        if teacher_result is not None and self.queries_by_sample:
            targets = build_key_targets(teacher_result, self.pool, self.current_task)
        report = TrainReport()
        log = progress or (lambda message: None)
        self.optimizer.zero_grad(set_to_none=True)

        for index, batch in enumerate(batches):
            weights = residual_weights(batch.sample_ids, batch.state_by_sample)
            selection = build_selection(
                batch.sample_ids, batch.state_by_sample, batch.experts_by_sample
            )
            with use_selection(selection):
                per_sample = self.forward_fn(batch.sample_ids)
            if per_sample.ndim != 1 or per_sample.shape[0] != len(batch.sample_ids):
                raise TrainerError(
                    "forward_fn must return one NLL per sample in batch order; "
                    "got {}".format(tuple(per_sample.shape))
                )
            loss = residual_answer_loss(per_sample, weights)
            key_report = None
            if targets is not None:
                key_report = alias_key_loss(
                    self.queries_by_sample, self.pool, targets, self.key_config
                )
                loss = loss + key_report.total
            loss.backward()
            self._record_gradient_footprint()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

            report.micro_batches += 1
            report.answer_loss_sum += float(loss.detach().item())
            report.sample_count += len(batch.sample_ids)
            report.residual_samples += int((weights > 0).sum().item())
            report.covered_samples += int((weights == 0).sum().item())
            report.steps = self.global_step
            report.gating.append(
                state_weight_report(batch.sample_ids, batch.state_by_sample, weights)
            )
            for state in batch.state_by_sample.values():
                report.states[state] = report.states.get(state, 0) + 1
            if key_report is not None:
                report.key_loss_sum += float(key_report.total.detach().item())
                report.key_steps += 1
            log("batch {}/{}: {} residual, {} covered".format(
                index + 1, len(batches), report.residual_samples, report.covered_samples
            ))

        report.gradient_experts = sorted(self._gradient_experts)
        report.gradient_keys = sorted(self._gradient_keys)
        report.frozen_gradient_offenders = self.frozen_gradient_offenders()
        if report.frozen_gradient_offenders:
            raise GradientGatingError(
                "frozen parameters received gradient: {}".format(
                    report.frozen_gradient_offenders[:8]
                )
            )
        self.last_targets = targets
        return report

    # ------------------------------------------------------------------
    def _record_gradient_footprint(self) -> None:
        for name, parameter in self.whitelist.items():
            gradient = parameter.grad
            if gradient is None or not bool(torch.count_nonzero(gradient)):
                continue
            if name.startswith("candidate"):
                expert_id = int(name[len("candidate"):].split(".", 1)[0])
                self._gradient_experts.add(expert_id)
            else:
                self._gradient_keys.add(name)

    def frozen_gradient_offenders(self) -> List[str]:
        """Any frozen tensor that carries a non-zero gradient."""
        offenders: List[str] = []
        for expert_id, tensors in self.lora_tensors().items():
            if expert_id in self.candidate_expert_ids:
                continue
            for name, tensor in tensors.items():
                if tensor.grad is not None and bool(torch.count_nonzero(tensor.grad)):
                    offenders.append("expert{}.{}".format(expert_id, name))
        for key_id in sorted(self.ledger.key_checksums):
            gradient = self.pool.keys[key_id].grad
            if gradient is not None and bool(torch.count_nonzero(gradient)):
                offenders.append(key_id)
        return offenders

    def leakage_probe(
        self,
        batch_residual_only: TrainBatch,
        batch_mixed: TrainBatch,
        parameter_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Prove the covered samples in a mixed batch change nothing.

        Runs on the standing optimizer and model, and clears the gradients
        afterwards, so it can be called mid-training.  It is deliberately not
        part of ``train_epoch``: it costs two extra forward/backward passes, and
        it belongs in the audit trail once per task rather than once per batch.
        """
        named = self.whitelist
        names = list(parameter_names) if parameter_names else list(named)

        def forward_backward(batch: TrainBatch) -> Dict[str, Optional[torch.Tensor]]:
            weights = residual_weights(batch.sample_ids, batch.state_by_sample)
            selection = build_selection(
                batch.sample_ids, batch.state_by_sample, batch.experts_by_sample
            )
            self.optimizer.zero_grad(set_to_none=True)
            with use_selection(selection):
                per_sample = self.forward_fn(batch.sample_ids)
            residual_answer_loss(per_sample, weights).backward()
            return {
                name: (
                    named[name].grad.detach().clone()
                    if name in named and named[name].grad is not None else None
                )
                for name in names
            }

        try:
            return gradient_leakage_probe(
                self.model, names, batch_residual_only, batch_mixed, forward_backward
            )
        finally:
            self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    # end of task
    # ------------------------------------------------------------------
    def verify_freeze(self) -> Dict[str, Any]:
        """Re-check every frozen tensor; raises if anything moved."""
        return verify_frozen_ledger(self.ledger, self.pool, self.lora_tensors())

    def candidate_redundancy(
        self,
        reference_vectors: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Report candidate experts that duplicate an already-committed key.

        Not applied automatically: see the module docstring's scope boundary.
        """
        candidate_keys = {
            key_id: self.pool.keys[key_id].detach()
            for key_id, record in sorted(self.pool.key_records.items())
            if record["expert_id"] in self.candidate_expert_ids
            and record["lifecycle"] != "pruned"
        }
        existing_keys = {
            key_id: (
                vector if isinstance(vector, torch.Tensor) else torch.as_tensor(vector)
            )
            for key_id, vector in dict(reference_vectors or {}).items()
            if key_id not in candidate_keys
        }
        if not existing_keys:
            existing_keys = {
                key_id: self.pool.keys[key_id].detach()
                for key_id, record in sorted(self.pool.key_records.items())
                if record["expert_id"] not in self.candidate_expert_ids
                and record["lifecycle"] != "pruned"
            }
        return plan_candidate_pruning(
            candidate_keys, existing_keys, self.config.pruning
        )

    def finalize(
        self,
        checkpoint_path: Optional[str | Path] = None,
        commit: bool = True,
    ) -> Dict[str, Any]:
        """Prune, verify the freeze, commit the task and checkpoint the pool."""
        decisions = [
            decision for decision in plan_key_pruning(self.pool)
            if decision.key_id not in self.ledger.key_checksums
        ]
        apply_pruning(self.pool, decisions)
        assert_pool_not_emptied(self.pool)

        report: Dict[str, Any] = {
            "pruned_keys": [decision.key_id for decision in decisions],
            "candidate_redundancy": self.candidate_redundancy(),
            "ledger": self.verify_freeze(),
        }
        if commit:
            commit_report = commit_task(
                self.pool,
                task_id=self.current_task,
                candidate_expert_ids=self.candidate_expert_ids,
            )
            report["commit"] = commit_report.to_dict()
        if checkpoint_path is not None:
            report["checkpoint"] = save_checkpoint(
                checkpoint_path,
                self.pool,
                self.config,
                current_task=self.current_task,
                global_step=self.global_step,
                optimizer_state=self.optimizer.state_dict(),
                frozen_ledger=self.ledger,
            )
        return report

    def state_dict(self) -> Dict[str, Any]:
        return {
            "global_step": int(self.global_step),
            "optimizer": self.optimizer.state_dict(),
            "gradient_experts": sorted(self._gradient_experts),
            "gradient_keys": sorted(self._gradient_keys),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.global_step = int(state.get("global_step", 0))
        optimizer_state = state.get("optimizer")
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)


__all__ = [
    "TrainBatch",
    "TrainReport",
    "TrainerError",
    "V8TaskTrainer",
]
