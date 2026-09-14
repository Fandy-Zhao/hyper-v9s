"""Per-sample gradient gating and the trainable-parameter whitelist.

V8 does **not** filter the dataset.  Every current-task sample keeps its reuse
decision, and the decision is expressed as a per-sample weight on the answer
loss (PART 21)::

    L_answer_residual = sum_i r_i * L_answer_i / max(sum_i r_i, 1)

with ``r_i = 0`` for samples the pool already covers (``BaseOnly``, ``Reuse1``,
``Reuse2``) and ``r_i = 1`` only for ``Residual`` samples -- the ones that
genuinely need capability the pool lacks.  A candidate expert therefore learns
only from the residual, instead of re-learning what the pool already does.

Forward-side gating comes for free and needs no new code: the candidate expert
is simply absent from the per-sample ``ComposeSelection`` of a covered sample, so
``ComposeLinear.forward`` never routes that sample's rows into the candidate and
its parameters receive no gradient from them.  This module adds the *loss*-side
weight and the audit that proves the isolation actually held.

``TRAINABLE PARAMETER AUDIT`` is printed at startup and re-checked at the end:
only the current task's alias keys, the current task's candidate LoRA and the
current task's candidate keys may require grad.  Historical LoRA and historical
committed keys are compared by exact tensor checksum and must be UNCHANGED.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from compose.v8.pool import MultiKeyExpertPool, tensor_checksum


class GradientGatingError(RuntimeError):
    """Raised when a frozen parameter would receive gradient, or vice versa."""


@dataclass
class TrainableAudit:
    """What is allowed to train, and what must never change."""

    trainable_residual_keys: List[str] = field(default_factory=list)
    trainable_candidate_experts: List[int] = field(default_factory=list)
    trainable_candidate_keys: List[str] = field(default_factory=list)
    frozen_historical_experts: List[int] = field(default_factory=list)
    frozen_historical_keys: List[str] = field(default_factory=list)
    unexpected_trainable: List[str] = field(default_factory=list)
    base_parameters_trainable: int = 0

    @property
    def ok(self) -> bool:
        return not self.unexpected_trainable and self.base_parameters_trainable == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trainable_alias_keys": self.trainable_residual_keys,
            "trainable_candidate_experts": self.trainable_candidate_experts,
            "trainable_candidate_keys": self.trainable_candidate_keys,
            "frozen_historical_experts": self.frozen_historical_experts,
            "frozen_historical_keys": self.frozen_historical_keys,
            "unexpected_trainable": self.unexpected_trainable,
            "base_parameters_trainable": self.base_parameters_trainable,
            "ok": self.ok,
        }

    def render(self) -> str:
        lines = [
            "==================== TRAINABLE PARAMETER AUDIT ====================",
            f"current-task alias/candidate keys : {len(self.trainable_residual_keys)}"
            + (f" {self.trainable_residual_keys}" if len(self.trainable_residual_keys) <= 12
               else f" {self.trainable_residual_keys[:12]}..."),
            f"current-task candidate experts   : {self.trainable_candidate_experts}",
            f"current-task candidate keys      : {len(self.trainable_candidate_keys)}",
            f"frozen historical experts        : {len(self.frozen_historical_experts)}",
            f"frozen historical keys           : {len(self.frozen_historical_keys)}",
            f"base parameters with grad        : {self.base_parameters_trainable} (must be 0)",
            f"unexpected trainable parameters  : {len(self.unexpected_trainable)} (must be 0)",
            f"AUDIT OK                         : {self.ok}",
            "===================================================================",
        ]
        return "\n".join(lines)


def residual_answer_loss(
    per_sample_nll: torch.Tensor,
    weights: torch.Tensor,
    minimum_denominator: float = 1.0,
) -> torch.Tensor:
    """``sum_i r_i * L_i / max(sum_i r_i, 1)``.

    Degenerates to a no-op (zero) when a batch contains no residual sample,
    which is the correct behaviour: nothing in that batch should train the
    candidate expert.
    """
    if per_sample_nll.ndim != 1 or weights.ndim != 1:
        raise GradientGatingError("per-sample NLL and weights must be rank-1")
    if per_sample_nll.shape != weights.shape:
        raise GradientGatingError(
            f"weight vector {tuple(weights.shape)} does not match "
            f"{tuple(per_sample_nll.shape)} per-sample losses"
        )
    if bool((weights < 0).any()):
        raise GradientGatingError("gradient-gating weights must be non-negative")
    denominator = torch.clamp(weights.sum(), min=float(minimum_denominator))
    return (per_sample_nll * weights).sum() / denominator


def audit_trainable_parameters(
    model: nn.Module,
    pool: MultiKeyExpertPool,
    current_task: int,
    candidate_expert_ids: Iterable[int] = (),
    candidate_parameters: Iterable[torch.Tensor] = (),
) -> TrainableAudit:
    """Whitelist audit: only this task's new keys and candidate LoRA may train.

    Membership is decided by **tensor identity**, mirroring the V7 optimizer
    whitelist (``compose/v7/hf_trainer.py:265``), so it does not depend on module
    naming and cannot be fooled by a renamed parameter.
    """
    candidates = {int(value) for value in candidate_expert_ids}
    audit = TrainableAudit()
    audit.trainable_candidate_experts = sorted(candidates)
    for key_id, record in pool.key_records.items():
        if record["task_id"] == int(current_task) and record["key_type"] == "task_alias":
            if record["trainable"]:
                audit.trainable_residual_keys.append(key_id)
            else:
                audit.frozen_historical_keys.append(key_id)
        else:
            audit.frozen_historical_keys.append(key_id)
    audit.frozen_historical_experts = sorted(
        expert_id for expert_id in pool.expert_records
        if expert_id not in candidates
    )

    allowed = {
        id(parameter) for parameter in pool.parameters() if parameter.requires_grad
    }
    for parameter in candidate_parameters:
        if parameter.requires_grad:
            allowed.add(id(parameter))
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in allowed:
            continue
        audit.base_parameters_trainable += 1
        audit.unexpected_trainable.append(name)

    # A frozen historical key that still asks for gradient is a hard failure.
    for key_id in audit.frozen_historical_keys:
        if bool(pool.keys[key_id].requires_grad):
            audit.unexpected_trainable.append(f"pool.keys[{key_id}]")
    return audit


def enforce_freeze_policy(
    model: nn.Module,
    manager,
    pool: MultiKeyExpertPool,
    current_task: int,
    candidate_expert_ids: Iterable[int] = (),
) -> Dict[str, Any]:
    """Put the hard PART 4 freeze into effect on real tensors.

    Frozen: base backbone, every historical LoRA expert, every historical
    committed key, historical alias keys.  Trainable: this task's candidate
    experts and this task's alias keys -- nothing else.

    Returns the audit; ``audit.ok`` must be True before training starts.
    """
    candidates = {int(value) for value in candidate_expert_ids}
    base_parameters = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.requires_grad_(False)
            base_parameters += 1

    registered = sorted({
        int(key) for layer in manager.layers.values() for key in layer.experts.keys()
    })
    frozen_experts: List[int] = []
    candidate_parameter_list: List[torch.Tensor] = []
    seen_candidates: Dict[int, bool] = {}
    for expert_id in registered:
        is_candidate = expert_id in candidates
        seen_candidates[expert_id] = is_candidate
        for layer in manager.layers.values():
            if str(expert_id) not in layer.experts:
                continue
            expert = layer.experts[str(expert_id)]
            for parameter in expert.parameters():
                parameter.requires_grad_(is_candidate)
                if is_candidate:
                    candidate_parameter_list.append(parameter)
        if not is_candidate:
            frozen_experts.append(expert_id)

    pool.freeze_historical(current_task=int(current_task))
    for key_id, record in pool.key_records.items():
        should_train = (
            record["task_id"] == int(current_task)
            and record["key_type"] == "task_alias"
        )
        pool.set_key_trainable(key_id, should_train)

    audit = audit_trainable_parameters(
        model=model,
        pool=pool,
        current_task=int(current_task),
        candidate_expert_ids=candidates,
        candidate_parameters=candidate_parameter_list,
    )
    return {
        "frozen_base_parameters": base_parameters,
        "frozen_experts": frozen_experts,
        "candidate_experts": sorted(
            expert_id for expert_id, is_candidate in seen_candidates.items() if is_candidate
        ),
        "audit": audit,
    }


def assert_gradient_isolation(
    gradients: Mapping[str, Optional[torch.Tensor]],
    frozen_names: Sequence[str],
) -> None:
    """Frozen parameters must have no gradient (or a vanishing one)."""
    offenders = []
    for name in frozen_names:
        grad = gradients.get(name)
        if grad is not None and bool(torch.count_nonzero(grad).item()):
            offenders.append(name)
    if offenders:
        raise GradientGatingError(
            f"{len(offenders)} frozen parameters received gradient: {offenders[:8]}"
        )


@dataclass
class FrozenLedger:
    """Exact-tensor ledger of everything that must not change during training."""

    key_checksums: Dict[str, str] = field(default_factory=dict)
    lora_checksums: Dict[int, str] = field(default_factory=dict)
    edge_checksums: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "historical_key_checksums": dict(self.key_checksums),
            "historical_lora_checksums": {str(k): v for k, v in self.lora_checksums.items()},
            "candidate_edges": dict(self.edge_checksums),
        }


def capture_frozen_ledger(
    pool: MultiKeyExpertPool,
    lora_tensors: Mapping[int, Mapping[str, torch.Tensor]],
    candidate_expert_ids: Iterable[int] = (),
    current_task: Optional[int] = None,
) -> FrozenLedger:
    """Snapshot historical keys and LoRA tensors before training."""
    candidates = {int(value) for value in candidate_expert_ids}
    ledger = FrozenLedger()
    for key_id, record in pool.key_records.items():
        is_historical = (
            record["key_type"] == "origin"
            or (current_task is not None and record["task_id"] != int(current_task))
        )
        if is_historical:
            ledger.key_checksums[key_id] = tensor_checksum(pool.keys[key_id])
    for expert_id, tensors in lora_tensors.items():
        if int(expert_id) in candidates:
            continue
        for name, tensor in tensors.items():
            ledger.edge_checksums[f"expert{int(expert_id)}.{name}"] = tensor_checksum(tensor)
    return ledger


def verify_frozen_ledger(
    ledger: FrozenLedger,
    pool: MultiKeyExpertPool,
    lora_tensors: Mapping[int, Mapping[str, torch.Tensor]],
) -> Dict[str, Any]:
    """Re-check the ledger; any change is a hard failure, not a warning."""
    changed_keys = []
    for key_id, expected in ledger.key_checksums.items():
        if key_id not in pool.key_records:
            raise GradientGatingError(f"historical key {key_id} disappeared")
        if tensor_checksum(pool.keys[key_id]) != expected:
            changed_keys.append(key_id)
    changed_lora = []
    for name, expected in ledger.edge_checksums.items():
        expert_token, _, tensor_name = name.partition(".")
        expert_id = int(expert_token[len("expert"):])
        tensors = lora_tensors.get(expert_id)
        if tensors is None or tensor_name not in tensors:
            raise GradientGatingError(f"historical LoRA tensor {name} disappeared")
        if tensor_checksum(tensors[tensor_name]) != expected:
            changed_lora.append(name)
    if changed_keys or changed_lora:
        raise GradientGatingError(
            "frozen parameters changed during training: "
            f"keys={changed_keys[:8]} lora={changed_lora[:8]}"
        )
    return {
        "historical_keys_checked": len(ledger.key_checksums),
        "historical_lora_tensors_checked": len(ledger.edge_checksums),
        "changed_keys": changed_keys,
        "changed_lora": changed_lora,
        "status": "UNCHANGED",
    }


def gradient_leakage_probe(
    model: nn.Module,
    parameter_names: Sequence[str],
    batch_residual_only: Mapping[str, Any],
    batch_mixed: Mapping[str, Any],
    forward_backward,
) -> Dict[str, Any]:
    """Prove a mixed batch produces the same candidate gradient as the residual subset.

    ``forward_backward(batch) -> {name: grad tensor}`` must run a full
    forward/backward for one batch and return the gradients of
    ``parameter_names``.  If gating works, adding covered samples to a batch
    changes those gradients by exactly nothing.
    """
    grads_residual = forward_backward(batch_residual_only)
    grads_mixed = forward_backward(batch_mixed)
    report: Dict[str, Any] = {"parameters": list(parameter_names), "max_abs_delta": 0.0}
    worst = None
    for name in parameter_names:
        left = grads_residual.get(name)
        right = grads_mixed.get(name)
        if left is None and right is None:
            continue
        if (left is None) != (right is None):
            raise GradientGatingError(
                f"{name}: gradient present in only one of the two batches"
            )
        delta = float((left - right).abs().max().item())
        if delta > report["max_abs_delta"]:
            report["max_abs_delta"] = delta
            worst = name
    report["worst_parameter"] = worst
    report["leakage"] = bool(report["max_abs_delta"] > 1e-6)
    if report["leakage"]:
        raise GradientGatingError(
            "gradient leakage on a mixed BaseOnly/Reuse batch: candidate "
            f"gradient moved by {report['max_abs_delta']:.3e} on {worst}"
        )
    return report


def state_weight_report(
    sample_ids: Sequence[str],
    state_by_sample: Mapping[str, str],
    weights: torch.Tensor,
) -> Dict[str, Any]:
    """Confirm the gating weights match the states they came from."""
    from compose.v8.selection import residual_weight

    mismatched = []
    counts: Dict[str, int] = {}
    active: Dict[str, int] = {}
    for index, sample_id in enumerate(sample_ids):
        state = state_by_sample[str(sample_id)]
        counts[state] = counts.get(state, 0) + 1
        expected = residual_weight(state)
        if abs(float(weights[index]) - expected) > 1e-6:
            mismatched.append(str(sample_id))
        if float(weights[index]) > 0:
            active[state] = active.get(state, 0) + 1
    return {
        "state_counts": counts,
        "gradient_active_counts": active,
        "mismatched_samples": mismatched,
        "residual_samples": int(weights.gt(0).sum().item()),
        "total_samples": len(sample_ids),
    }


def dumps(audit: TrainableAudit) -> str:
    return json.dumps(audit.to_dict(), ensure_ascii=False, indent=2)


__all__ = [
    "FrozenLedger",
    "GradientGatingError",
    "TrainableAudit",
    "assert_gradient_isolation",
    "audit_trainable_parameters",
    "capture_frozen_ledger",
    "dumps",
    "enforce_freeze_policy",
    "gradient_leakage_probe",
    "residual_answer_loss",
    "state_weight_report",
    "verify_frozen_ledger",
]
