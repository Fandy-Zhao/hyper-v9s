"""Selected-current V7 losses, step execution and gradient isolation audits."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from compose.adapters.runtime import use_selection

from .pool import V7ExpertKeyPool
from .routing import GlobalTop2Result, GlobalTop2Router


def full_data_coverage_audit(
    num_train_samples: int,
    unique_sample_ids: Iterable[str],
    optimizer_micro_steps: int,
    optimizer_steps: int,
    observed_sample_count: int,
    require_full: bool,
) -> Dict[str, object]:
    sample_count = int(num_train_samples)
    unique_count = len(set(str(value) for value in unique_sample_ids))
    coverage = unique_count / sample_count if sample_count else 0.0
    effective_epochs = observed_sample_count / sample_count if sample_count else 0.0
    result = {
        "num_train_samples": sample_count,
        "unique_train_sample_ids_seen": unique_count,
        "optimizer_micro_steps": int(optimizer_micro_steps),
        "optimizer_steps": int(optimizer_steps),
        "effective_epochs": float(effective_epochs),
        "train_sample_coverage": float(coverage),
        "full_data_required": bool(require_full),
    }
    if require_full and unique_count != sample_count:
        raise RuntimeError(
            "formal V7 training did not cover the full declared split: {}".format(result)
        )
    return result


def supervised_token_mask(labels: Tensor, ignore_index: int = -100) -> Tensor:
    if labels.ndim != 2:
        raise ValueError("labels must have shape [B,T]")
    return labels[:, 1:].ne(ignore_index)


def per_sample_teacher_forcing_token_nll(
    logits: Tensor, labels: Tensor, ignore_index: int = -100
) -> Tensor:
    """Return one answer-token mean NLL per sample.

    Summing these values across a micro-batch preserves the established V7
    batch-size-one accumulation contract when execution batches are packed.
    """
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits/labels must have shapes [B,T,V] and [B,T]")
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = supervised_token_mask(labels, ignore_index)
    counts = valid.sum(dim=1)
    if bool(counts.eq(0).any()):
        raise ValueError("every sample requires at least one supervised answer token")
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(shift_labels)
    return (losses * valid).sum(dim=1) / counts


def teacher_forcing_token_nll(logits: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
    """Standard next-token NLL averaged over target answer tokens only."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits/labels must have shapes [B,T,V] and [B,T]")
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = supervised_token_mask(labels, ignore_index)
    if not bool(valid.any()):
        raise ValueError("answer loss requires at least one supervised token")
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(shift_labels)
    return losses[valid].mean()


def selected_current_key_loss(
    queries: Tensor,
    selected_ids: Tensor,
    key_pool: V7ExpertKeyPool,
) -> Tuple[Tensor, Tensor]:
    """Mean per-sample attraction over selected current keys; old keys excluded."""
    if queries.ndim != 2 or selected_ids.shape != (queries.shape[0], 2):
        raise ValueError("queries and selected ids must be [B,D] and [B,2]")
    detached_queries = F.normalize(queries.detach().float(), dim=-1)
    current = set(key_pool.current_ids)
    per_sample = []
    for row in range(queries.shape[0]):
        terms = []
        for expert_id in selected_ids[row].detach().cpu().tolist():
            expert_id = int(expert_id)
            if expert_id in current:
                key = F.normalize(key_pool.keys[str(expert_id)], dim=0)
                terms.append(1.0 - F.cosine_similarity(detached_queries[row], key, dim=0))
        per_sample.append(
            torch.stack(terms).mean()
            if terms
            else detached_queries.new_zeros(())
        )
    values = torch.stack(per_sample)
    return values.mean(), values


def _grad_norm(parameters: Iterable[Tensor]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum())
    return total ** 0.5


def assert_historical_key_gradients_frozen(key_pool: V7ExpertKeyPool) -> None:
    for expert_id in key_pool.historical_ids:
        parameter = key_pool.keys[str(expert_id)]
        if parameter.requires_grad:
            raise AssertionError("historical key {} is trainable".format(expert_id))
        if parameter.grad is not None and bool(parameter.grad.detach().ne(0).any()):
            raise AssertionError("historical key {} received gradient".format(expert_id))


def adapter_checksums(manager, expert_ids: Iterable[int]) -> Dict[int, str]:
    result = {}
    for expert_id in expert_ids:
        digest = hashlib.sha256()
        for layer_name, layer in sorted(manager.layers.items()):
            expert = layer.experts[str(int(expert_id))]
            for name, value in sorted(expert.state_dict().items()):
                tensor = value.detach().cpu().contiguous()
                digest.update(layer_name.encode("utf-8"))
                digest.update(name.encode("utf-8"))
                digest.update(str(tensor.dtype).encode("utf-8"))
                digest.update(str(tuple(tensor.shape)).encode("utf-8"))
                digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        result[int(expert_id)] = digest.hexdigest()
    return result


def assert_historical_lora_frozen(manager, historical_ids: Iterable[int]) -> None:
    for expert_id in historical_ids:
        for layer in manager.layers.values():
            for parameter in layer.experts[str(int(expert_id))].parameters():
                if parameter.requires_grad:
                    raise AssertionError("historical LoRA {} is trainable".format(expert_id))
                if parameter.grad is not None and bool(parameter.grad.detach().ne(0).any()):
                    raise AssertionError("historical LoRA {} received gradient".format(expert_id))


class V7JsonlLogger:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: Mapping[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


class V7StepEngine:
    """Small reusable step core used by the real trainer and unit smokes."""

    def __init__(
        self,
        router: GlobalTop2Router,
        lambda_key: float,
        adapter_manager=None,
        logger: Optional[V7JsonlLogger] = None,
    ) -> None:
        self.router = router
        self.key_pool = router.key_pool
        self.lambda_key = float(lambda_key)
        self.adapter_manager = adapter_manager
        self.logger = logger
        self.step = 0
        self.route_counts = Counter()
        self.selection_counts = Counter()
        self.pair_counts = Counter()
        self.old_old_noop_steps = 0

    def compute(
        self,
        queries: Tensor,
        answer_loss_fn: Callable[[object], Tensor],
    ) -> Tuple[Tensor, Dict[str, object], GlobalTop2Result]:
        routed = self.router(queries)
        with use_selection(routed.selection):
            answer_loss = answer_loss_fn(routed.selection)
        key_loss, per_sample_key = selected_current_key_loss(
            queries, routed.expert_ids, self.key_pool
        )
        total = answer_loss + self.lambda_key * key_loss
        for route_type in routed.route_types:
            self.route_counts[route_type] += 1
        for pair in routed.expert_ids.detach().cpu().tolist():
            canonical = tuple(sorted(int(value) for value in pair))
            self.pair_counts[canonical] += 1
            self.selection_counts.update(canonical)
        current_selected = sorted(
            set(routed.expert_ids.detach().cpu().view(-1).tolist())
            & set(self.key_pool.current_ids)
        )
        no_trainable_selected = not current_selected
        if no_trainable_selected:
            self.old_old_noop_steps += 1
        metrics = {
            "step": self.step,
            "answer_loss": float(answer_loss.detach()),
            "key_loss": float(key_loss.detach()),
            "total_loss": float(total.detach()),
            "per_sample_key_loss": [float(value) for value in per_sample_key.detach().cpu()],
            "route_types": list(routed.route_types),
            "selected_expert_ids": routed.expert_ids.detach().cpu().tolist(),
            "selected_current_ids": current_selected,
            "old_old_noop": no_trainable_selected,
        }
        return total, metrics, routed

    def backward(self, total: Tensor, metrics: Dict[str, object]) -> bool:
        if bool(metrics["old_old_noop"]):
            # There is intentionally no expert graph for Old+Old. Returning
            # without backward makes the optimizer step a safe no-op.
            return False
        total.backward()
        assert_historical_key_gradients_frozen(self.key_pool)
        if self.adapter_manager is not None:
            assert_historical_lora_frozen(
                self.adapter_manager, self.key_pool.historical_ids
            )
        selected = set(int(value) for value in metrics["selected_current_ids"])
        metrics["selected_current_key_grad_norm"] = _grad_norm(
            self.key_pool.keys[str(value)] for value in selected
        )
        unselected = set(self.key_pool.current_ids) - selected
        for expert_id in unselected:
            gradient = self.key_pool.keys[str(expert_id)].grad
            if gradient is not None and bool(gradient.detach().ne(0).any()):
                raise AssertionError("unselected current key received gradient")
        if self.adapter_manager is not None:
            metrics["selected_current_lora_grad_norm"] = _grad_norm(
                parameter
                for expert_id in selected
                for layer in self.adapter_manager.layers.values()
                for parameter in layer.experts[str(expert_id)].parameters()
            )
        return True

    def finish_step(self, metrics: Dict[str, object]) -> None:
        if self.logger is not None:
            self.logger.write(metrics)
        self.step += 1

    def summary(self) -> Dict[str, object]:
        total = sum(self.route_counts.values())
        return {
            "samples": total,
            "OldOldRate": self.route_counts["OldOld"] / total if total else 0.0,
            "OldNewRate": self.route_counts["OldNew"] / total if total else 0.0,
            "NewNewRate": self.route_counts["NewNew"] / total if total else 0.0,
            "selection_count": {str(k): v for k, v in self.selection_counts.items()},
            "pair_frequency": {"{},{}".format(*k): v for k, v in self.pair_counts.items()},
            "old_old_noop_steps": self.old_old_noop_steps,
        }
