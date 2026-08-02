"""The sole V6 connection from Compose state to Hyper LoRA execution."""

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn


def _ordered_unique(values: Iterable[int]) -> Tuple[int, ...]:
    return tuple(dict.fromkeys(int(value) for value in values))


class AdapterBridge:
    """Control Hyper experts without teaching Hyper about V6 concepts.

    The bridge recognizes the general Hyper LoRA storage contract
    (``lora_A[adapter].loraA`` and ``lora_B[adapter].loraB``).  It owns no model
    tensors.  State is instance-local and therefore must not be shared between
    concurrent threads; each DDP process owns its own bridge, with optional
    cross-rank ID consistency checks.
    """

    def __init__(self, model: nn.Module, adapter_name: str = None, verify_ddp: bool = True) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be torch.nn.Module")
        self.model = model
        self.verify_ddp = bool(verify_ddp)
        self._layers = [
            module
            for module in model.modules()
            if hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and hasattr(module, "cur_task")
            and hasattr(module, "disable_adapters")
        ]
        if not self._layers:
            raise ValueError("model has no compatible Hyper LoRA layers")
        inferred_names = {str(getattr(layer, "active_adapter", "")) for layer in self._layers}
        if adapter_name is None:
            if len(inferred_names) != 1 or "" in inferred_names:
                raise ValueError("could not infer one active Hyper adapter name")
            adapter_name = next(iter(inferred_names))
        self.adapter_name = str(adapter_name)
        self._expert_parameters = self._discover_expert_parameters()
        self._expert_ids = tuple(sorted(self._expert_parameters))
        if not self._expert_ids:
            raise ValueError("Hyper adapter {!r} contains no experts".format(self.adapter_name))
        self._active_ids = ()  # type: Tuple[int, ...]
        self._trainable_ids = tuple(
            expert_id
            for expert_id, parameters in self._expert_parameters.items()
            if parameters and all(parameter.requires_grad for _, parameter in parameters)
        )
        self._forward_hooks = [layer.register_forward_pre_hook(self._guard_forward) for layer in self._layers]

    def _discover_expert_parameters(self):
        result = defaultdict(list)
        for parameter_name, parameter in self.model.named_parameters():
            parts = parameter_name.split(".")
            for container, owner in (("loraA", "lora_A"), ("loraB", "lora_B")):
                if container not in parts:
                    continue
                index = parts.index(container)
                if (
                    index < 2
                    or index + 1 >= len(parts)
                    or parts[index - 2] != owner
                    or parts[index - 1] != self.adapter_name
                ):
                    continue
                try:
                    expert_id = int(parts[index + 1])
                except ValueError:
                    continue
                result[expert_id].append((parameter_name, parameter))
                break
        # Every compatible Hyper layer must expose both A and B for each expert.
        for layer in self._layers:
            try:
                lora_a = layer.lora_A[self.adapter_name].loraA
                lora_b = layer.lora_B[self.adapter_name].loraB
            except (KeyError, AttributeError) as error:
                raise ValueError("invalid Hyper adapter storage for {!r}".format(self.adapter_name)) from error
            if len(lora_a) != len(lora_b):
                raise ValueError("Hyper LoRA A/B expert counts disagree")
        return {key: value for key, value in result.items()}

    def _require_experts(self, values: Iterable[int]) -> Tuple[int, ...]:
        ordered = _ordered_unique(values)
        missing = sorted(set(ordered) - set(self._expert_ids))
        if missing:
            raise KeyError("unregistered Hyper experts: {}".format(missing))
        return ordered

    def _verify_rank_ids(self, role: str, values: Tuple[int, ...]) -> None:
        if not self.verify_ddp or not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, values)
        if any(tuple(item) != values for item in gathered):
            raise RuntimeError("DDP ranks disagree on {} expert IDs: {}".format(role, gathered))

    def _guard_forward(self, module, inputs) -> None:
        if len(self._active_ids) > 1:
            raise NotImplementedError(
                "Stage 02 records multiple active experts but does not execute their combined delta"
            )

    def set_active_experts(self, expert_ids: Sequence[int]) -> None:
        ordered = self._require_experts(expert_ids)
        self._verify_rank_ids("active", ordered)
        for layer in self._layers:
            if not ordered:
                layer.disable_adapters = True
            elif len(ordered) == 1:
                layer.disable_adapters = False
                layer.cur_task = ordered[0]
                # Hyper's historical single-expert branch is selected by this
                # runtime flag.  Child dropout train/eval state is untouched.
                layer.training = True
            else:
                # State/gradient representation is allowed in Stage 02.  The
                # pre-forward guard prevents any silent first-expert fallback.
                layer.disable_adapters = False
        self._active_ids = ordered

    def set_trainable_experts(self, expert_ids: Sequence[int]) -> None:
        ordered = self._require_experts(expert_ids)
        self._verify_rank_ids("trainable", ordered)
        selected = set(ordered)
        for expert_id, parameters in self._expert_parameters.items():
            for _, parameter in parameters:
                parameter.requires_grad_(expert_id in selected)
        self._trainable_ids = ordered

    def get_active_experts(self) -> Tuple[int, ...]:
        return self._active_ids

    def get_trainable_experts(self) -> Tuple[int, ...]:
        return self._trainable_ids

    def freeze_all_experts(self) -> None:
        self.set_trainable_experts(())

    def verify_grad_flags(self) -> Dict[str, Any]:
        experts = {}
        for expert_id, parameters in sorted(self._expert_parameters.items()):
            flags = {name: parameter.requires_grad for name, parameter in parameters}
            experts[str(expert_id)] = {
                "parameter_count": len(flags),
                "all_trainable": bool(flags) and all(flags.values()),
                "all_frozen": all(not value for value in flags.values()),
                "flags": flags,
            }
        return {
            "active_expert_ids": list(self._active_ids),
            "trainable_expert_ids": list(self._trainable_ids),
            "experts": experts,
        }

    def snapshot_runtime_state(self) -> Dict[str, Any]:
        return {
            "active_expert_ids": list(self._active_ids),
            "trainable_expert_ids": list(self._trainable_ids),
            "layers": [
                {
                    "disable_adapters": bool(layer.disable_adapters),
                    "cur_task": int(layer.cur_task),
                    "training": bool(layer.training),
                    "active_adapter": getattr(layer, "active_adapter", None),
                }
                for layer in self._layers
            ],
            "requires_grad": {
                name: parameter.requires_grad for name, parameter in self.model.named_parameters()
            },
        }

    def restore_runtime_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("bridge runtime state must be a dictionary")
        layers = state.get("layers")
        if not isinstance(layers, list) or len(layers) != len(self._layers):
            raise ValueError("bridge runtime state has incompatible layers")
        for layer, saved in zip(self._layers, layers):
            layer.disable_adapters = bool(saved["disable_adapters"])
            layer.cur_task = int(saved["cur_task"])
            layer.training = bool(saved["training"])
            if saved.get("active_adapter") is not None:
                layer.active_adapter = saved["active_adapter"]
        parameter_map = dict(self.model.named_parameters())
        saved_flags = state.get("requires_grad", {})
        if set(parameter_map) != set(saved_flags):
            raise ValueError("bridge runtime state parameter names do not match the model")
        for name, parameter in parameter_map.items():
            parameter.requires_grad_(bool(saved_flags[name]))
        self._active_ids = tuple(int(value) for value in state.get("active_expert_ids", []))
        self._trainable_ids = tuple(int(value) for value in state.get("trainable_expert_ids", []))

    def close(self) -> None:
        for hook in self._forward_hooks:
            hook.remove()
        self._forward_hooks = []
