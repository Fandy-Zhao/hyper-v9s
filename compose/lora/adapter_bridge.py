"""The sole V6 connection from Compose state to LoRA execution."""

from collections import defaultdict
from typing import Any, Dict, Iterable, Sequence, Tuple

import torch
import torch.nn as nn


def _ordered_unique(values: Iterable[int]) -> Tuple[int, ...]:
    return tuple(dict.fromkeys(int(value) for value in values))


class AdapterBridge:
    """Expose independent expert deltas without adding V6 semantics to Hyper."""

    def __init__(self, model: nn.Module, adapter_name: str = None, verify_ddp: bool = True) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be torch.nn.Module")
        self.model, self.verify_ddp = model, bool(verify_ddp)
        records = []
        for name, module in model.named_modules():
            if all(hasattr(module, field) for field in ("lora_A", "lora_B", "cur_task", "disable_adapters")):
                records.append((name, module, "hyper"))
            elif hasattr(module, "base_layer") and hasattr(module, "experts") and hasattr(module, "clear_default_selection"):
                records.append((name, module, "compose"))
        if not records:
            raise ValueError("model has no compatible LoRA layers")
        kinds = {kind for _, _, kind in records}
        if len(kinds) != 1:
            raise ValueError("mixed Hyper and Compose LoRA layers are unsupported")
        self._records = records
        self._layers = [module for _, module, _ in records]
        self._kind = next(iter(kinds))
        inferred_names = {str(getattr(layer, "active_adapter", "")) for layer in self._layers} if self._kind == "hyper" else set()
        if self._kind == "hyper" and adapter_name is None:
            if len(inferred_names) != 1 or "" in inferred_names:
                raise ValueError("could not infer one active Hyper adapter name")
            adapter_name = next(iter(inferred_names))
        self.adapter_name = str(adapter_name) if adapter_name is not None else "compose"
        self._expert_parameters = self._discover_expert_parameters()
        self._expert_ids = tuple(sorted(self._expert_parameters))
        if not self._expert_ids:
            raise ValueError("adapter contains no experts")
        self._active_ids = ()
        self._trainable_ids = tuple(expert_id for expert_id, parameters in self._expert_parameters.items()
                                    if parameters and all(parameter.requires_grad for _, parameter in parameters))
        self._pair_execution_depth = 0
        self._forward_hooks = [layer.register_forward_pre_hook(self._guard_forward) for layer in self._layers]

    @property
    def expert_ids(self):
        return self._expert_ids

    @property
    def named_layers(self):
        return tuple((name, module) for name, module, _ in self._records)

    def _discover_expert_parameters(self):
        result = defaultdict(list)
        if self._kind == "compose":
            for name, parameter in self.model.named_parameters():
                parts = name.split(".")
                if "experts" in parts:
                    index = parts.index("experts")
                    if index + 1 < len(parts):
                        try:
                            result[int(parts[index + 1])].append((name, parameter))
                        except ValueError:
                            pass
            return dict(result)
        for name, parameter in self.model.named_parameters():
            parts = name.split(".")
            for container, owner in (("loraA", "lora_A"), ("loraB", "lora_B")):
                if container not in parts:
                    continue
                index = parts.index(container)
                if index >= 2 and index + 1 < len(parts) and parts[index - 2] == owner and parts[index - 1] == self.adapter_name:
                    try:
                        result[int(parts[index + 1])].append((name, parameter))
                    except ValueError:
                        pass
                    break
        for layer in self._layers:
            try:
                if len(layer.lora_A[self.adapter_name].loraA) != len(layer.lora_B[self.adapter_name].loraB):
                    raise ValueError("Hyper LoRA A/B expert counts disagree")
            except (KeyError, AttributeError) as error:
                raise ValueError("invalid Hyper adapter storage") from error
        return dict(result)

    def _require_experts(self, values: Iterable[int]) -> Tuple[int, ...]:
        ordered = _ordered_unique(values)
        missing = sorted(set(ordered) - set(self._expert_ids))
        if missing:
            raise KeyError("unregistered experts: {}".format(missing))
        return ordered

    def _verify_rank_ids(self, role: str, values: Tuple[int, ...]) -> None:
        if not self.verify_ddp or not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, values)
        if any(tuple(item) != values for item in gathered):
            raise RuntimeError("DDP ranks disagree on {} expert IDs: {}".format(role, gathered))

    def _guard_forward(self, module, inputs) -> None:
        if len(self._active_ids) > 1 and self._pair_execution_depth <= 0:
            raise NotImplementedError("Stage 02 pair state requires a Stage 03 CompositionRuntime")

    def enable_pair_execution(self) -> None:
        self._pair_execution_depth += 1

    def disable_pair_execution(self) -> None:
        if self._pair_execution_depth <= 0:
            raise RuntimeError("pair execution depth underflow")
        self._pair_execution_depth -= 1

    def compute_expert_delta(self, module, expert_id: int, hidden_states: torch.Tensor) -> torch.Tensor:
        expert_id = self._require_experts((expert_id,))[0]
        if module not in self._layers:
            raise KeyError("module is not registered with this bridge")
        if self._kind == "compose":
            return module.experts[str(expert_id)](hidden_states)
        adapter = self.adapter_name
        expert_dtype = module.lora_A[adapter].loraA[expert_id].weight.dtype
        inputs = module.lora_dropout[adapter](hidden_states.to(expert_dtype)) if hasattr(module, "lora_dropout") else hidden_states.to(expert_dtype)
        value = module.lora_A[adapter].loraA[expert_id](inputs)
        scaling = module.scaling[adapter] if hasattr(module, "scaling") else 1.0
        return module.lora_B[adapter].loraB[expert_id](value) * scaling

    def set_active_experts(self, expert_ids: Sequence[int]) -> None:
        ordered = self._require_experts(expert_ids)
        self._verify_rank_ids("active", ordered)
        for layer in self._layers:
            if self._kind == "compose":
                if len(ordered) == 1:
                    layer.set_default_selection(ordered, [1.0], normalization="none")
                else:
                    layer.clear_default_selection()
            elif not ordered or len(ordered) == 2:
                layer.disable_adapters = True
            else:
                layer.disable_adapters = False
                layer.cur_task = ordered[0]
                layer.training = True
        self._active_ids = ordered

    def set_trainable_experts(self, expert_ids: Sequence[int]) -> None:
        ordered = self._require_experts(expert_ids)
        self._verify_rank_ids("trainable", ordered)
        selected = set(ordered)
        for expert_id, parameters in self._expert_parameters.items():
            for _, parameter in parameters:
                parameter.requires_grad_(expert_id in selected)
        self._trainable_ids = ordered

    def get_active_experts(self):
        return self._active_ids

    def get_trainable_experts(self):
        return self._trainable_ids

    def freeze_all_experts(self) -> None:
        self.set_trainable_experts(())

    def verify_grad_flags(self) -> Dict[str, Any]:
        experts = {}
        for expert_id, parameters in sorted(self._expert_parameters.items()):
            flags = {name: parameter.requires_grad for name, parameter in parameters}
            experts[str(expert_id)] = {"parameter_count": len(flags), "all_trainable": bool(flags) and all(flags.values()),
                                       "all_frozen": all(not value for value in flags.values()), "flags": flags}
        return {"active_expert_ids": list(self._active_ids), "trainable_expert_ids": list(self._trainable_ids), "experts": experts}

    def snapshot_runtime_state(self) -> Dict[str, Any]:
        layers = []
        for layer in self._layers:
            if self._kind == "compose":
                layers.append({"default_expert_ids": layer._default_expert_ids, "default_gates": layer._default_gates,
                               "default_normalization": layer._default_normalization})
            else:
                layers.append({"disable_adapters": bool(layer.disable_adapters), "cur_task": int(layer.cur_task),
                               "training": bool(layer.training), "active_adapter": getattr(layer, "active_adapter", None)})
        return {"active_expert_ids": list(self._active_ids), "trainable_expert_ids": list(self._trainable_ids),
                "pair_execution_depth": self._pair_execution_depth, "layers": layers,
                "requires_grad": {name: parameter.requires_grad for name, parameter in self.model.named_parameters()}}

    def restore_runtime_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or len(state.get("layers", [])) != len(self._layers):
            raise ValueError("bridge runtime state has incompatible layers")
        for layer, saved in zip(self._layers, state["layers"]):
            if self._kind == "compose":
                layer._default_expert_ids, layer._default_gates = saved["default_expert_ids"], saved["default_gates"]
                layer._default_normalization = saved["default_normalization"]
            else:
                layer.disable_adapters, layer.cur_task, layer.training = bool(saved["disable_adapters"]), int(saved["cur_task"]), bool(saved["training"])
                if saved.get("active_adapter") is not None:
                    layer.active_adapter = saved["active_adapter"]
        parameter_map = dict(self.model.named_parameters())
        if set(parameter_map) != set(state.get("requires_grad", {})):
            raise ValueError("bridge runtime state parameter names do not match the model")
        for name, parameter in parameter_map.items():
            parameter.requires_grad_(bool(state["requires_grad"][name]))
        self._active_ids = tuple(map(int, state.get("active_expert_ids", [])))
        self._trainable_ids = tuple(map(int, state.get("trainable_expert_ids", [])))
        self._pair_execution_depth = int(state.get("pair_execution_depth", 0))

    def close(self) -> None:
        for hook in self._forward_hooks:
            hook.remove()
        self._forward_hooks = []
