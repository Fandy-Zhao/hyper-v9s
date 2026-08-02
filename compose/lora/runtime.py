"""Scoped coupling of Registry, activation state, bridge and composer."""

from typing import Iterable, List

from compose.experts import ExpertActivationContext


class CompositionRuntime:
    """Execute one frozen composition selection and restore exact prior state."""

    def __init__(self, registry, bridge, composer, active_expert_ids: Iterable[int], trainable_expert_ids: Iterable[int], mode: str) -> None:
        self.registry, self.bridge, self.composer = registry, bridge, composer
        self.active_expert_ids = tuple(int(value) for value in active_expert_ids)
        self.trainable_expert_ids = tuple(int(value) for value in trainable_expert_ids)
        self.mode = mode
        self._activation = None
        self._hooks = []  # type: List[object]

    def audit_snapshot(self):
        return {"active_expert_ids": list(self.active_expert_ids), "trainable_expert_ids": list(self.trainable_expert_ids), "mode": self.mode}

    def __enter__(self):
        canonical = self.composer.validate(self.active_expert_ids, self.mode)
        if not set(self.trainable_expert_ids).issubset(canonical):
            raise ValueError("trainable experts must be a subset of active experts")
        self.active_expert_ids = canonical
        self._activation = ExpertActivationContext(self.registry, self.bridge, canonical, self.trainable_expert_ids)
        self._activation.__enter__()
        try:
            # Single remains on the exact Stage-02/Hyper execution branch.
            if self.mode in ("direct_sum", "rms_calibrated"):
                self.bridge.enable_pair_execution()
                for layer_name, module in self.bridge.named_layers:
                    def hook(current_module, inputs, base_output, name=layer_name):
                        if not inputs:
                            raise ValueError("LoRA layer forward hook received no hidden states")
                        return self.composer.forward(current_module, inputs[0], canonical, self.mode,
                                                     {"base_output": base_output, "layer_name": name}).output
                    self._hooks.append(module.register_forward_hook(hook))
        except BaseException:
            self._activation.__exit__(*__import__("sys").exc_info())
            self._activation = None
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in reversed(self._hooks):
            hook.remove()
        if self.mode in ("direct_sum", "rms_calibrated") and self._activation is not None:
            self.bridge.disable_pair_execution()
        self._hooks = []
        if self._activation is not None:
            self._activation.__exit__(exc_type, exc_value, traceback)
            self._activation = None
        return False
