"""Scoped expert activation with exact restoration on every exit path."""

from typing import Any, Dict, Iterable, Optional

from .registry import ExpertRegistry


class ExpertActivationContext:
    """Apply registry and bridge state for one lexical execution scope.

    Context instances are deliberately single-use.  Separate instances may be
    nested; each snapshots the current registry and bridge state and restores it
    in ``__exit__``, including when model execution raises.
    """

    def __init__(
        self,
        registry: ExpertRegistry,
        bridge,
        active_expert_ids: Iterable[int],
        trainable_expert_ids: Iterable[int],
        allow_trainable_outside_active: bool = False,
    ) -> None:
        self.registry = registry
        self.bridge = bridge
        self.active_expert_ids = tuple(dict.fromkeys(int(value) for value in active_expert_ids))
        self.trainable_expert_ids = tuple(dict.fromkeys(int(value) for value in trainable_expert_ids))
        self.allow_trainable_outside_active = bool(allow_trainable_outside_active)
        self._registry_state = None  # type: Optional[Dict[str, Any]]
        self._bridge_state = None  # type: Optional[Dict[str, Any]]
        self._entered = False

    def audit_snapshot(self) -> Dict[str, Any]:
        return {
            "active_expert_ids": list(self.active_expert_ids),
            "trainable_expert_ids": list(self.trainable_expert_ids),
            "allow_trainable_outside_active": self.allow_trainable_outside_active,
        }

    def __enter__(self) -> "ExpertActivationContext":
        if self._entered:
            raise RuntimeError("ExpertActivationContext instances are single-use")
        if not self.allow_trainable_outside_active and not set(self.trainable_expert_ids).issubset(
            self.active_expert_ids
        ):
            raise ValueError("trainable experts must be a subset of active experts")
        # Registry validation also rejects missing and archived experts.
        for expert_id in self.active_expert_ids + self.trainable_expert_ids:
            self.registry.get(expert_id)
        self._registry_state = self.registry.state_dict()
        self._bridge_state = self.bridge.snapshot_runtime_state()
        try:
            self.registry.set_active_ids(self.active_expert_ids)
            self.registry.set_trainable_ids(self.trainable_expert_ids)
            self.registry.validate()
            self.bridge.set_active_experts(self.active_expert_ids)
            self.bridge.set_trainable_experts(self.trainable_expert_ids)
        except BaseException:
            self.registry.load_state_dict(self._registry_state)
            self.bridge.restore_runtime_state(self._bridge_state)
            raise
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self._entered:
            try:
                self.bridge.restore_runtime_state(self._bridge_state)
            finally:
                self.registry.load_state_dict(self._registry_state)
            self._entered = False
        return False
