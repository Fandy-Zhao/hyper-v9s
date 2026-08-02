import unittest

from compose.experts import ExpertActivationContext, ExpertMetadata, ExpertRegistry


class FakeBridge:
    def __init__(self):
        self.active = ()
        self.trainable = ()

    def snapshot_runtime_state(self):
        return {"active": self.active, "trainable": self.trainable}

    def restore_runtime_state(self, state):
        self.active, self.trainable = state["active"], state["trainable"]

    def set_active_experts(self, values):
        self.active = tuple(values)

    def set_trainable_experts(self, values):
        self.trainable = tuple(values)


def _registry():
    registry = ExpertRegistry()
    for expert_id in (0, 1):
        registry.register(ExpertMetadata(expert_id=expert_id, adapter_name=str(expert_id)))
    return registry


class ExpertActivationContextTest(unittest.TestCase):
    def test_nested_contexts_restore_outer_then_initial_state(self):
        registry, bridge = _registry(), FakeBridge()
        with ExpertActivationContext(registry, bridge, [0], []):
            self.assertEqual(bridge.active, (0,))
            with ExpertActivationContext(registry, bridge, [1], [1]):
                self.assertEqual(bridge.trainable, (1,))
            self.assertEqual(bridge.active, (0,))
            self.assertEqual(registry.active_expert_ids, (0,))
        self.assertEqual(bridge.active, ())
        self.assertEqual(registry.active_expert_ids, ())

    def test_exception_exit_restores_and_subset_is_default(self):
        registry, bridge = _registry(), FakeBridge()
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with ExpertActivationContext(registry, bridge, [0], [0]):
                raise RuntimeError("boom")
        self.assertEqual((bridge.active, bridge.trainable), ((), ()))
        with self.assertRaisesRegex(ValueError, "subset"):
            with ExpertActivationContext(registry, bridge, [0], [1]):
                pass


if __name__ == "__main__":
    unittest.main()
