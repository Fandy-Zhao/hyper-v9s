import unittest

from compose.experts import ExpertMetadata, ExpertRegistry, ExpertStatus


def _metadata(expert_id):
    return ExpertMetadata(expert_id=expert_id, adapter_name="expert-{}".format(expert_id))


class ExpertRegistryTest(unittest.TestCase):
    def test_active_and_trainable_are_ordered_independent_sets(self):
        registry = ExpertRegistry()
        for expert_id in (2, 0, 1):
            registry.register(_metadata(expert_id))
        registry.set_active_ids([1, 0, 1])
        registry.set_trainable_ids([0])
        self.assertEqual(registry.active_expert_ids, (1, 0))
        self.assertEqual(registry.trainable_expert_ids, (0,))
        self.assertEqual([item.expert_id for item in registry.list_all()], [2, 0, 1])

    def test_duplicate_missing_archived_and_bound_unregister_are_protected(self):
        registry = ExpertRegistry()
        registry.register(_metadata(0))
        with self.assertRaises(ValueError):
            registry.register(_metadata(0))
        with self.assertRaises(KeyError):
            registry.set_active_ids([9])
        registry.archive(0)
        self.assertIs(registry.get(0).status, ExpertStatus.ARCHIVED)
        with self.assertRaisesRegex(ValueError, "archived"):
            registry.set_trainable_ids([0])
        bound = ExpertMetadata(expert_id=1, adapter_name="one", checkpoint_path="one.bin")
        registry.register(bound)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            registry.unregister(1)

    def test_state_dict_round_trip_preserves_order_and_roles(self):
        source = ExpertRegistry()
        source.register(_metadata(4))
        source.register(_metadata(2))
        source.set_active_ids([2, 4])
        source.set_trainable_ids([4])
        target = ExpertRegistry()
        target.load_state_dict(source.state_dict())
        self.assertEqual([item.expert_id for item in target.list_all()], [4, 2])
        self.assertEqual(target.active_expert_ids, (2, 4))
        self.assertEqual(target.trainable_expert_ids, (4,))


if __name__ == "__main__":
    unittest.main()
