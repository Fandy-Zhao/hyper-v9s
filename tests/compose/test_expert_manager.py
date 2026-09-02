import unittest

import torch

from compose.adapters import ExpertManager, inject_compose_adapters
from compose.config import ComposeAdapterConfig
from compose.experts import ExpertPool, ExpertStatus
from test_injection import TinyModel


def _pool():
    model = TinyModel(layer_count=1)
    inject_compose_adapters(model, ComposeAdapterConfig(rank=1, alpha=1))
    manager = ExpertManager(model)
    return model, manager, ExpertPool(manager)


class ExpertManagerTest(unittest.TestCase):
    def test_pool_registration_and_trainability(self):
        _, manager, pool = _pool()
        pool.register(2, name="task-two", origin_task_id="ucit-task-two")
        pool.register(5, name="task-five")
        pool.train_only([5])
        self.assertEqual(pool.expert_ids(), [2, 5])
        self.assertEqual(pool.get(2).status, ExpertStatus.FROZEN)
        self.assertEqual(pool.get(2).origin_task_id, "ucit-task-two")
        self.assertEqual(pool.get(5).status, ExpertStatus.TRAINING)
        layer = next(iter(manager.layers.values()))
        self.assertFalse(any(parameter.requires_grad for parameter in layer.experts["2"].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in layer.experts["5"].parameters()))
        self.assertFalse(layer.base_layer.weight.requires_grad)
        pool.sync_training_step(7)
        pool.sync_training_step(3)
        self.assertEqual(pool.get(5).trained_steps, 7)

    def test_manager_builds_fixed_top2_selection(self):
        _, _, pool = _pool()
        pool.register(0)
        pool.register(1)
        selection = pool.make_selection([0, 1], batch_size=3, gates=[0.2, 0.8])
        # The unified selection is MAX_ACTIVE_EXPERTS=4 slots wide; the two
        # active experts occupy slots 0-1, slots 2-3 are the -1 pads.
        self.assertEqual(selection.expert_ids.shape, (3, 4))
        torch.testing.assert_close(selection.gates.sum(dim=1), torch.ones(3))
