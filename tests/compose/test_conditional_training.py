import unittest

import torch

from compose.adapters import ExpertManager
from compose.config import ComposeAdapterConfig
from compose.experts import ExpertPool
from compose.train.data import ComposeSelectionCollator
from compose.train.train_compose import _resolve_expert_roles
from test_injection import TinyModel
from compose.adapters import inject_compose_adapters


class DummyTokenizer:
    pad_token_id = 0
    model_max_length = 16


class ConditionalTrainingTest(unittest.TestCase):
    def test_roles_require_trainable_subset_of_active(self):
        self.assertEqual(_resolve_expert_roles([0, 1], "1"), ([0, 1], [1]))
        self.assertEqual(_resolve_expert_roles([2], ""), ([2], [2]))
        with self.assertRaisesRegex(ValueError, "subset"):
            _resolve_expert_roles([0], "1")

    def test_frozen_old_expert_participates_without_gradient(self):
        model = TinyModel()
        inject_compose_adapters(model, ComposeAdapterConfig(rank=2, alpha=2))
        manager = ExpertManager(model)
        pool = ExpertPool(manager)
        pool.register(0, name="old")
        pool.register(1, name="residual")
        pool.train_only([1])
        manager.set_default_selection([0, 1], [1.0, 1.0], normalization="none")
        output = model.model.layers[0].self_attn.q_proj(torch.randn(2, 3)).sum()
        output.backward()
        old_parameters = [p for n, p in model.named_parameters() if ".experts.0." in n]
        new_parameters = [p for n, p in model.named_parameters() if ".experts.1." in n]
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None for parameter in old_parameters))
        self.assertTrue(all(parameter.requires_grad for parameter in new_parameters))
        self.assertTrue(any(parameter.grad is not None for parameter in new_parameters))

    def test_compose_collator_exposes_supervision_summary(self):
        # Regression: ComposeSelectionCollator delegates the zero-supervision
        # audit to the wrapped legacy collator (train_compose prints the
        # summary after training in cluster_expert mode).
        collator = ComposeSelectionCollator(DummyTokenizer())
        batch = collator([
            {
                "sample_id": "a",
                "expert_ids": [0, 1],
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([1, 2, 3]),
            },
        ])
        self.assertEqual(batch["compose_selections"][0][0], (0, 1, -1, -1))
        self.assertEqual(batch["compose_selections"][0][1], (1.0, 1.0, 0.0, 0.0))
        self.assertEqual(collator.supervision_summary()["samples"], 1)
        self.assertEqual(collator.supervision_summary()["zero_supervision"], 0)


if __name__ == "__main__":
    unittest.main()
