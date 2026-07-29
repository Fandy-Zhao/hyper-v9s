import unittest

import torch
import torch.nn as nn

from compose.adapters.inject import inject_compose_adapters
from compose.adapters.lora import ComposeLinear
from compose.config import ComposeAdapterConfig


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(3, 3)
        self.other = nn.Linear(3, 3)


class InjectionTest(unittest.TestCase):
    def test_injection_targets_only_named_linear_leaves(self):
        model = TinyBlock()
        original_weight = model.q_proj.weight.detach().clone()
        matches = inject_compose_adapters(
            model, ComposeAdapterConfig(rank=2, alpha=4, target_modules=["q_proj"])
        )
        self.assertEqual(matches, ["q_proj"])
        self.assertIsInstance(model.q_proj, ComposeLinear)
        self.assertIsInstance(model.other, nn.Linear)
        torch.testing.assert_close(model.q_proj.weight, original_weight)

    def test_injection_fails_when_nothing_matches(self):
        with self.assertRaisesRegex(ValueError, "no target"):
            inject_compose_adapters(
                TinyBlock(), ComposeAdapterConfig(target_modules=["not_present"])
            )
