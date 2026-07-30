import unittest

import torch
import torch.nn as nn

from compose.adapters.lora import ComposeLinear
from compose.adapters.runtime import use_selection
from compose.adapters.types import ComposeSelection


def _linear_with_experts():
    base = nn.Linear(2, 1, bias=False)
    nn.init.zeros_(base.weight)
    layer = ComposeLinear(base, rank=1, alpha=1.0)
    first = layer.add_expert(0)
    second = layer.add_expert(1)
    with torch.no_grad():
        first.lora_A.weight.copy_(torch.tensor([[1.0, 0.0]]))
        first.lora_B.weight.copy_(torch.tensor([[2.0]]))
        second.lora_A.weight.copy_(torch.tensor([[0.0, 1.0]]))
        second.lora_B.weight.copy_(torch.tensor([[4.0]]))
    return layer


class ComposeLinearTest(unittest.TestCase):
    def test_sample_level_top1_selection(self):
        layer = _linear_with_experts()
        inputs = torch.tensor([[[3.0, 5.0]], [[7.0, 11.0]]])
        selection = ComposeSelection(
            torch.tensor([[0], [1]], dtype=torch.long),
            torch.tensor([[1.0], [1.0]]),
        )
        with use_selection(selection):
            output = layer(inputs)
        torch.testing.assert_close(output, torch.tensor([[[6.0]], [[44.0]]]))

    def test_sample_level_top2_composition_normalizes_gates(self):
        layer = _linear_with_experts()
        inputs = torch.tensor([[[3.0, 5.0]]])
        selection = ComposeSelection(
            torch.tensor([[0, 1]], dtype=torch.long),
            torch.tensor([[1.0, 3.0]]),
            normalization="l1",
        )
        with use_selection(selection):
            output = layer(inputs)
        torch.testing.assert_close(output, torch.tensor([[[16.5]]]))

    def test_selection_rejects_more_than_two_experts(self):
        with self.assertRaisesRegex(ValueError, "top_k"):
            ComposeSelection(torch.tensor([[0, 1, 2]]), torch.ones(1, 3))
