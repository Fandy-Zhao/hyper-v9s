import unittest

import torch
import torch.nn as nn

from compose.adapters.lora import ComposeLinear
from compose.adapters.runtime import use_selection
from compose.adapters.types import PAD_EXPERT_ID, ComposeSelection


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
            torch.tensor([[0, PAD_EXPERT_ID, PAD_EXPERT_ID, PAD_EXPERT_ID], [1, PAD_EXPERT_ID, PAD_EXPERT_ID, PAD_EXPERT_ID]], dtype=torch.long),
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        )
        with use_selection(selection):
            output = layer(inputs)
        torch.testing.assert_close(output, torch.tensor([[[6.0]], [[44.0]]]))

    def test_sample_level_top2_composition_applies_pair_scale(self):
        layer = _linear_with_experts()
        inputs = torch.tensor([[[3.0, 5.0]]])
        selection = ComposeSelection(
            torch.tensor([[0, 1, PAD_EXPERT_ID, PAD_EXPERT_ID]], dtype=torch.long),
            torch.tensor([[1.0, 3.0, 0.0, 0.0]]),
            normalization="l1",
        )
        with use_selection(selection):
            output = layer(inputs)
        # l1-normalized gates: 0.25 / 0.75; deltas 2*3=6 and 4*5=20; the
        # pair composition rule scales the sum by 1/sqrt(2).
        expected = (0.25 * 6.0 + 0.75 * 20.0) / (2.0 ** 0.5)
        torch.testing.assert_close(output, torch.tensor([[[expected]]]))

    def test_selection_rejects_non_four_slot_shapes(self):
        # The unified ComposeSelection is exactly MAX_ACTIVE_EXPERTS=4 slots
        # wide; every slot count other than 4 is rejected.
        with self.assertRaisesRegex(ValueError, "exactly 4 slots"):
            ComposeSelection(torch.tensor([[0, 1]]), torch.ones(1, 2))
        with self.assertRaisesRegex(ValueError, "exactly 4 slots"):
            ComposeSelection(
                torch.tensor([[0, 1, 2, 3, PAD_EXPERT_ID]]), torch.ones(1, 5)
            )
