"""Test L (spec §27): the composition scale rule is actually applied in
``ComposeLinear.forward`` -- single = 1.0, pair = 1/sqrt(2), three-expert
cluster-training selection = 1/sqrt(3)."""

import unittest

import torch
from torch import nn

from compose.adapters.lora import DEFAULT_PAIR_SCALE, ComposeLinear
from compose.adapters.runtime import use_selection
from compose.adapters.types import PAD_EXPERT_ID, ComposeSelection


def _layer():
    torch.manual_seed(47)
    layer = ComposeLinear(nn.Linear(16, 8, bias=False), rank=4, alpha=8.0)
    for expert_id in range(3):
        layer.add_expert(expert_id)
    # LoRAExpert zero-initializes lora_B; perturb B so the per-cardinality
    # scales act on nonzero deltas and the composition is observable.
    with torch.no_grad():
        for expert in layer.experts.values():
            expert.lora_B.weight.normal_(0.0, 0.05)
    return layer


def _selection(batch_size, *rows):
    ids = torch.full((batch_size, 3), PAD_EXPERT_ID, dtype=torch.long)
    gates = torch.zeros(batch_size, 3)
    for sample_index, (expert_ids, expert_gates) in enumerate(rows):
        for slot, (expert_id, gate) in enumerate(zip(expert_ids, expert_gates)):
            ids[sample_index, slot] = expert_id
            gates[sample_index, slot] = gate
    return ComposeSelection(ids, gates)


class PairScaleTest(unittest.TestCase):
    def setUp(self):
        self.layer = _layer()
        self.inputs = torch.randn(1, 16)
        with torch.no_grad():
            self.d0 = self.layer.experts["0"](self.inputs).detach()
            self.d1 = self.layer.experts["1"](self.inputs).detach()
            self.d2 = self.layer.experts["2"](self.inputs).detach()
        self.base = self.layer.base_layer(self.inputs).detach()

    def test_default_pair_scale_is_one_over_sqrt_two(self):
        self.assertAlmostEqual(DEFAULT_PAIR_SCALE, 1.0 / (2.0 ** 0.5))

    def test_single_scale_is_one(self):
        with use_selection(_selection(1, ([0, -1, -1], [1.0, 0.0, 0.0]))):
            output = self.layer(self.inputs).detach()
        self.assertTrue(torch.allclose(output, self.base + self.d0, atol=1e-5))

    def test_pair_scale_is_one_over_sqrt_two(self):
        with use_selection(_selection(1, ([0, 1, -1], [1.0, 1.0, 0.0]))):
            output = self.layer(self.inputs).detach()
        expected = self.base + (self.d0 + self.d1) / (2.0 ** 0.5)
        self.assertTrue(
            torch.allclose(output, expected, atol=1e-5),
            "pair composition must scale by 1/sqrt(2), got deviation {}".format(
                (output - expected).abs().max().item()
            ),
        )
        # A plain sum (no scaling) must differ.
        self.assertFalse(torch.allclose(output, self.base + self.d0 + self.d1))

    def test_triple_scale_is_one_over_sqrt_three(self):
        with use_selection(_selection(1, ([0, 1, 2], [1.0, 1.0, 1.0]))):
            output = self.layer(self.inputs).detach()
        expected = self.base + (self.d0 + self.d1 + self.d2) / (3.0 ** 0.5)
        self.assertTrue(
            torch.allclose(output, expected, atol=1e-5),
            "three-expert cluster-training selection must scale by 1/sqrt(3)",
        )

    def test_batched_rows_apply_per_sample_scale(self):
        rows = [
            ([0, -1, -1], [1.0, 0.0, 0.0]),  # single
            ([0, 1, -1], [1.0, 1.0, 0.0]),   # pair
            ([0, 1, 2], [1.0, 1.0, 1.0]),    # triple
        ]
        inputs = self.inputs.repeat(3, 1)
        with use_selection(_selection(3, *rows)):
            output = self.layer(inputs).detach()
        expected = torch.cat(
            [
                (self.base + self.d0),
                (self.base + (self.d0 + self.d1) / (2.0 ** 0.5)),
                (self.base + (self.d0 + self.d1 + self.d2) / (3.0 ** 0.5)),
            ],
            dim=0,
        )
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
