"""Test K (spec §27): RMS kappa coefficients are ACTUALLY applied inside
``ComposeLinear.forward`` (via ``set_expert_calibration``), not merely
reported; clearing restores the uncalibrated output."""

import unittest

import torch
from torch import nn

from compose.adapters.lora import ComposeLinear
from compose.adapters.runtime import use_selection
from compose.adapters.types import PAD_EXPERT_ID, ComposeSelection


def _layer():
    torch.manual_seed(43)
    layer = ComposeLinear(nn.Linear(16, 8, bias=False), rank=4, alpha=8.0)
    for expert_id in range(2):
        layer.add_expert(expert_id)
    # LoRAExpert zero-initializes lora_B; perturb B so the kappa-scaling of
    # nonzero deltas is observable (a zero delta would make every kappa
    # produce the same output and the test would vacuously pass).
    with torch.no_grad():
        for expert in layer.experts.values():
            expert.lora_B.weight.normal_(0.0, 0.05)
    return layer


def _selection(batch_size, expert_ids, gates):
    ids = torch.full((batch_size, 3), PAD_EXPERT_ID, dtype=torch.long)
    gate_tensor = torch.zeros(batch_size, 3)
    for index, (expert_id, gate) in enumerate(zip(expert_ids, gates)):
        ids[0, index] = expert_id
        gate_tensor[0, index] = gate
    return ComposeSelection(ids, gate_tensor)


def _raw_deltas(layer, inputs, expert_ids):
    deltas = []
    for expert_id in expert_ids:
        with torch.no_grad():
            deltas.append(layer.experts[str(expert_id)](inputs).detach())
    return deltas


class RmsAppliedTest(unittest.TestCase):
    def setUp(self):
        self.layer = _layer()
        self.inputs = torch.randn(1, 16)

    def test_kappa_changes_the_output(self):
        base = self.layer.base_layer(self.inputs).detach()
        d0, d1 = _raw_deltas(self.layer, self.inputs, [0, 1])
        pair_scale = 1.0 / (2.0 ** 0.5)
        with use_selection(_selection(1, [0, 1], [1.0, 1.0])):
            uncalibrated = self.layer(self.inputs).detach()
        expected_uncalibrated = base + (d0 + d1) * pair_scale
        self.assertTrue(
            torch.allclose(uncalibrated, expected_uncalibrated, atol=1e-5),
            "pair output without kappa must match manual composition",
        )
        # Apply runtime kappa: expert 0 boosted, expert 1 suppressed.
        self.layer.set_expert_calibration({0: 2.0, 1: 0.5})
        with use_selection(_selection(1, [0, 1], [1.0, 1.0])):
            calibrated = self.layer(self.inputs).detach()
        expected_calibrated = base + (d0 * 2.0 + d1 * 0.5) * pair_scale
        self.assertTrue(
            torch.allclose(calibrated, expected_calibrated, atol=1e-5),
            "kappa must scale the expert deltas inside forward",
        )
        self.assertFalse(torch.allclose(calibrated, uncalibrated))

    def test_clear_calibration_restores_output(self):
        with use_selection(_selection(1, [0, -1], [1.0, 0.0])):
            plain = self.layer(self.inputs).detach()
        self.layer.set_expert_calibration({0: 3.0})
        with use_selection(_selection(1, [0, -1], [1.0, 0.0])):
            boosted = self.layer(self.inputs).detach()
        self.assertFalse(torch.allclose(plain, boosted))
        self.layer.clear_expert_calibration()
        with use_selection(_selection(1, [0, -1], [1.0, 0.0])):
            restored = self.layer(self.inputs).detach()
        self.assertTrue(torch.allclose(plain, restored, atol=1e-6))

    def test_unlisted_experts_keep_kappa_one(self):
        base = self.layer.base_layer(self.inputs).detach()
        d0, = _raw_deltas(self.layer, self.inputs, [0])
        self.layer.set_expert_calibration({1: 0.25})  # expert 0 unlisted
        with use_selection(_selection(1, [0, -1], [1.0, 0.0])):
            output = self.layer(self.inputs).detach()
        self.assertTrue(torch.allclose(output, base + d0, atol=1e-5))

    def test_calibration_exposed_for_persistence(self):
        self.layer.set_expert_calibration({0: 1.5, 1: 0.8})
        state = self.layer.expert_calibration()
        self.assertEqual(state["kappa"], {0: 1.5, 1: 0.8})
        self.assertAlmostEqual(state["pair_scale"], 1.0 / (2.0 ** 0.5))


if __name__ == "__main__":
    unittest.main()
