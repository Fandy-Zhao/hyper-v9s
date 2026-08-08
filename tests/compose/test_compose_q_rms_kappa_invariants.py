"""RMS runtime-kappa invariant checks (pre-formal validation §5.1/§5.2).

Two contracts, both BLOCKING for the formal experiment:

1. ``build_kappa_calibration`` synthetic invariant: the reference is the
   arithmetic mean of raw RMS over ALL active experts in the layer, and
   ``kappa_k_l = clip(ref / (raw + eps), kappa_min, kappa_max)``. With
   raw = [1, 3] the reference is 2.0, kappa_A = 2.0 and kappa_B = 2/3
   (no clipping); with raw = [1, 2, 4] the reference is 7/3.
2. Runtime forward: a calibrated ComposeLinear must reproduce exactly
   ``base + gate * kappa * scale * delta`` per expert, with the 1/sqrt(2)
   pair-composition scale, so the persisted kappa map is not a
   report-only diagnostic but part of the composition.
"""

import unittest

import torch
from torch import nn

from compose.adapters.lora import ComposeLinear
from compose.adapters.runtime import use_selection
from compose.adapters.types import PAD_EXPERT_ID, ComposeSelection
from compose.lora.rms import RMSStatistics, StatisticKey, build_kappa_calibration


def _stats_with_raw(raw_values):
    """RMSStatistics whose per-expert delta RMS equals raw_values exactly."""
    stats = RMSStatistics(
        {
            "calibration_split": "validation",
            "checkpoint_hash": "test",
            "composition_config_hash": "test",
            "dataset_manifest_hash": "test",
        }
    )
    for expert_id, raw in enumerate(raw_values):
        delta = torch.full((1, 1), float(raw))
        key = StatisticKey(
            expert_id=expert_id,
            layer_name="layer_0",
            module_name="layer_0",
            target_module_type="ComposeLinear",
        )
        stats.update(key, delta, delta)
    return stats


class KappaCalibrationInvariantTest(unittest.TestCase):
    def test_raw_1_3_reference_is_mean_kappa_2_and_two_thirds(self):
        calibration = build_kappa_calibration(_stats_with_raw([1.0, 3.0]), [0, 1])
        layer = calibration["layer_0"]
        self.assertAlmostEqual(layer["0"], 2.0, places=6)
        self.assertAlmostEqual(layer["1"], 2.0 / 3.0, places=6)

    def test_raw_1_2_4_reference_7_over_3(self):
        calibration = build_kappa_calibration(_stats_with_raw([1.0, 2.0, 4.0]), [0, 1, 2])
        layer = calibration["layer_0"]
        self.assertAlmostEqual(layer["0"], 7.0 / 3.0, places=6)
        self.assertAlmostEqual(layer["1"], 7.0 / 6.0, places=6)
        self.assertAlmostEqual(layer["2"], 7.0 / 12.0, places=6)

    def test_extreme_ratios_are_clipped(self):
        calibration = build_kappa_calibration(_stats_with_raw([1.0, 8.0]), [0, 1])
        layer = calibration["layer_0"]
        self.assertEqual(layer["0"], 4.0)  # 4.5 -> kappa_max
        self.assertAlmostEqual(layer["1"], 0.5625, places=6)  # 9/16, no clip

    def test_single_expert_layer_calibrates_to_one(self):
        calibration = build_kappa_calibration(_stats_with_raw([2.5]), [0])
        self.assertAlmostEqual(calibration["layer_0"]["0"], 1.0, places=6)

    def test_test_split_is_rejected_for_calibration(self):
        from compose.lora.rms import ComposeRMSConfig

        with self.assertRaises(ValueError):
            ComposeRMSConfig(calibration_split="test")


class KappaRuntimeForwardTest(unittest.TestCase):
    def _build_layer(self):
        torch.manual_seed(1)
        layer = ComposeLinear(nn.Linear(8, 8, bias=False), rank=4, alpha=8.0)
        layer.add_expert(0)
        layer.add_expert(1)
        with torch.no_grad():
            for key in ("0", "1"):
                layer.experts[key].lora_A.weight.normal_(0.0, 0.1)
                layer.experts[key].lora_B.weight.normal_(0.0, 0.1)
        layer.set_expert_calibration({0: 2.0, 1: 2.0 / 3.0})
        return layer

    def test_pair_forward_matches_kappa_scaled_composition(self):
        layer = self._build_layer()
        x = torch.randn(4, 8)
        ids = torch.full((4, 3), PAD_EXPERT_ID, dtype=torch.long)
        gates = torch.zeros(4, 3)
        ids[:, 0] = 0
        ids[:, 1] = 1
        gates[:, 0] = 1.0
        gates[:, 1] = 1.0
        selection = ComposeSelection(ids, gates)
        with use_selection(selection):
            out = layer(x)
        pair_scale = 1.0 / 2.0 ** 0.5
        delta_0 = layer.experts["0"](x)
        delta_1 = layer.experts["1"](x)
        expected = layer.base_layer(x) + pair_scale * (
            2.0 * delta_0 + (2.0 / 3.0) * delta_1
        )
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_single_forward_scales_by_kappa_only(self):
        layer = self._build_layer()
        x = torch.randn(3, 8)
        ids = torch.full((3, 3), PAD_EXPERT_ID, dtype=torch.long)
        gates = torch.zeros(3, 3)
        ids[:, 0] = 0
        gates[:, 0] = 1.0
        selection = ComposeSelection(ids, gates)
        with use_selection(selection):
            out = layer(x)
        expected = layer.base_layer(x) + 2.0 * layer.experts["0"](x)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_unlisted_expert_keeps_kappa_one(self):
        layer = self._build_layer()
        layer.set_expert_calibration({0: 2.0})  # expert 1 unlisted -> 1.0
        x = torch.randn(2, 8)
        ids = torch.full((2, 3), PAD_EXPERT_ID, dtype=torch.long)
        gates = torch.zeros(2, 3)
        ids[:, 0] = 1
        gates[:, 0] = 1.0
        selection = ComposeSelection(ids, gates)
        with use_selection(selection):
            out = layer(x)
        expected = layer.base_layer(x) + 1.0 * layer.experts["1"](x)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
