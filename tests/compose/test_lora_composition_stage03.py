import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from compose.adapters.lora import ComposeLinear
from compose.experts import ExpertMetadata, ExpertRegistry
from compose.lora import (AdapterBridge, CompositionRuntime, ExpertComposer,
                          OnlineMoments, RMSCompositionConfig, RMSStatistics,
                          StatisticKey, stable_hash)


def registry(count=3):
    value = ExpertRegistry()
    for expert_id in range(count):
        value.register(ExpertMetadata(expert_id=expert_id, adapter_name=str(expert_id)))
    return value


def model_and_bridge(dtype=torch.float32):
    base = nn.Linear(3, 2, bias=False, dtype=dtype)
    layer = ComposeLinear(base, rank=2, alpha=2, dropout=0)
    for expert_id in range(3):
        expert = layer.add_expert(expert_id)
        with torch.no_grad():
            expert.lora_A.weight.fill_(0.1 * (expert_id + 1))
            expert.lora_B.weight.fill_(0.2 * (expert_id + 1))
    model = nn.Sequential(layer)
    return model, layer, AdapterBridge(model, verify_ddp=False)


def stats_for(bridge, layer, inputs, config_hash="cfg"):
    stats = RMSStatistics({"calibration_split": "train_calibration", "checkpoint_hash": "ckpt",
                           "dataset_manifest_hash": "data", "composition_config_hash": config_hash})
    name = bridge.named_layers[0][0]
    base = layer.base_layer(inputs)
    for expert_id in (0, 1):
        delta = bridge.compute_expert_delta(layer, expert_id, inputs)
        stats.update(StatisticKey(expert_id, name, name, type(layer).__name__), delta, base + delta, base)
    return stats


class CompositionMathTest(unittest.TestCase):
    def test_base_single_and_direct_sum_regressions(self):
        model, layer, bridge = model_and_bridge()
        inputs = torch.randn(4, 3)
        reg = registry()
        composer = ExpertComposer(bridge)
        with CompositionRuntime(reg, bridge, composer, [], [], "base_only"):
            torch.testing.assert_close(model(inputs), layer.base_layer(inputs))
        layer.set_default_selection([0])
        stage02 = model(inputs).detach()
        with CompositionRuntime(reg, bridge, composer, [0], [], "single"):
            torch.testing.assert_close(model(inputs), stage02)
        layer.set_default_selection([0, 1], [1, 1], "none")
        old = model(inputs).detach()
        with CompositionRuntime(reg, bridge, composer, [1, 0], [], "direct_sum"):
            new = model(inputs)
        torch.testing.assert_close(new, old)
        with CompositionRuntime(reg, bridge, composer, [0, 1], [], "direct_sum"):
            ordered = model(inputs)
        torch.testing.assert_close(new, ordered)

    def test_zero_delta_and_no_implicit_pair_scale(self):
        model, layer, bridge = model_and_bridge()
        with torch.no_grad():
            layer.experts["0"].lora_B.weight.zero_()
        inputs = torch.randn(2, 3)
        expected = layer.base_layer(inputs) + layer.experts["1"](inputs)
        with CompositionRuntime(registry(), bridge, ExpertComposer(bridge), [0, 1], [], "direct_sum"):
            actual = model(inputs)
        torch.testing.assert_close(actual, expected)

    def test_reject_duplicate_too_many_missing_and_archived(self):
        model, _, bridge = model_and_bridge()
        composer, reg = ExpertComposer(bridge), registry()
        for ids, mode, error in [([0, 0], "direct_sum", ValueError), ([0, 1, 2], "direct_sum", ValueError), ([0, 9], "direct_sum", KeyError)]:
            with self.assertRaises(error):
                composer.validate(ids, mode)
        reg.archive(1)
        with self.assertRaises(ValueError):
            with CompositionRuntime(reg, bridge, composer, [0, 1], [], "direct_sum"):
                pass

    def test_gradient_isolation_and_restore(self):
        model, layer, bridge = model_and_bridge()
        for parameter in layer.base_layer.parameters():
            parameter.requires_grad_(False)
        initial = bridge.snapshot_runtime_state()
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with CompositionRuntime(registry(), bridge, ExpertComposer(bridge), [0, 1], [1], "direct_sum"):
                model(torch.randn(3, 3)).sum().backward()
                self.assertTrue(all(parameter.grad is None for parameter in layer.experts["0"].parameters()))
                self.assertTrue(any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in layer.experts["1"].parameters()))
                self.assertTrue(all(parameter.grad is None for parameter in layer.base_layer.parameters()))
                raise RuntimeError("boom")
        self.assertEqual(bridge.snapshot_runtime_state(), initial)

    def test_two_trainable_experts_are_explicitly_supported(self):
        model, layer, bridge = model_and_bridge()
        with CompositionRuntime(registry(), bridge, ExpertComposer(bridge), [0, 1], [0, 1], "direct_sum"):
            model(torch.randn(3, 3)).sum().backward()
        for expert_id in (0, 1):
            self.assertTrue(any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in layer.experts[str(expert_id)].parameters()))

    def test_rms_pair_swap_frozen_coefficients_and_bf16(self):
        model, layer, bridge = model_and_bridge(torch.bfloat16)
        inputs = torch.randn(4, 3, dtype=torch.bfloat16)
        stats = stats_for(bridge, layer, inputs)
        composer = ExpertComposer(bridge, stats, RMSCompositionConfig())
        with CompositionRuntime(registry(), bridge, composer, [0, 1], [], "rms_calibrated"):
            first = model(inputs)
        with CompositionRuntime(registry(), bridge, composer, [1, 0], [], "rms_calibrated"):
            second = model(inputs)
        torch.testing.assert_close(first, second)
        self.assertTrue(torch.isfinite(first).all())


class StatisticsTest(unittest.TestCase):
    def test_online_moments_and_epsilon(self):
        moments = OnlineMoments()
        moments.update(torch.tensor([1.0, 2.0]))
        moments.update(torch.tensor([3.0, 4.0]))
        self.assertAlmostEqual(moments.mean, 2.5)
        self.assertAlmostEqual(moments.variance, 1.25)
        self.assertAlmostEqual(moments.rms, math.sqrt(7.5))
        model, layer, bridge = model_and_bridge()
        inputs = torch.zeros(2, 3)
        stats = stats_for(bridge, layer, inputs)
        composer = ExpertComposer(bridge, stats)
        with CompositionRuntime(registry(), bridge, composer, [0, 1], [], "rms_calibrated"):
            self.assertTrue(torch.isfinite(model(inputs)).all())

    def test_atomic_checkpoint_roundtrip_and_hash_invalidation(self):
        model, layer, bridge = model_and_bridge()
        stats = stats_for(bridge, layer, torch.randn(3, 3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rms.json"
            stats.save_json(path)
            restored = RMSStatistics.load_json(path, stats.provenance, registry())
            self.assertEqual(restored.summary(), stats.summary())
            changed = dict(stats.provenance, composition_config_hash="changed")
            with self.assertRaisesRegex(ValueError, "invalidated"):
                RMSStatistics.load_json(path, changed)
            archived = registry()
            archived.archive(0)
            with self.assertRaisesRegex(ValueError, "archived expert 0"):
                RMSStatistics.load_json(path, stats.provenance, archived)
            missing = registry(0)
            with self.assertRaisesRegex(KeyError, "not registered"):
                RMSStatistics.load_json(path, stats.provenance, missing)
        self.assertEqual(stable_hash({"b": 2, "a": 1}), stable_hash({"a": 1, "b": 2}))
        with self.assertRaisesRegex(ValueError, "test"):
            RMSStatistics(dict(stats.provenance, calibration_split="test"))

    def test_ddp_all_reduce_reconstructs_global_moments(self):
        moments = OnlineMoments()
        moments.update(torch.tensor([1.0, 3.0]))
        def double(tensor):
            tensor.mul_(2)
        with mock.patch("torch.distributed.is_available", return_value=True), \
             mock.patch("torch.distributed.is_initialized", return_value=True), \
             mock.patch("torch.distributed.all_reduce", side_effect=double):
            moments.all_reduce_(torch.device("cpu"))
        self.assertEqual(moments.count, 4)
        self.assertAlmostEqual(moments.mean, 2.0)
        self.assertAlmostEqual(moments.variance, 1.0)

        model, layer, bridge = model_and_bridge()
        stats = stats_for(bridge, layer, torch.randn(2, 3))
        with mock.patch("torch.distributed.is_available", return_value=True), \
             mock.patch("torch.distributed.is_initialized", return_value=True), \
             mock.patch("torch.distributed.all_reduce", side_effect=double):
            stats.all_reduce_(torch.device("cpu"))
        self.assertTrue(all(entry["sample_count"] == 4 for entry in stats.entries.values()))


if __name__ == "__main__":
    unittest.main()
