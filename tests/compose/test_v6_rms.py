"""V6 Stage E9: RMS statistics collection, calibration report, freshness."""

import tempfile
import unittest

import torch

from compose.adapters import ExpertManager, inject_compose_adapters
from compose.config import ComposeAdapterConfig
from compose.experts import ExpertPool
from compose.lora.statistics import RMSStatistics
from compose.lora.v6_rms import (
    PairDeltaMoments,
    V6RMSConfig,
    build_rms_provenance,
    compute_v6_expert_rms,
    rms_report,
    validate_rms_freshness,
)
from test_injection import TinyModel


def _decoder_output(model, inputs):
    hidden = inputs
    layer = model.model.layers[0]
    hidden = layer.self_attn.o_proj(
        layer.self_attn.q_proj(hidden)
        + layer.self_attn.k_proj(hidden)
        + layer.self_attn.v_proj(hidden)
    )
    return layer.mlp.down_proj(
        layer.mlp.gate_proj(hidden) + layer.mlp.up_proj(hidden)
    )


def _model_and_pool(seed: int = 7):
    torch.manual_seed(seed)
    model = TinyModel(layer_count=2)
    inject_compose_adapters(model, ComposeAdapterConfig(rank=1, alpha=2))
    return model, ExpertPool(ExpertManager(model))


def _provenance(checkpoint_hash="h1"):
    return build_rms_provenance(
        calibration_split="validation",
        checkpoint_hash=checkpoint_hash,
        dataset_manifest_hash="data-hash",
        composition_config_hash="cfg-hash",
    )


def _dataloader(batches=2, rows=3):
    for _ in range(batches):
        yield torch.randn(rows, 4, 3)


class ComputeRMSTest(unittest.TestCase):
    def test_collects_per_layer_delta_rms(self):
        model, pool = _model_and_pool()
        pool.register(0)
        for layer in pool.manager.layers.values():
            with torch.no_grad():
                layer.experts["0"].lora_A.weight.fill_(0.5)
                layer.experts["0"].lora_B.weight.fill_(0.25)
        stats, pair = compute_v6_expert_rms(
            model, [0], _dataloader(), _provenance(),
            V6RMSConfig(), prepare_batch=lambda batch: batch, device="cpu",
            forward_fn=lambda inputs: _decoder_output(model, inputs),
        )
        entries = stats.entries
        self.assertGreaterEqual(len(entries), 7)  # 7 ComposeLinear layers
        for entry in entries.values():
            self.assertGreater(entry["delta"].rms, 0.0)
            self.assertEqual(entry["key"]["expert_id"], 0)
        self.assertEqual(pair, {})  # single expert: no pair diagnostics

    def test_two_experts_collect_pair_cross_terms(self):
        model, pool = _model_and_pool()
        pool.register(0)
        pool.register(1)
        for layer in pool.manager.layers.values():
            with torch.no_grad():
                layer.experts["0"].lora_A.weight.fill_(0.5)
                layer.experts["0"].lora_B.weight.fill_(0.25)
                layer.experts["1"].lora_A.weight.fill_(-0.5)
                layer.experts["1"].lora_B.weight.fill_(0.25)
        stats, pair = compute_v6_expert_rms(
            model, [0, 1], _dataloader(), _provenance(),
            V6RMSConfig(), prepare_batch=lambda batch: batch, device="cpu",
            forward_fn=lambda inputs: _decoder_output(model, inputs),
        )
        self.assertEqual(len(pair), 1)  # first layer
        moments = next(iter(pair.values()))
        self.assertGreater(moments.count, 0)
        self.assertIsNotNone(moments.cosine())
        self.assertIsNotNone(moments.cancellation())

    def test_unregistered_expert_rejected(self):
        model, pool = _model_and_pool()
        pool.register(0)
        with self.assertRaisesRegex(KeyError, "not registered"):
            compute_v6_expert_rms(
                model, [5], _dataloader(), _provenance(),
                V6RMSConfig(), prepare_batch=lambda batch: batch, device="cpu",
                forward_fn=lambda inputs: _decoder_output(model, inputs),
            )


class RMSReportTest(unittest.TestCase):
    def test_report_fields(self):
        model, pool = _model_and_pool()
        pool.register(0)
        for layer in pool.manager.layers.values():
            with torch.no_grad():
                layer.experts["0"].lora_A.weight.fill_(0.5)
                layer.experts["0"].lora_B.weight.fill_(0.25)
        stats, pair = compute_v6_expert_rms(
            model, [0], _dataloader(batches=1), _provenance(),
            V6RMSConfig(), prepare_batch=lambda batch: batch, device="cpu",
            forward_fn=lambda inputs: _decoder_output(model, inputs),
        )
        report = rms_report(stats, [0], pair, V6RMSConfig())
        self.assertIn("per_expert", report)
        self.assertIn("clip_ratio", report)
        self.assertIn("dominance_ratio", report)
        self.assertIn("pair", report)
        layer_name = list(report["per_expert"]["0"])[0]
        layer_entry = report["per_expert"]["0"][layer_name]
        self.assertGreater(layer_entry["raw_rms"], 0.0)
        self.assertGreater(layer_entry["calibrated_rms"], 0.0)
        self.assertGreaterEqual(layer_entry["kappa"], V6RMSConfig().kappa_min)

    def test_pair_diagnostics_values(self):
        moments = PairDeltaMoments()
        delta_a = torch.randn(3, 4)
        delta_b = torch.randn(3, 4)
        moments.update(delta_a, delta_b)
        moments.update(delta_a, delta_b)
        self.assertTrue(-1.0 <= moments.cosine() <= 1.0)
        self.assertGreaterEqual(moments.cancellation(), 0.0)

    def test_identical_deltas_show_no_cancellation(self):
        moments = PairDeltaMoments()
        delta = torch.ones(2, 3)
        moments.update(delta, delta)
        self.assertAlmostEqual(moments.cosine(), 1.0, places=6)
        self.assertAlmostEqual(moments.cancellation(), 2.0, places=6)


class FreshnessTest(unittest.TestCase):
    def test_freshness_bound_to_checkpoint_hash(self):
        model, pool = _model_and_pool()
        pool.register(0)
        for layer in pool.manager.layers.values():
            with torch.no_grad():
                layer.experts["0"].lora_A.weight.fill_(0.5)
                layer.experts["0"].lora_B.weight.fill_(0.25)
        stats, _ = compute_v6_expert_rms(
            model, [0], _dataloader(batches=1), _provenance("hash-A"),
            V6RMSConfig(), prepare_batch=lambda batch: batch, device="cpu",
            forward_fn=lambda inputs: _decoder_output(model, inputs),
        )
        self.assertTrue(validate_rms_freshness(stats, "hash-A"))
        self.assertFalse(validate_rms_freshness(stats, "hash-B"))

    def test_save_load_enforces_provenance(self):
        import tempfile
        from pathlib import Path

        model, pool = _model_and_pool()
        pool.register(0)
        for layer in pool.manager.layers.values():
            with torch.no_grad():
                layer.experts["0"].lora_A.weight.fill_(0.5)
                layer.experts["0"].lora_B.weight.fill_(0.25)
        stats, _ = compute_v6_expert_rms(
            model, [0], _dataloader(batches=1), _provenance("hash-A"),
            V6RMSConfig(), prepare_batch=lambda batch: batch, device="cpu",
            forward_fn=lambda inputs: _decoder_output(model, inputs),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "rms.json")
            stats.save_json(path)
            loaded = RMSStatistics.load_json(
                path, expected_provenance=_provenance("hash-A")
            )
            self.assertTrue(validate_rms_freshness(loaded, "hash-A"))
            with self.assertRaises(ValueError):
                RMSStatistics.load_json(
                    path, expected_provenance=_provenance("hash-B")
                )


class RMSConfigTest(unittest.TestCase):
    def test_defaults(self):
        config = V6RMSConfig()
        self.assertAlmostEqual(config.epsilon, 1.0e-8)
        self.assertAlmostEqual(config.kappa_min, 0.25)
        self.assertAlmostEqual(config.kappa_max, 4.0)

    def test_test_split_rejected(self):
        with self.assertRaisesRegex(ValueError, "test"):
            V6RMSConfig(calibration_split="test")

    def test_kappa_bounds(self):
        with self.assertRaisesRegex(ValueError, "kappa"):
            V6RMSConfig(kappa_min=2.0, kappa_max=1.0)


if __name__ == "__main__":
    unittest.main()
