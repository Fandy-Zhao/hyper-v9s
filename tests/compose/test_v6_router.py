"""V6 Stage E3: dual-modal query, expert keys and the dual-mode router."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from compose.router.v6_router import (
    V6QueryEncoder,
    V6Router,
    load_v6_router_checkpoint,
    save_v6_router_checkpoint,
)


def _router(seed: int = 42, **overrides):
    encoder = V6QueryEncoder(visual_dim=4, text_dim=5, query_dim=8)
    router = V6Router(encoder, top_m=3, seed=seed, **overrides)
    return encoder, router


def _populated_router(seed: int = 42):
    encoder, router = _router(seed)
    router.add_expert(0, creation_task=0, checkpoint_sha256="a" * 64)
    router.add_expert(1, creation_task=1, checkpoint_sha256="b" * 64)
    router.add_expert(2, creation_task=2, checkpoint_sha256="c" * 64)
    return encoder, router


class V6QueryEncoderTest(unittest.TestCase):
    def test_output_is_normalized_and_dim_128_style(self):
        encoder, _ = _router()
        z_v = torch.randn(3, 4)
        z_s = torch.randn(3, 5)
        query = encoder(z_v, z_s)
        self.assertEqual(query.shape, (3, 8))
        self.assertTrue(torch.allclose(query.norm(dim=-1), torch.ones(3), atol=1e-5))

    def test_layer_norm_is_applied_per_modality(self):
        encoder, _ = _router()
        z_v = torch.full((1, 4), 7.0)  # constant input; layernorm -> ~0
        z_s = torch.randn(1, 5)
        query_v = encoder(z_v, z_s)
        z_v_shifted = z_v + 100.0
        query_shifted = encoder(z_v_shifted, z_s)
        torch.testing.assert_close(query_v, query_shifted, atol=1e-5, rtol=1e-5)

    def test_batch_mismatch_rejected(self):
        encoder, _ = _router()
        with self.assertRaisesRegex(ValueError, "batch size"):
            encoder(torch.randn(3, 4), torch.randn(5, 5))

    def test_forward_is_autograd_compatible(self):
        encoder, _ = _router()
        z_v = torch.randn(2, 4, requires_grad=True)
        z_s = torch.randn(2, 5)
        query = encoder(z_v, z_s)
        query.sum().backward()
        self.assertIsNotNone(z_v.grad)


class V6RouterTest(unittest.TestCase):
    def test_add_expert_registers_key_and_bias(self):
        _, router = _router()
        router.add_expert(0, creation_task=0, checkpoint_sha256="a" * 64, bias=0.3)
        self.assertEqual(router.expert_ids, (0,))
        self.assertAlmostEqual(float(router.expert_bias["0"]), 0.3)
        key = router.key_store.normalized([0])
        self.assertTrue(torch.allclose(key.norm(dim=-1), torch.ones(1), atol=1e-5))

    def test_provided_key_is_used_verbatim_normalized(self):
        _, router = _router()
        raw = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        router.add_expert(7, creation_task=0, checkpoint_sha256="d" * 64, key=raw)
        stored = router.key_store.normalized([7])[0]
        torch.testing.assert_close(stored, raw)

    def test_duplicate_expert_rejected(self):
        _, router = _router()
        router.add_expert(0, creation_task=0, checkpoint_sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "already present"):
            router.add_expert(0, creation_task=0, checkpoint_sha256="a" * 64)

    def test_training_retrieval_top_m_and_full_pool_when_k_small(self):
        _, router = _populated_router()
        query = torch.randn(2, 8)
        # K=3, top_m=3 -> whole pool.
        result = router.run(query, "training_retrieval", router.expert_ids)
        self.assertEqual(set(result.expert_ids), {0, 1, 2})  # sorted by score
        self.assertEqual(result.probabilities.shape, (2, 3))
        self.assertEqual(result.all_visible_ids, (0, 1, 2))
        # K=3, top_m=2 -> top-2.
        router.top_m = 2
        result = router.run(query, "training_retrieval", router.expert_ids)
        self.assertEqual(len(result.expert_ids), 2)
        # K=3, top_m=8 -> whole pool.
        router.top_m = 8
        result = router.run(query, "training_retrieval", router.expert_ids)
        self.assertEqual(len(result.expert_ids), 3)

    def test_training_retrieval_empty_pool(self):
        _, router = _router()
        query = torch.randn(2, 8)
        result = router.run(query, "training_retrieval", [])
        self.assertEqual(result.expert_ids, ())
        self.assertEqual(result.probabilities.shape, (2, 0))

    def test_inference_selection_empty_single_pair(self):
        _, router = _populated_router()
        # Align keys so expert 0 is very close to the query direction.
        with torch.no_grad():
            router.key_store.keys["0"].copy_(
                torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
            )
            router.key_store.keys["1"].copy_(
                torch.tensor([0.7, 0.7, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
            )
            router.key_store.keys["2"].copy_(
                torch.tensor([0, 0, 0, 1.0, 0, 0, 0, 0], dtype=torch.float32)
            )
        router.set_thresholds(tau_none=0.5, tau_second=0.6)
        query_none = torch.tensor([[-0.5, 0, 0, -1.0, 0, 0, 0, 0]], dtype=torch.float32)
        result = router.run(query_none, "inference_selection", router.expert_ids)
        self.assertEqual(result.sets, ((),))  # all below tau_none

        # key0 = [1,0,...], key1 = [0.7,0.7,...]: q=[0.9,-0.9] gives cos 0.9 vs 0
        query_single = torch.tensor([[0.9, -0.9, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
        result = router.run(query_single, "inference_selection", router.expert_ids)
        self.assertEqual(len(result.sets[0]), 1)  # second expert below tau_second

        # q=[0.7,0.7]: cos 0.7 vs 0.98 -> both above thresholds -> pair
        query_pair = torch.tensor([[0.7, 0.7, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
        result = router.run(query_pair, "inference_selection", router.expert_ids)
        self.assertEqual(len(result.sets[0]), 2)  # top two pass

    def test_selection_respects_visible_ids(self):
        _, router = _populated_router()
        query = torch.randn(1, 8)
        result = router.run(query, "inference_selection", [0])
        self.assertLessEqual(len(result.sets[0]), 1)
        self.assertEqual(result.expert_ids, (0,))

    def test_selection_empty_pool(self):
        _, router = _router()
        query = torch.randn(1, 8)
        result = router.run(query, "inference_selection", [])
        self.assertEqual(result.sets, ((),))

    def test_unknown_mode_rejected(self):
        _, router = _router()
        with self.assertRaisesRegex(ValueError, "mode"):
            router.run(torch.randn(1, 8), "bogus", [])

    def test_probability_formula_matches_task_book(self):
        # prob = sigmoid((cos - bias) / temperature)
        encoder, router = _router()
        router.add_expert(0, creation_task=0, checkpoint_sha256="a" * 64, bias=0.2)
        with torch.no_grad():
            router.key_store.keys["0"].copy_(
                torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
            )
        query = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
        probabilities, ids = router._scores(query, [0])
        expected = torch.sigmoid(torch.tensor((1.0 - 0.2) / router.temperature))
        torch.testing.assert_close(probabilities[0, 0], expected)

    def test_checkpoint_round_trip_binds_all_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "router.json")
            _, router = _populated_router()
            router.set_thresholds(tau_none=0.55, tau_second=0.6)
            save_v6_router_checkpoint(
                path, router, pool_version=3, config_hash="cfg123",
                extra={"note": "after task 2"},
            )
            encoder2 = V6QueryEncoder(visual_dim=4, text_dim=5, query_dim=8)
            router2 = V6Router(encoder2, seed=99)  # different seed/state
            extra = load_v6_router_checkpoint(path, router2)
            self.assertEqual(extra, {"note": "after task 2"})
            self.assertEqual(router2.expert_ids, (0, 1, 2))
            self.assertEqual(router2.tau_none, 0.55)
            self.assertEqual(router2.tau_second, 0.6)
            self.assertEqual(router2.pool_version, 3)
            self.assertEqual(router2.config_hash, "cfg123")
            self.assertEqual(router2.key_store.expert_ids, (0, 1, 2))
            # Keys restored exactly.
            torch.testing.assert_close(
                router2.key_store.normalized([0]), router.key_store.normalized([0])
            )
            # Metadata restored (temporal visibility works).
            self.assertEqual(
                router2.key_store.visible_expert_ids(task_id=1, historical_only=True),
                (0,),
            )

    def test_pool_version_validation(self):
        _, router = _router()
        with self.assertRaisesRegex(ValueError, "pool_version"):
            router.validate_pool_version(1)
        router.pool_version = 2
        router.validate_pool_version(2)
        with self.assertRaisesRegex(ValueError, "does not match"):
            router.validate_pool_version(3)

    def test_temperature_bounds(self):
        _, router = _router()
        with torch.no_grad():
            router.log_temperature.fill_(100.0)
        self.assertLessEqual(router.temperature, 20.0)
        with torch.no_grad():
            router.log_temperature.fill_(-100.0)
        self.assertGreaterEqual(router.temperature, 0.05)


if __name__ == "__main__":
    unittest.main()
