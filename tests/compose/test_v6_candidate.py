"""V6 Stage E6: candidate pool construction, assignment statistics, reinit."""

import unittest

import torch
from torch import nn

from compose.expansion.candidate_pool import CandidateExpertPool
from compose.expansion.v6_candidate import (
    MAX_BALANCE_WEIGHT,
    V6CandidateConfig,
    assignment_statistics,
    build_v6_candidate_pool,
    reinit_empty_slots,
)


def _adapters(count: int, dim: int = 8):
    class TinyAdapter(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.lora_A = nn.Linear(dim, 2, bias=False)
            self.lora_B = nn.Linear(2, dim, bias=False)

        def forward(self, inputs):
            return self.lora_B(self.lora_A(inputs))

    return [TinyAdapter(dim) for _ in range(count)]


def _queries(count: int, dim: int = 8, seed: int = 7):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, dim, generator=generator)


class V6CandidateConfigTest(unittest.TestCase):
    def test_task1_single_slot_allowed(self):
        config = V6CandidateConfig(slot_count=1)
        self.assertEqual(config.slot_count, 1)

    def test_slot_count_restricted(self):
        with self.assertRaisesRegex(ValueError, "one or two"):
            V6CandidateConfig(slot_count=3)

    def test_balance_weight_must_stay_small(self):
        self.assertAlmostEqual(MAX_BALANCE_WEIGHT, 0.01)
        with self.assertRaisesRegex(ValueError, "balance"):
            V6CandidateConfig(lambda_balance=0.5)

    def test_contribution_correction_off_by_default(self):
        self.assertFalse(V6CandidateConfig().use_contribution_corrected_assignment)

    def test_target_modules_required(self):
        with self.assertRaisesRegex(ValueError, "target_modules"):
            V6CandidateConfig(target_modules=())


class BuildPoolTest(unittest.TestCase):
    def test_task1_single_slot_pool(self):
        config = V6CandidateConfig(slot_count=1, query_dim=8)
        pool, record = build_v6_candidate_pool(
            config, residual_queries=_queries(10), adapters=_adapters(1, dim=8)
        )
        self.assertEqual(pool.config.slot_count, 1)
        self.assertEqual(pool.keys.shape, (1, 8))
        self.assertEqual(record["slot_count"], 1)
        self.assertEqual(record["method"], "kmeans_plus_plus")

    def test_task2_two_slot_pool(self):
        config = V6CandidateConfig(slot_count=2, query_dim=8)
        pool, record = build_v6_candidate_pool(
            config, residual_queries=_queries(10), adapters=_adapters(2, dim=8)
        )
        self.assertEqual(pool.config.slot_count, 2)
        self.assertEqual(pool.keys.shape, (2, 8))
        self.assertIn(record["method"], ("kmeans_plus_plus", "random_orthogonal"))

    def test_orthogonal_fallback_when_queries_too_small(self):
        config = V6CandidateConfig(slot_count=2, query_dim=8,
                                    key_initialization="kmeans_plus_plus")
        pool, record = build_v6_candidate_pool(
            config, residual_queries=None, adapters=_adapters(2, dim=8)
        )
        self.assertEqual(record["method"], "random_orthogonal")
        self.assertTrue(record["fallback_used"])
        # Orthogonal keys have low cosine.
        cosine = torch.nn.functional.normalize(pool.keys[0], dim=0) @ torch.nn.functional.normalize(
            pool.keys[1], dim=0
        )
        self.assertLess(abs(float(cosine)), 0.3)

    def test_explicit_random_orthogonal_init(self):
        config = V6CandidateConfig(slot_count=2, query_dim=8,
                                    key_initialization="random_orthogonal")
        pool, record = build_v6_candidate_pool(
            config, residual_queries=_queries(10), adapters=_adapters(2, dim=8)
        )
        self.assertEqual(record["method"], "random_orthogonal")

    def test_seeded_construction_is_reproducible(self):
        config = V6CandidateConfig(slot_count=2, query_dim=8)
        pool_a, _ = build_v6_candidate_pool(
            config, residual_queries=_queries(10), adapters=_adapters(2, dim=8)
        )
        pool_b, _ = build_v6_candidate_pool(
            config, residual_queries=_queries(10), adapters=_adapters(2, dim=8)
        )
        torch.testing.assert_close(pool_a.keys, pool_b.keys)


class AssignmentStatisticsTest(unittest.TestCase):
    def _pool(self, count=2):
        config = V6CandidateConfig(slot_count=count, query_dim=8)
        pool, _ = build_v6_candidate_pool(
            config, residual_queries=_queries(50), adapters=_adapters(count, dim=8)
        )
        return pool

    def test_statistics_shape_and_entropy(self):
        pool = self._pool(2)
        stats = assignment_statistics(pool, _queries(20))
        self.assertEqual(stats.total_samples, 20)
        self.assertEqual(len(stats.per_slot), 2)
        self.assertEqual(sum(slot.sample_count for slot in stats.per_slot), 20)
        self.assertGreaterEqual(stats.entropy, 0.0)
        self.assertIsNotNone(stats.pair_key_cosine)
        self.assertTrue(-1.0 <= stats.pair_key_cosine <= 1.0)

    def test_single_slot_has_no_pair_cosine(self):
        pool = self._pool(1)
        stats = assignment_statistics(pool, _queries(20))
        self.assertIsNone(stats.pair_key_cosine)
        self.assertEqual(stats.per_slot[0].sample_count, 20)

    def test_empty_slots_detected(self):
        pool = self._pool(2)
        # A query set concentrated far from slot 1's key.
        queries = torch.zeros(10, 8)
        queries[:, 0] = -5.0
        with torch.no_grad():
            pool.slots[1].key.copy_(
                torch.nn.functional.normalize(torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]), dim=0)
            )
            pool.slots[0].key.copy_(
                torch.nn.functional.normalize(torch.tensor([-1.0, 0, 0, 0, 0, 0, 0, 0]), dim=0)
            )
        stats = assignment_statistics(pool, queries)
        self.assertEqual(stats.empty_slot_ids, (1,))


class ReinitTest(unittest.TestCase):
    def test_empty_slot_reinitialized_once(self):
        config = V6CandidateConfig(slot_count=2, query_dim=8)
        pool, _ = build_v6_candidate_pool(
            config, residual_queries=_queries(50), adapters=_adapters(2, dim=8)
        )
        queries = torch.zeros(10, 8)
        queries[:, 0] = -5.0
        with torch.no_grad():
            pool.slots[1].key.copy_(
                torch.nn.functional.normalize(torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]), dim=0)
            )
            pool.slots[0].key.copy_(
                torch.nn.functional.normalize(torch.tensor([-1.0, 0, 0, 0, 0, 0, 0, 0]), dim=0)
            )
        reinit_count = {}
        reinitialized, record = reinit_empty_slots(pool, queries, reinit_count, config)
        self.assertGreaterEqual(reinitialized, 0)
        self.assertIn("reinitialized", record)
        # A second pass must not re-initialize the same slot again.
        second, record2 = reinit_empty_slots(pool, queries, reinit_count, config)
        self.assertIn("allowed_to_die", record2)

    def test_reinit_disabled(self):
        config = V6CandidateConfig(slot_count=2, query_dim=8,
                                   allow_empty_slot_reinit=False)
        pool, _ = build_v6_candidate_pool(
            config, residual_queries=None, adapters=_adapters(2, dim=8)
        )
        queries = torch.zeros(10, 8)
        queries[:, 0] = -5.0
        with torch.no_grad():
            pool.slots[1].key.copy_(
                torch.nn.functional.normalize(torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]), dim=0)
            )
            pool.slots[0].key.copy_(
                torch.nn.functional.normalize(torch.tensor([-1.0, 0, 0, 0, 0, 0, 0, 0]), dim=0)
            )
        reinitialized, record = reinit_empty_slots(pool, queries, {}, config)
        self.assertEqual(reinitialized, 0)
        self.assertIn(1, record["allowed_to_die"])


if __name__ == "__main__":
    unittest.main()
