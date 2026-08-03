import unittest

import torch
from torch import nn

from compose.experts import ExpertRegistry

from compose.expansion.candidate_pool import (
    CandidateExpertPool,
    CandidatePoolConfig,
    SlotValidation,
    kmeans_plus_plus_keys,
)


class CandidatePoolTest(unittest.TestCase):
    def pool(self):
        keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        return CandidateExpertPool([nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)], keys, CandidatePoolConfig(query_dim=2, min_support=2))

    def test_top1_only_selected_slot_receives_gradient(self):
        pool = self.pool()
        hidden = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        queries = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        output, assignments = pool(hidden, queries, torch.zeros_like(hidden))
        output.sum().backward()
        self.assertEqual(assignments.tolist(), [0, 0])
        self.assertTrue(any(parameter.grad is not None for parameter in pool.slots[0].adapter.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in pool.slots[1].adapter.parameters()))

    def test_optimizer_states_are_isolated(self):
        pool = self.pool()
        optimizers = pool.independent_optimizers(lr=1e-3)
        first = {id(parameter) for group in optimizers[0].param_groups for parameter in group["params"]}
        second = {id(parameter) for group in optimizers[1].param_groups for parameter in group["params"]}
        self.assertFalse(first & second)

    def test_commit_zero_one_two_and_provisional_status(self):
        pool = self.pool()
        good = SlotValidation(3, 0.2, 0.0, 0.8, ("a", "b"))
        bad = SlotValidation(0, -0.1, -0.1, 0.0, ())
        self.assertEqual(pool.commit((bad, bad))["commit_count"], 0)
        one = pool.commit((good, bad))
        self.assertEqual(one["commit_count"], 1)
        self.assertEqual(one["slots"][0]["status"], "provisional")
        second_good = SlotValidation(3, 0.1, 0.1, 0.9, ("c", "d"))
        two = pool.commit((good, second_good))
        self.assertEqual(two["commit_count"], 2)
        registry = ExpertRegistry()
        committed = pool.commit_to_registry(registry, two, first_expert_id=10, creation_task=3)
        self.assertEqual([item.expert_id for item in committed], [10, 11])
        self.assertTrue(all(item.extra["lifecycle_state"] == "provisional" for item in committed))

    def test_kmeans_plus_plus_is_reproducible(self):
        queries = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        torch.testing.assert_close(kmeans_plus_plus_keys(queries, seed=7), kmeans_plus_plus_keys(queries, seed=7))


if __name__ == "__main__":
    unittest.main()
