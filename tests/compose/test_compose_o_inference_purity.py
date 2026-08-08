"""Test O (spec §27): inference purity -- the router selects with ONLY the
frozen query encoder + expert keys + cosine; no answers, no oracle, no
task-id lookup, no clustering at test time; selections are empty/single/
pair (0/1/2 experts)."""

import inspect
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from compose.router.router import ComposeRouter, ComposeRouterSelection


class _StubEncoder(nn.Module):
    query_dim = 128

    def forward(self, z_v, z_s):
        raise NotImplementedError


class InferencePurityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)
        self.router = ComposeRouter(
            _StubEncoder(), top_m=8, tau_none=0.5, tau_second=0.5
        )
        for expert_id in range(3):
            self.router.add_expert(
                expert_id, creation_task=0, checkpoint_sha256="sha",
                key=F.normalize(torch.randn(128), dim=0),
            )
        self.keys = [self.router.key_store.keys[str(i)].detach() for i in range(3)]

    def test_select_signature_is_answer_free(self):
        signature = inspect.signature(ComposeRouter.select)
        parameters = list(signature.parameters)
        self.assertEqual(parameters[:2], ["self", "query"])
        # No answer/oracle/task-id/cluster inputs exist in the API.
        for forbidden in ("answer", "oracle", "task_id", "cluster", "label"):
            self.assertFalse(
                any(forbidden in name for name in parameters),
                "select() must not accept {!r}".format(forbidden),
            )

    def test_audit_flags_all_false(self):
        query = F.normalize(0.9 * self.keys[0] + 0.5 * self.keys[1], dim=0).unsqueeze(0)
        selection = self.router.select(query, self.router.expert_ids)
        self.assertIsInstance(selection, ComposeRouterSelection)
        self.assertFalse(selection.answer_features_used)
        self.assertFalse(selection.oracle_used)
        self.assertFalse(selection.task_id_lookup_used)
        self.assertFalse(selection.clustering_used_at_test)

    def test_empty_single_pair_grammar(self):
        # Pair: both top sims above the second threshold.
        pair_query = F.normalize(0.9 * self.keys[0] + 0.7 * self.keys[1], dim=0).unsqueeze(0)
        pair = self.router.select(pair_query, self.router.expert_ids)
        self.assertEqual(pair.sets, ((0, 1),))
        # Single: top above tau_none, second below tau_second.
        single_query = F.normalize(0.95 * self.keys[0] + 0.1 * self.keys[1], dim=0).unsqueeze(0)
        single = self.router.select(single_query, self.router.expert_ids)
        self.assertEqual(single.sets, ((0,),))
        # Empty: everything below tau_none (orthogonal query).
        torch.manual_seed(59)
        empty_query = F.normalize(torch.randn(128), dim=0).unsqueeze(0)
        empty = self.router.select(empty_query, self.router.expert_ids)
        self.assertEqual(empty.sets, ((),))
        # Never more than max_active_experts=2 experts per sample.
        for selection in (pair, single, empty):
            self.assertTrue(
                all(len(selected) <= 2 for selected in selection.sets)
            )

    def test_probabilities_are_pure_cosine(self):
        query = F.normalize(self.keys[1] * 0.8 + self.keys[0] * 0.4, dim=0).unsqueeze(0)
        selection = self.router.select(query, self.router.expert_ids)
        keys = torch.stack(self.keys)
        expected = (query @ keys.T).squeeze(0)
        self.assertTrue(
            torch.allclose(selection.probabilities, expected, atol=1e-6),
            "probabilities must be plain cosine similarities",
        )

    def test_frozen_old_keys_never_modified_by_inference(self):
        before = {key: value.detach().clone() for key, value in self.router.key_store.keys.items()}
        query = F.normalize(self.keys[0].clone(), dim=0).unsqueeze(0)
        for _ in range(5):
            self.router.select(query, self.router.expert_ids)
            self.router.retrieve(query, self.router.expert_ids)
        after = {key: value.detach().clone() for key, value in self.router.key_store.keys.items()}
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]))


if __name__ == "__main__":
    unittest.main()
