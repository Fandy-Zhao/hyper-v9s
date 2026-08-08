"""Test H (spec §27): cluster-supervised local key learning -- only the
current task's new keys are updated; historical keys are untouched; CE +
old hard-negative hinge + divergence are all active; prototype mode skips
training."""

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from compose.router.key_learning import (
    ComposeKeyLearningConfig,
    learn_cluster_keys,
    key_divergence_loss,
    key_loss_total,
    old_negative_hinge_loss,
)


def _two_cluster_queries(n=24):
    torch.manual_seed(29)
    a = F.normalize(torch.eye(128)[2] + torch.randn(n, 128) * 0.05, dim=1)
    b = F.normalize(torch.eye(128)[70] + torch.randn(n, 128) * 0.05, dim=1)
    queries = torch.cat([a, b], dim=0)
    labels = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    return queries, labels


class KeyLearningTest(unittest.TestCase):
    def test_new_keys_move_toward_cluster_centroids(self):
        queries, labels = _two_cluster_queries()
        torch.manual_seed(31)
        keys = nn.ParameterDict(
            {
                "0": nn.Parameter(F.normalize(torch.randn(128), dim=0)),
                "1": nn.Parameter(F.normalize(torch.randn(128), dim=0)),
            }
        )
        before = {key: value.detach().clone() for key, value in keys.items()}
        # temperature must stay high enough that CE is not saturated from
        # the first step: with temp=0.05 the random-start logits (~1.2 vs 0)
        # already give CE ~ 0 and the key never moves. temp=0.5 keeps a
        # real gradient while the key converges.
        config = ComposeKeyLearningConfig(
            learning_rate=5e-3, epochs=60, temperature=0.5, margin=0.3,
            lambda_old=0.5, lambda_div=0.1, key_separation_margin=0.1,
            batch_size=256, seed=42, key_mode="learnable",
        )
        results = learn_cluster_keys(
            queries, labels, torch.zeros(0, 128), keys, [0, 1], config
        )
        for expert_id in (0, 1):
            result = results[expert_id]
            self.assertEqual(result.key_mode, "learnable")
            self.assertEqual(result.epochs_run, 60)
            self.assertGreater(
                result.positive_similarity_after,
                result.positive_similarity_before + 0.3,
                "key {} did not move toward its cluster".format(expert_id),
            )
            self.assertFalse(
                torch.equal(keys[str(expert_id)].detach(), before[str(expert_id)])
            )

    def test_historical_keys_never_updated(self):
        queries, labels = _two_cluster_queries()
        torch.manual_seed(33)
        keys = nn.ParameterDict(
            {
                "0": nn.Parameter(F.normalize(torch.randn(128), dim=0)),
                "1": nn.Parameter(F.normalize(torch.randn(128), dim=0)),
                "9": nn.Parameter(F.normalize(torch.randn(128), dim=0)),  # historical
            }
        )
        old_before = keys["9"].detach().clone()
        config = ComposeKeyLearningConfig(learning_rate=1e-3, epochs=40, seed=42)
        learn_cluster_keys(queries, labels, torch.zeros(0, 128), keys, [0, 1], config)
        self.assertTrue(torch.equal(keys["9"].detach(), old_before))

    def test_old_negative_hinge_pulls_away_from_history(self):
        queries, labels = _two_cluster_queries(n=16)
        positive = F.normalize(queries[0].clone(), dim=0)
        old_negatives = F.normalize(
            torch.stack([queries[0].clone(), queries[0].clone() * 0.99]), dim=1
        )
        loss = old_negative_hinge_loss(
            queries[:1], positive, old_negatives, margin=0.3
        )
        # cos(q, positive) ~ 1, cos(q, old_neg) ~ 1 -> margin - 1 + 1 ~ 0.3
        self.assertGreater(float(loss), 0.0)
        zero = old_negative_hinge_loss(queries[:1], positive, torch.zeros(0, 128), 0.3)
        self.assertEqual(float(zero), 0.0)

    def test_divergence_loss_penalizes_collinear_new_keys(self):
        keys = F.normalize(
            torch.stack([torch.ones(128), torch.ones(128) * 0.999, torch.ones(128)]),
            dim=1,
        )
        loss = key_divergence_loss(keys, separation_margin=0.1)
        self.assertGreater(float(loss), 0.0)
        orthogonal = F.normalize(torch.stack([torch.eye(128)[0], torch.eye(128)[1]]), dim=1)
        self.assertEqual(float(key_divergence_loss(orthogonal, 0.1)), 0.0)

    def test_loss_terms_compose(self):
        queries, labels = _two_cluster_queries(n=8)
        keys = F.normalize(torch.stack([torch.eye(128)[2], torch.eye(128)[70]]), dim=1)
        config = ComposeKeyLearningConfig(temperature=0.1)
        total, terms = key_loss_total(
            queries, labels, keys, torch.zeros(0, 128), config
        )
        self.assertTrue(torch.isfinite(total))
        self.assertGreaterEqual(float(terms["cluster_ce"]), 0.0)
        self.assertEqual(set(terms.keys()), {"cluster_ce", "old_negative_hinge", "divergence", "total"})

    def test_stats_use_own_key_when_label_positions_differ_from_ids(self):
        # Regression (smoke task 1): labels are POSITIONS into
        # new_expert_ids, while the router key store is keyed by expert-id
        # strings. With new_expert_ids=[7] and a historical key "0" present,
        # _stats() used to index new_keys[str(label)] = new_keys["0"] and
        # recorded similarity to the OLD key instead of the new key 7.
        torch.manual_seed(41)
        queries = F.normalize(torch.eye(128)[2] + torch.randn(16, 128) * 0.05, dim=1)
        labels = torch.zeros(16, dtype=torch.long)  # position 0 -> expert 7
        # Decoy historical key "0" sits AT the cluster centroid: with the
        # bug, _stats() reads new_keys["0"] and records ~0.99 both before
        # and after (the decoy never moves). With the fix it reads the
        # actual new key 7 (init far away, ~0 cosine), whose hinge-driven
        # movement shows up as a real increase.
        keys = nn.ParameterDict(
            {
                "0": nn.Parameter(F.normalize(torch.eye(128)[2], dim=0)),  # decoy old key
                "7": nn.Parameter(F.normalize(torch.eye(128)[90], dim=0)),  # far from cluster
            }
        )
        config = ComposeKeyLearningConfig(
            learning_rate=5e-3, epochs=60, temperature=0.5, margin=0.3,
            lambda_old=0.5, lambda_div=0.1, key_separation_margin=0.1,
            batch_size=256, seed=42, key_mode="learnable",
        )
        old_negatives = F.normalize(torch.eye(128)[70].unsqueeze(0), dim=1)
        results = learn_cluster_keys(
            queries, labels, old_negatives, keys, [7], config
        )
        result = results[7]
        # Buggy stats: 0.99 -> 0.99 (delta 0). Fixed stats: ~0.0 -> 0.35+
        # (the hinge saturates at the margin boundary, cos = margin + neg).
        self.assertGreater(
            result.positive_similarity_after,
            result.positive_similarity_before + 0.2,
            "recorded similarity must track the new key's own movement",
        )

    def test_prototype_mode_skips_training(self):
        queries, labels = _two_cluster_queries(n=8)
        torch.manual_seed(37)
        keys = nn.ParameterDict(
            {
                "0": nn.Parameter(F.normalize(torch.randn(128), dim=0)),
                "1": nn.Parameter(F.normalize(torch.randn(128), dim=0)),
            }
        )
        before = {key: value.detach().clone() for key, value in keys.items()}
        config = ComposeKeyLearningConfig(key_mode="prototype")
        results = learn_cluster_keys(
            queries, labels, torch.zeros(0, 128), keys, [0, 1], config
        )
        for expert_id in (0, 1):
            self.assertEqual(results[expert_id].epochs_run, 0)
            self.assertEqual(results[expert_id].final_loss, 0.0)
            self.assertTrue(torch.equal(keys[str(expert_id)].detach(), before[str(expert_id)]))


if __name__ == "__main__":
    unittest.main()
