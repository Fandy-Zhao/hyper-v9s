"""Test E (spec §27): clusters smaller than ``min_cluster_samples`` are
noise -- never trained as experts, never part of the cluster manifest."""

import unittest

import torch
from torch.nn import functional as F

from compose.expansion.query_clustering import (
    ComposeClusteringConfig,
    cluster_assignment_map,
    cluster_residual_queries,
)


class SmallClusterNoiseTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        big = F.normalize(torch.eye(128)[3] + torch.randn(28, 128) * 0.03, dim=1)
        small = F.normalize(torch.eye(128)[99] + torch.randn(2, 128) * 0.03, dim=1)
        self.queries = torch.cat([big, small], dim=0)
        self.sample_ids = ["big{:02d}".format(index) for index in range(28)] + [
            "small{:02d}".format(index) for index in range(2)
        ]
        self.config = ComposeClusteringConfig(max_clusters=4, min_cluster_samples=8)

    def test_undersized_cluster_becomes_noise(self):
        result = cluster_residual_queries(
            self.queries, self.sample_ids, self.config, task_id=2, query_hash="h"
        )
        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(result.clusters[0].size, 28)
        self.assertEqual(len(result.noise_sample_ids), 2)
        self.assertEqual(set(result.noise_sample_ids), {"small00", "small01"})
        for sample in result.noise_sample_ids:
            self.assertNotIn(sample, cluster_assignment_map(result))

    def test_noise_samples_never_assigned(self):
        result = cluster_residual_queries(
            self.queries, self.sample_ids, self.config, task_id=2, query_hash="h"
        )
        mapping = cluster_assignment_map(result)
        self.assertEqual(set(mapping.keys()), set(self.sample_ids[:28]))
        self.assertEqual(
            sorted(set(mapping.values())), sorted(cluster.cluster_id for cluster in result.clusters)
        )

    def test_noise_are_not_expert_material(self):
        result = cluster_residual_queries(
            self.queries, self.sample_ids, self.config, task_id=2, query_hash="h"
        )
        # Noise sample ids must never feed expert formation.
        expert_sample_ids = {
            sample
            for cluster in result.clusters
            for sample in cluster.sample_ids
        }
        self.assertTrue(expert_sample_ids.isdisjoint(set(result.noise_sample_ids)))


if __name__ == "__main__":
    unittest.main()
