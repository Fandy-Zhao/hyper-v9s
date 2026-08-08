"""Test D (spec §27): two well-separated spherical residual blobs are
recovered as K=2 with pure cluster memberships and matching centroids."""

import math
import unittest

import torch
from torch.nn import functional as F

from compose.expansion.query_clustering import (
    ComposeClusteringConfig,
    cluster_residual_queries,
)


class TwoSphericalClustersTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        # Cluster A along e_7, cluster B along e_53 (orthogonal in 128-D).
        a = F.normalize(torch.eye(128)[7] + torch.randn(20, 128) * 0.03, dim=1)
        b = F.normalize(torch.eye(128)[53] + torch.randn(20, 128) * 0.03, dim=1)
        self.queries = torch.cat([a, b], dim=0)
        self.sample_ids = ["a{:02d}".format(index) for index in range(20)] + [
            "b{:02d}".format(index) for index in range(20)
        ]
        self.config = ComposeClusteringConfig(max_clusters=4, min_cluster_samples=8)

    def test_selects_k2(self):
        result = cluster_residual_queries(
            self.queries, self.sample_ids, self.config, task_id=1, query_hash="h"
        )
        self.assertEqual(result.selected_k, 2)
        self.assertIsNotNone(result.selected_silhouette)
        self.assertGreaterEqual(result.selected_silhouette, 0.15)

    def test_cluster_memberships_are_pure(self):
        result = cluster_residual_queries(
            self.queries, self.sample_ids, self.config, task_id=1, query_hash="h"
        )
        self.assertEqual(len(result.clusters), 2)
        sizes = sorted(cluster.size for cluster in result.clusters)
        self.assertEqual(sizes, [20, 20])
        for cluster in result.clusters:
            members = set(cluster.sample_ids)
            prefix = {"a" if sample.startswith("a") else "b" for sample in members}
            self.assertEqual(len(prefix), 1, "cluster mixes both blobs: {}".format(members))
        self.assertEqual(result.noise_sample_ids, [])

    def test_centroids_match_blob_directions(self):
        result = cluster_residual_queries(
            self.queries, self.sample_ids, self.config, task_id=1, query_hash="h"
        )
        by_axis = {}
        for cluster in result.clusters:
            axis = 7 if cluster.sample_ids[0].startswith("a") else 53
            by_axis[axis] = cluster.centroid
        for axis, centroid in by_axis.items():
            cos = float(sum(
                centroid[index] * (1.0 if index == axis else 0.0)
                for index in range(128)
            ))
            self.assertGreater(cos, 0.9, "centroid diverged from axis {}".format(axis))


if __name__ == "__main__":
    unittest.main()
