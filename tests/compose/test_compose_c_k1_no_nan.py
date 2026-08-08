"""Test C (spec §27): a single residual blob degrades to K=1 with a finite,
deterministic assignment -- no NaN centroids, no noise, every sample covered."""

import math
import unittest

import torch
from torch.nn import functional as F

from compose.expansion.query_clustering import (
    ComposeClusteringConfig,
    cluster_residual_queries,
)


class K1NoNanTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        centroid = F.normalize(torch.randn(128), dim=0)
        # Per-row normalization (dim=1): each row is the centroid plus tiny
        # noise. Normalizing dim=0 would normalize every COLUMN across the
        # batch and collapse all rows into nearly the same vector.
        queries = F.normalize(centroid + torch.randn(32, 128) * 0.02, dim=1)
        self.queries = queries
        self.sample_ids = ["s{:02d}".format(index) for index in range(32)]
        self.config = ComposeClusteringConfig(max_clusters=4, min_cluster_samples=8)

    def test_single_blob_selects_k1(self):
        result = cluster_residual_queries(
            self.queries, self.sample_ids, self.config, task_id=0, query_hash="h"
        )
        self.assertEqual(result.selected_k, 1)
        self.assertIsNone(result.selected_silhouette)
        self.assertEqual(len(result.clusters), 1)

    def test_centroid_finite_and_normalized(self):
        result = cluster_residual_queries(
            self.queries, self.sample_ids, self.config, task_id=0, query_hash="h"
        )
        centroid = result.clusters[0].centroid
        self.assertEqual(len(centroid), 128)
        for value in centroid:
            self.assertTrue(math.isfinite(float(value)), "NaN centroid value")
        norm = math.sqrt(sum(float(value) ** 2 for value in centroid))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_all_samples_covered_no_noise(self):
        result = cluster_residual_queries(
            self.queries, self.sample_ids, self.config, task_id=0, query_hash="h"
        )
        assigned = set(result.clusters[0].sample_ids)
        self.assertEqual(assigned, set(self.sample_ids))
        self.assertEqual(result.noise_sample_ids, [])

    def test_deterministic_across_runs(self):
        first = cluster_residual_queries(
            self.queries, self.sample_ids, self.config, task_id=0, query_hash="h"
        )
        second = cluster_residual_queries(
            self.queries, self.sample_ids, self.config, task_id=0, query_hash="h"
        )
        self.assertEqual(
            set(first.clusters[0].sample_ids), set(second.clusters[0].sample_ids)
        )
        self.assertEqual(first.clusters[0].centroid, second.clusters[0].centroid)

    def test_max_clusters_one_is_k1(self):
        config = ComposeClusteringConfig(max_clusters=1, min_cluster_samples=8)
        result = cluster_residual_queries(
            self.queries, self.sample_ids, config, task_id=0, query_hash="h"
        )
        self.assertEqual(result.selected_k, 1)
        self.assertEqual(result.silhouette_by_k, {1: None})


if __name__ == "__main__":
    unittest.main()
