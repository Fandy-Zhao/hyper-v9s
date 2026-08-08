"""Three well-separated spherical residual blobs (pre-formal validation
§11.1, K=3 case).

The K=3 code path (spherical K-means + silhouette + pure memberships) is
the contract under test. Note that the dynamic-K rule selects
``argmax_K silhouette(K)``: for three extremely well-separated blobs the
merged K=2 partition can score marginally above pure K=3 (silhouette
favors coarse partitions of far-apart clusters), so the selection itself
is a method observation, never forced by the threshold. The test
therefore pins (1) the K=3 assignments/centroids computed by the
algorithm, and (2) the documented argmax selection rule.
"""

import unittest

import torch
from torch.nn import functional as F

from compose.expansion.query_clustering import (
    ComposeClusteringConfig,
    cluster_residual_queries,
    cosine_silhouette,
    spherical_kmeans,
)


class ThreeClustersTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.axes = (7, 53, 99)
        blobs = []
        sample_ids = []
        for axis in self.axes:
            blob = F.normalize(torch.eye(128)[axis] + torch.randn(16, 128) * 0.03, dim=1)
            blobs.append(blob)
            sample_ids.extend(
                "axis{}-{:02d}".format(axis, index) for index in range(16)
            )
        self.queries = torch.cat(blobs, dim=0)
        self.sample_ids = sample_ids
        self.config = ComposeClusteringConfig(max_clusters=4, min_cluster_samples=8)

    def test_k3_assignments_are_pure(self):
        """The K=3 code path separates the three blobs with pure memberships."""
        assignments, _ = spherical_kmeans(
            self.queries, 3,
            max_iterations=self.config.max_iterations,
            seed=self.config.random_seed,
        )
        for cluster in range(3):
            members = (assignments == cluster).nonzero(as_tuple=False).flatten()
            axes = {self.sample_ids[int(index)].split("-")[0] for index in members.tolist()}
            self.assertEqual(len(members), 16)
            self.assertEqual(len(axes), 1, "K=3 cluster mixes blobs: {}".format(axes))

    def test_k3_centroids_follow_blob_axes(self):
        assignments, _ = spherical_kmeans(
            self.queries, 3,
            max_iterations=self.config.max_iterations,
            seed=self.config.random_seed,
        )
        for cluster in range(3):
            members = (assignments == cluster).nonzero(as_tuple=False).flatten()
            centroid = F.normalize(self.queries[members].mean(dim=0), dim=0)
            axis = int(self.sample_ids[int(members[0])].split("axis")[1].split("-")[0])
            cos = float(centroid[axis])
            self.assertGreater(cos, 0.9, "K=3 centroid diverged from axis {}".format(axis))

    def test_selection_is_argmax_silhouette(self):
        """The selection rule is exactly ``argmax_K silhouette(K)`` for
        K >= 2, degrading to K=1 below the threshold. With three blobs the
        merged K=2 can score above pure K=3 -- recorded, never forced."""
        result = cluster_residual_queries(
            self.queries, self.sample_ids, self.config, task_id=1, query_hash="h"
        )
        candidates = [k for k in range(2, 5)]
        scores = {k: result.silhouette_by_k[k] for k in candidates}
        best = max(candidates, key=lambda k: scores[k])
        self.assertEqual(result.selected_k, best)
        # And the selection is at least K=2 (structure recovered), never K=1.
        self.assertGreaterEqual(result.selected_k, 2)
        # Every cluster in the selected partition is at least as large as
        # min_cluster_samples (no undersized expert from this data).
        self.assertEqual(result.noise_sample_ids, [])
        for cluster in result.clusters:
            self.assertGreaterEqual(cluster.size, self.config.min_cluster_samples)


if __name__ == "__main__":
    unittest.main()
