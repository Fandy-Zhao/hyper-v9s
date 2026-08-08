"""Test F (spec §27): cluster assignment is fixed at manifest time and
frozen from then on -- reruns reproduce the identical manifest hash,
and LoRA/key training never re-decide membership."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from compose.expansion.query_clustering import (
    CLUSTER_MANIFEST_VERSION,
    ComposeClusteringConfig,
    cluster_assignment_map,
    cluster_residual_queries,
    load_cluster_manifest,
    write_cluster_manifest,
)


class FixedAssignmentTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        a = F.normalize(torch.eye(128)[5] + torch.randn(16, 128) * 0.03, dim=1)
        b = F.normalize(torch.eye(128)[61] + torch.randn(16, 128) * 0.03, dim=1)
        self.queries = torch.cat([a, b], dim=0)
        self.sample_ids = ["s{:02d}".format(index) for index in range(32)]
        self.config = ComposeClusteringConfig(max_clusters=4, min_cluster_samples=8)

    def test_rerun_produces_identical_assignment(self):
        first = cluster_residual_queries(
            self.queries, self.sample_ids, self.config, task_id=3, query_hash="q"
        )
        second = cluster_residual_queries(
            self.queries, self.sample_ids, self.config, task_id=3, query_hash="q"
        )
        self.assertEqual(
            cluster_assignment_map(first), cluster_assignment_map(second)
        )

    def test_manifest_hash_is_stable(self):
        result = cluster_residual_queries(
            self.queries, self.sample_ids, self.config, task_id=3, query_hash="q"
        )
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.json"
            second_path = Path(directory) / "second.json"
            first_hash = write_cluster_manifest(result, str(first_path), extra={"task_id": 3})
            second_hash = write_cluster_manifest(result, str(second_path), extra={"task_id": 3})
            self.assertEqual(first_hash, second_hash)

    def test_manifest_round_trip_frozen(self):
        result = cluster_residual_queries(
            self.queries, self.sample_ids, self.config, task_id=3, query_hash="q"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cluster_manifest.json"
            write_cluster_manifest(result, str(path), extra={"task_id": 3})
            payload = load_cluster_manifest(str(path))
        self.assertEqual(payload["schema_version"], CLUSTER_MANIFEST_VERSION)
        stored = payload["result"]
        self.assertEqual(stored["selected_k"], result.selected_k)
        self.assertEqual(
            {cluster["cluster_id"] for cluster in stored["clusters"]},
            {cluster.cluster_id for cluster in result.clusters},
        )
        # Assignment persists across a reload: sample -> cluster stable.
        remapped = {
            sample: cluster["cluster_id"]
            for cluster in stored["clusters"]
            for sample in cluster["sample_ids"]
        }
        self.assertEqual(remapped, cluster_assignment_map(result))

    def test_wrong_schema_version_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cluster_manifest(str(path))


if __name__ == "__main__":
    unittest.main()
