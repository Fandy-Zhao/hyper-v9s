import hashlib
import json
import unittest

import torch
from torch.nn import functional as F

from compose.expansion.query_clustering import (
    ComposeClusteringConfig,
    build_single_bootstrap_cluster,
    cluster_residual_queries,
    cosine_silhouette,
)
from compose.lora.rms import merge_commit_frozen_calibration
from compose.experts.metadata import ExpertMetadata, ExpertStatus


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class V62BootstrapClusteringRMSTest(unittest.TestCase):
    def test_task0_builds_exactly_one_full_data_bootstrap(self):
        queries = F.normalize(torch.randn(17, 128, generator=torch.Generator().manual_seed(4)), dim=1)
        result = build_single_bootstrap_cluster(
            queries, ["s{}".format(index) for index in range(17)]
        )
        self.assertEqual(result.selected_k, 1)
        self.assertEqual(result.effective_cluster_count, 1)
        self.assertEqual(result.clusters[0].size, 17)
        self.assertEqual(result.noise_sample_ids, [])
        self.assertEqual(result.silhouette_by_k, {1: None})

    def test_standard_silhouette_uses_mean_cluster_distances(self):
        # Point 0 has a close neighbour and a distant same-cluster point.
        # Nearest-neighbour silhouette would ignore the distant member.
        raw = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [-1.0, 0.0]])
        queries = F.pad(F.normalize(raw, dim=1), (0, 126))
        labels = torch.tensor([0, 0, 0, 1])
        score = cosine_silhouette(queries, labels, 2)
        distances = (1.0 - queries @ queries.T).clamp(0.0, 2.0)
        expected = []
        for index in range(4):
            own = labels == labels[index]
            own[index] = False
            if not own.any():
                expected.append(0.0)
                continue
            a = distances[index, own].mean()
            b = min(distances[index, labels == other].mean() for other in set(labels.tolist()) if other != int(labels[index]))
            expected.append(float((b - a) / torch.maximum(a, b)))
        self.assertAlmostEqual(score, sum(expected) / len(expected), places=6)

    def test_residual_clustering_records_n_init_20(self):
        generator = torch.Generator().manual_seed(5)
        a = F.normalize(torch.randn(20, 128, generator=generator) * 0.01 + F.pad(torch.tensor([1.0]), (0, 127)), dim=1)
        b = F.normalize(torch.randn(20, 128, generator=generator) * 0.01 + F.pad(torch.tensor([-1.0]), (0, 127)), dim=1)
        result = cluster_residual_queries(
            torch.cat([a, b]), [str(i) for i in range(40)],
            ComposeClusteringConfig(n_init=20, min_cluster_samples=2), task_id=1,
        )
        self.assertEqual(result.n_init, 20)
        self.assertEqual(result.effective_cluster_count, len(result.clusters))
        self.assertGreaterEqual(result.selected_k, 2)

    def test_old_kappa_hash_is_immutable_after_pool_growth(self):
        frozen = {"layer": {"0": 1.25, "1": 0.75}}
        before = _digest(frozen)
        dynamic = {"layer": {"0": 3.0, "1": 2.0, "2": 0.5}}
        merged = merge_commit_frozen_calibration(frozen, dynamic, [2])
        old_after = {layer: {key: value for key, value in entries.items() if key in {"0", "1"}}
                     for layer, entries in merged.items()}
        self.assertEqual(_digest(old_after), before)
        self.assertEqual(merged["layer"]["2"], 0.5)

    def test_historical_metadata_is_not_mutated_when_new_rms_is_bound(self):
        old = ExpertMetadata(
            expert_id=0, adapter_name="expert_0000", rank=8, alpha=16,
            status=ExpertStatus.FROZEN, active=True,
            rms_stats_path="task0/rms.json",
            extra={"rms_calibration_sha256": "old-hash"},
        )
        before = _digest(old.to_dict())
        new = ExpertMetadata(
            expert_id=1, adapter_name="expert_0001", rank=8, alpha=16,
            status=ExpertStatus.FROZEN, active=True,
        )
        new.rms_stats_path = "task1/rms.json"
        new.extra["rms_calibration_sha256"] = "new-hash"
        self.assertEqual(_digest(old.to_dict()), before)


if __name__ == "__main__":
    unittest.main()
