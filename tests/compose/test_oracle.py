import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from llava.constants import IGNORE_INDEX

from compose.oracle.cache import config_hash, read_jsonl, write_jsonl
from compose.oracle.candidate_sets import CandidateSetIndex, build_candidate_sets
from compose.oracle.losses import compute_per_sample_nll
from compose.oracle.metrics import compute_oracle_record, summarize_oracle_records
from compose.oracle.build_mixture import build_mixture
from compose.oracle.aggregate import aggregate_records


class OracleLossTest(unittest.TestCase):
    def test_per_sample_nll_matches_manual_and_single_sample_results(self):
        logits = torch.tensor(
            [
                [[3.0, 1.0], [0.0, 2.0], [9.0, 9.0]],
                [[1.0, 2.0], [4.0, 0.0], [9.0, 9.0]],
            ],
            dtype=torch.bfloat16,
        )
        labels = torch.tensor(
            [[IGNORE_INDEX, 0, 1], [IGNORE_INDEX, 1, IGNORE_INDEX]]
        )
        result = compute_per_sample_nll(logits, labels, return_details=True)
        manual_first = torch.stack(
            [
                F.cross_entropy(logits[0, 0].float().unsqueeze(0), torch.tensor([0])),
                F.cross_entropy(logits[0, 1].float().unsqueeze(0), torch.tensor([1])),
            ]
        ).mean()
        manual_second = F.cross_entropy(
            logits[1, 0].float().unsqueeze(0), torch.tensor([1])
        )
        torch.testing.assert_close(
            result["mean_nll"], torch.stack([manual_first, manual_second])
        )
        self.assertEqual(result["valid_token_count"].tolist(), [2, 1])
        individual = torch.cat(
            [compute_per_sample_nll(logits[index:index + 1], labels[index:index + 1]) for index in range(2)]
        )
        torch.testing.assert_close(result["mean_nll"], individual)
        torch.testing.assert_close(
            compute_per_sample_nll(logits, labels),
            compute_per_sample_nll(logits, labels),
        )

    def test_zero_target_tokens_raise(self):
        with self.assertRaisesRegex(ValueError, "zero target tokens"):
            compute_per_sample_nll(
                torch.zeros(1, 3, 2),
                torch.full((1, 3), IGNORE_INDEX),
            )


class CandidateSetTest(unittest.TestCase):
    def test_four_experts_produce_stable_eleven_set_mapping(self):
        candidates = build_candidate_sets([3, 1, 2, 0])
        self.assertEqual(len(candidates), 11)
        self.assertEqual(candidates[0].expert_ids, ())
        self.assertEqual(candidates[1].expert_ids, (0,))
        self.assertEqual(candidates[5].expert_ids, (0, 1))
        self.assertEqual(candidates[-1].expert_ids, (2, 3))
        self.assertEqual(candidates[0].normalization, "none")
        self.assertEqual(candidates[5].normalization, "l2")
        self.assertAlmostEqual(candidates[5].gates[0], 1.0 / math.sqrt(2.0))
        index = CandidateSetIndex([0, 1, 2, 3])
        self.assertEqual(index.to_index([3, 1]), index.to_index([1, 3]))
        self.assertEqual(index.from_index(index.to_index([1, 3])).expert_ids, (1, 3))

    def test_best_single_pair_synergy_and_summary(self):
        candidates = build_candidate_sets([0, 1])
        metrics = compute_oracle_record([1.4, 1.0, 1.2, 0.7], candidates)
        self.assertEqual(metrics["best_single_index"], 1)
        self.assertEqual(metrics["best_pair_index"], 3)
        self.assertEqual(metrics["best_overall_index"], 3)
        self.assertAlmostEqual(metrics["synergy"], 0.3)
        self.assertTrue(metrics["pair_oracle"])
        summary = summarize_oracle_records([metrics, metrics])
        self.assertEqual(summary["pair_oracle_rate"], 1.0)
        self.assertEqual(summary["positive_synergy_rate"], 1.0)
        row = {
            **metrics,
            "set_nll": [1.4, 1.0, 1.2, 0.7],
            "candidate_expert_ids": [[], [0], [1], [0, 1]],
        }
        aggregate = aggregate_records([row])
        self.assertEqual(aggregate["average_active_experts"], 2.0)
        self.assertEqual(
            aggregate["synergy_threshold_pair_acceptance"]["0.05"], 1.0
        )


class OracleCacheTest(unittest.TestCase):
    def test_cache_round_trip_and_config_hash_are_stable(self):
        records = [{"sample_id": "a", "set_nll": [1.0, 0.5]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            self.assertEqual(write_jsonl(str(path), records), 1)
            self.assertEqual(list(read_jsonl(str(path))), records)
        left = config_hash({"b": 2, "a": [1]})
        right = config_hash(json.loads('{"a":[1],"b":2}'))
        self.assertEqual(left, right)

    def test_mixture_namespaces_sample_ids_and_preserves_task_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            first.write_text(json.dumps([{"question_id": "1", "text": "a"}]))
            second.write_text(json.dumps([{"question_id": "1", "text": "b"}]))
            records, counts = build_mixture(
                [str(first), str(second)], ["task-a", "task-b"]
            )
            self.assertEqual([row["question_id"] for row in records], ["task-a:1", "task-b:1"])
            self.assertEqual([row["task_id"] for row in records], ["task-a", "task-b"])
            self.assertEqual(counts, {"task-a": 1, "task-b": 1})


if __name__ == "__main__":
    unittest.main()
