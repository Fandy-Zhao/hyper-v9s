import json
import tempfile
import unittest
from pathlib import Path

from compose.oracle.posthoc_audit import CONFIG_NAMES, _distribution, audit


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class PosthocAuditTest(unittest.TestCase):
    def test_percentiles_and_negative_characterization(self):
        result = _distribution([-0.04, -0.03, -0.02, -0.01, 0.01])
        self.assertAlmostEqual(result["median"], -0.02)
        self.assertEqual(result["negative_mean_characterization"], "widespread_small_negative_gains")

    def test_oracles_follow_nll_selection_and_attached_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = ("task:1", "task:2")
            oracle_rows = [
                {
                    "sample_id": ids[0], "task_id": "task",
                    "candidate_expert_ids": [[], [0], [1], [0, 1]],
                    "set_nll": [1.0, 0.5, 0.6, 0.4],
                },
                {
                    "sample_id": ids[1], "task_id": "task",
                    "candidate_expert_ids": [[], [0], [1], [0, 1]],
                    "set_nll": [0.7, 0.3, 0.4, 0.5],
                },
            ]
            _jsonl(root / "oracle.jsonl", oracle_rows)
            _jsonl(root / "rank16_scores.jsonl", [
                {"sample_id": ids[0], "task_id": "task", "nll": 0.45},
                {"sample_id": ids[1], "task_id": "task", "nll": 0.6},
            ])
            _jsonl(root / "direct.jsonl", [
                {"sample_id": ids[0], "task_id": "task", "nll": 0.35},
                {"sample_id": ids[1], "task_id": "task", "nll": 0.8},
            ])
            (root / "annotations.json").write_text(json.dumps([
                {"question_id": ids[0], "answer": "yes"},
                {"question_id": ids[1], "answer": "yes"},
            ]), encoding="utf-8")
            prediction_paths = {}
            for name in CONFIG_NAMES:
                path = root / (name + ".jsonl")
                values = ["no", "no"]
                if name == "pair_direct_sum":
                    values[0] = "yes"
                if name == "single0":
                    values[1] = "yes"
                _jsonl(path, [
                    {"question_id": ids[index], "text": values[index]} for index in range(2)
                ])
                prediction_paths[name] = str(path)
            result, rows = audit(
                str(root / "oracle.jsonl"), str(root / "rank16_scores.jsonl"),
                str(root / "direct.jsonl"), str(root / "annotations.json"), prediction_paths,
            )
            self.assertEqual(rows[0]["oracle_b_selection"], "pair_direct_sum")
            self.assertTrue(rows[0]["oracle_b_correct"])
            self.assertEqual(rows[1]["oracle_b_selection"], "pair_l2")
            self.assertEqual(result["overall"]["pair_exclusive_correct_count"], 1)
            self.assertAlmostEqual(result["overall"]["oracle_b"]["generation_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
