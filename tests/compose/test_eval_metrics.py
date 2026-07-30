import json
import os
import tempfile
import unittest

from compose.eval.load_compose import _read_compose_manifest
from compose.eval.metrics import imagenet_r_exact_match
from compose.experts.checkpoint import MANIFEST_NAME


class EvalMetricsTest(unittest.TestCase):
    def test_task1_exact_match_is_case_insensitive_and_id_aligned(self):
        annotations = [
            {"question_id": "a", "answer": "Centipede"},
            {"question_id": "b", "answer": "Red fox"},
        ]
        predictions = [
            {"question_id": "b", "text": "red fox"},
            {"question_id": "a", "text": "  CENTIPEDE  "},
        ]
        result = imagenet_r_exact_match(annotations, predictions)
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["accuracy"], 1.0)

    def test_task1_metric_rejects_partial_predictions(self):
        with self.assertRaisesRegex(ValueError, "count does not match"):
            imagenet_r_exact_match(
                [
                    {"question_id": "a", "answer": "one"},
                    {"question_id": "b", "answer": "two"},
                ],
                [{"question_id": "a", "text": "one"}],
            )

    def test_compose_manifest_requires_complete_adapter_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, MANIFEST_NAME)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"adapter": {"rank": 8}}, handle)
            with self.assertRaisesRegex(ValueError, "missing"):
                _read_compose_manifest(directory)

    def test_compose_manifest_accepts_strict_adapter_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, MANIFEST_NAME)
            manifest = {
                "adapter": {
                    "rank": 8,
                    "alpha": 16,
                    "dropout": 0,
                    "layers": ["model.layers.0.self_attn.q_proj"],
                }
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            self.assertEqual(_read_compose_manifest(directory), manifest)


if __name__ == "__main__":
    unittest.main()
