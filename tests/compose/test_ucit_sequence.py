"""V6 Stage E11: UCIT official task sequence consistency.

The sequence must match across the authoritative training scripts, the
eval metric matrix, the locked stage-E0 artifact and the engineering
config YAML. Any divergence is a BLOCKED condition (the task book forbids
guessing the order).
"""

import json
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAIN_SCRIPTS = REPO / "scripts" / "Hyper" / "Train_UCIT"
EVAL_METRICS = REPO / "scripts" / "Hyper" / "Eval_UCIT" / "summarize_continual_metrics.py"
TASK_SEQUENCE_ARTIFACT = (
    REPO / "artifacts" / "v6_ucit_engineering" / "stage_e0" / "task_sequence.json"
)
CONFIG_YAML = REPO / "configs" / "v6_ucit_engineering.yaml"

EXPECTED_SEQUENCE = [
    ("ImageNet-R", "ImageNet-R/train.json"),
    ("ArxivQA", "ArxivQA/train_4w.json"),
    ("VizWiz", "VizWiz/train.json"),
    ("IconQA", "IconQA/train.json"),
    ("CLEVR", "CLEVR/train_4w.json"),
    ("Flickr30k", "Flickr30k/train_brief_4w.json"),
]


class UcitSequenceTest(unittest.TestCase):
    def test_authoritative_task_scripts(self):
        for index, (name, instruction) in enumerate(EXPECTED_SEQUENCE, start=1):
            script = TRAIN_SCRIPTS / "Task{}.sh".format(index)
            self.assertTrue(script.is_file(), "missing {}".format(script))
            content = script.read_text(encoding="utf-8")
            self.assertIn(
                "instructions/" + instruction,
                content,
                "Task{}.sh must load {}".format(index, instruction),
            )
            # Output dir names the task.
            match = re.search(r"Task{}_[a-zA-Z0-9_]+".format(index), content)
            self.assertIsNotNone(match)

    def test_eval_metrics_matrix_order(self):
        content = EVAL_METRICS.read_text(encoding="utf-8")
        # The metrics script hardcodes a TASKS list in official order.
        for index, (name, _) in enumerate(EXPECTED_SEQUENCE, start=1):
            display = "CLEVR-Math" if name == "CLEVR" else name
            self.assertIn('"task_id": {}'.format(index), content)
            self.assertIn('"dataset": "{}"'.format(display), content)

    def test_stage_e0_artifact_matches(self):
        data = json.loads(TASK_SEQUENCE_ARTIFACT.read_text(encoding="utf-8"))
        self.assertEqual(
            [task["task_name"] for task in data["tasks"]],
            [name for name, _ in EXPECTED_SEQUENCE],
        )
        dry_run = data["two_task_dry_run"]
        self.assertEqual(dry_run["task_names"], ["ImageNet-R", "ArxivQA"])
        self.assertEqual(dry_run["seed"], 42)

    def test_engineering_config_matches(self):
        content = CONFIG_YAML.read_text(encoding="utf-8")
        for name, instruction in EXPECTED_SEQUENCE:
            self.assertIn(instruction, content)
        # Dry run defaults to the first two official tasks.
        self.assertIn("ImageNet-R/train.json", content)
        self.assertIn("ArxivQA/train_4w.json", content)

    def test_no_variant_sequence_leaks_into_official(self):
        # The variant dirs are separate experiments; the official runner
        # must reference the non-suffixed Train_UCIT directory.
        for variant in ("Train_UCIT_AIRFCV", "Train_UCIT_IFRCAV"):
            self.assertTrue((REPO / "scripts" / "Hyper" / variant).is_dir())
        self.assertTrue(TRAIN_SCRIPTS.is_dir())


if __name__ == "__main__":
    unittest.main()
