"""Compose Stage E11: UCIT official task sequence consistency.

The sequence must match across the authoritative Compose sources -- the
in-code default config (``compose/experiments/task_run.py``), the external
config (``configs/compose_ucit.yaml``), the unified runner script
(``scripts/Compose/Run_UCIT/six_task_run.sh``) -- and the original Hyper
baseline scripts that encode the same order. Any divergence is a BLOCKED
condition (the task book forbids guessing the order).
"""

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASK_RUN = REPO / "compose" / "experiments" / "task_run.py"
CONFIG_YAML = REPO / "configs" / "compose_ucit.yaml"
RUN_SCRIPTS = REPO / "scripts" / "Compose" / "Run_UCIT"
TRAIN_SCRIPTS = REPO / "scripts" / "Hyper" / "Train_UCIT"
EVAL_METRICS = (
    REPO / "scripts" / "Hyper" / "Eval_UCIT" / "summarize_continual_metrics.py"
)

EXPECTED_SEQUENCE = [
    ("ImageNet-R", "ImageNet-R/train.json"),
    ("ArxivQA", "ArxivQA/train_4w.json"),
    ("VizWiz", "VizWiz/train.json"),
    ("IconQA", "IconQA/train.json"),
    ("CLEVR", "CLEVR/train_4w.json"),
    ("Flickr30k", "Flickr30k/train_brief_4w.json"),
]


def _train_instruction_paths(text):
    return re.findall(r"instructions/([A-Za-z0-9_./-]+\.json)", text)


class ComposeSequenceTest(unittest.TestCase):
    def test_task_run_default_config_sequence(self):
        content = TASK_RUN.read_text(encoding="utf-8")
        paths = _train_instruction_paths(content)
        for name, instruction in EXPECTED_SEQUENCE:
            self.assertIn(instruction, paths)
            self.assertEqual(paths.count(instruction), 1, name)
        self.assertIn('"name": "ImageNet-R"', content)
        self.assertIn('"name": "Flickr30k"', content)

    def test_compose_config_matches(self):
        content = CONFIG_YAML.read_text(encoding="utf-8")
        positions = []
        for name, instruction in EXPECTED_SEQUENCE:
            match = re.search(
                r"instructions/" + re.escape(instruction), content
            )
            self.assertIsNotNone(
                match,
                "configs/compose_ucit.yaml must list {}".format(instruction),
            )
            positions.append(match.start())
        self.assertEqual(
            positions, sorted(positions),
            "compose_ucit.yaml must keep the official task order",
        )

    def test_runner_script_references_task_run(self):
        script = RUN_SCRIPTS / "six_task_run.sh"
        self.assertTrue(script.is_file(), "missing {}".format(script))
        content = script.read_text(encoding="utf-8")
        self.assertIn("task_run.py", content)
        self.assertIn("compose_ucit.yaml", content)
        self.assertTrue((RUN_SCRIPTS / "resume_run.sh").is_file())

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

    def test_no_variant_sequence_leaks_into_official(self):
        # The variant dirs are separate experiments; the official runner
        # must reference the non-suffixed Train_UCIT directory.
        for variant in ("Train_UCIT_AIRFCV", "Train_UCIT_IFRCAV"):
            self.assertTrue((REPO / "scripts" / "Hyper" / variant).is_dir())
        self.assertTrue(TRAIN_SCRIPTS.is_dir())

    def test_compose_cli_contract_uses_hyphenated_flags(self):
        # Regression: the runner invokes train_compose with the spec-§12
        # hyphenated flags (--compose-mode, --compose-selection-manifest,
        # --compose-cluster-expert-ids, --compose-checkpoint), and
        # train_compose normalizes them onto the underscore flags that
        # transformers 4.33's HfArgumentParser registers.
        from compose.train.train_compose import _normalize_compose_argv

        argv = [
            "train_compose.py",
            "--compose-mode", "cluster_expert",
            "--compose-selection-manifest", "manifest.json",
            "--compose-cluster-expert-ids", "0,1",
            "--compose-checkpoint", "/tmp/ckpt",
            "--model_name_or_path", "x",
            "--per_device_train_batch_size", "1",
        ]
        normalized = _normalize_compose_argv(argv)
        self.assertEqual(
            normalized,
            [
                "train_compose.py",
                "--compose_mode", "cluster_expert",
                "--compose_selection_manifest", "manifest.json",
                "--compose_cluster_expert_ids", "0,1",
                "--compose_checkpoint", "/tmp/ckpt",
                "--model_name_or_path", "x",
                "--per_device_train_batch_size", "1",
            ],
        )
        # Non-compose arguments (including underscore-style TrainingArguments
        # flags) pass through untouched.
        self.assertEqual(normalized[9:], argv[9:])


if __name__ == "__main__":
    unittest.main()
