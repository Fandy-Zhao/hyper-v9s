"""Formal UCIT run config sanity (pre-formal validation §23 regression).

Guards the six-task run file against stale instruction paths: every
``train_instructions`` / ``test_instructions`` in
``configs/compose_ucit.yaml`` must exist on disk, and the algorithm
contract values (tau_res, silhouette threshold, router thresholds, key
learning hyperparameters, RMS bounds, seed 42) must match the design.

Regression for: tasks 2-5 ``test_instructions`` pointed at
``test.json`` while the dataset only ships ``test_3000.json``.
"""

import os
import unittest
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "compose_ucit.yaml"


class UcitFormalConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_all_instruction_paths_exist(self):
        missing = []
        for task in self.config["task_sequence"]:
            for key in ("train_instructions", "test_instructions"):
                path = Path(task[key])
                if not path.is_file():
                    missing.append("task {} {}: {}".format(
                        task["task_id"], key, path))
        self.assertEqual(
            missing, [],
            "formal config references missing instruction files: {}".format(missing),
        )

    def test_six_tasks_in_order(self):
        names = [task["name"] for task in self.config["task_sequence"]]
        self.assertEqual(
            names, ["ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR", "Flickr30k"]
        )
        self.assertEqual([t["task_id"] for t in self.config["task_sequence"]], list(range(6)))

    def test_algorithm_contract_values(self):
        data = self.config["data"]
        self.assertEqual(data["seed"], 42)
        self.assertEqual(self.config["residual"]["tau_res"], 2.0)
        self.assertEqual(self.config["clustering"]["silhouette_threshold"], 0.15)
        self.assertEqual(self.config["clustering"]["random_seed"], 42)
        router = self.config["router"]
        self.assertEqual(router["tau_none"], 0.5)
        self.assertEqual(router["tau_second"], 0.5)
        self.assertEqual(router["max_active_experts"], 2)
        self.assertEqual(router["key_mode"], "learnable")
        keys = self.config["key_learning"]
        self.assertEqual(keys["learning_rate"], 3.0e-4)
        self.assertEqual(keys["epochs"], 50)
        self.assertEqual(keys["temperature"], 0.07)
        self.assertEqual(keys["margin"], 0.3)
        self.assertEqual(keys["lambda_old"], 0.5)
        self.assertEqual(keys["lambda_div"], 0.1)
        self.assertEqual(keys["seed"], 42)
        rms = self.config["rms"]
        self.assertEqual(rms["epsilon"], 1.0e-8)
        self.assertEqual(rms["kappa_min"], 0.25)
        self.assertEqual(rms["kappa_max"], 4.0)
        self.assertEqual(self.config["lora"]["rank"], 8)
        self.assertEqual(self.config["lora"]["alpha"], 16.0)
        training = self.config["training"]
        self.assertEqual(training["learning_rate"], 2.0e-4)
        self.assertTrue(training["gradient_checkpointing"])
        self.assertEqual(training["num_train_epochs"], 3)


if __name__ == "__main__":
    unittest.main()
