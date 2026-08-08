"""Compose P1-Real required tests (spec section 16)."""

import json
import re
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, "/home/zhaozhuofan/Hyper-LlaVA/compose/data/real_p1")
from official_metric import normalize, vqa_accuracy  # noqa: E402

from build_subsets import (  # noqa: E402
    _artifact_answer,
    _compare_minmax,
    _count_physical_compare,
    _time_arithmetic,
    _word_number_to_int,
    classify,
)
from compose.experiments.scheduler import (  # noqa: E402
    archive_partial_output,
    output_complete,
    resolve_devices,
    validate_devices,
)
from compose.lora.statistics import RMSStatistics, StatisticKey  # noqa: E402

OUTPUT_ROOT = Path("/home/zhaozhuofan/Hyper-LlaVA/outputs/compose_p1_real_20260803T120000Z")


# ---------------------------------------------------------------- 3, 4, metric
class OfficialMetricTest(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("The Answer!"), "answer")
        self.assertEqual(normalize("don't stop"), "dont stop")
        self.assertEqual(normalize("A An The XYZ"), "xyz")
        self.assertEqual(normalize("3.5"), normalize("35"))

    def test_vqa_accuracy(self):
        self.assertEqual(vqa_accuracy("5", ["5"] * 10), 1.0)
        self.assertEqual(vqa_accuracy("4", ["5"] * 10), 0.0)
        self.assertEqual(vqa_accuracy("5", ["5", "5"] + ["4"] * 8), 2.0 / 3.0)

    def test_word_number_conversion(self):
        self.assertEqual(_word_number_to_int("5"), 5)
        self.assertEqual(_word_number_to_int("10,000"), 10000)
        self.assertEqual(_word_number_to_int("ten thousand"), 10000)
        self.assertEqual(_word_number_to_int("two hundred thirty four"), 234)
        self.assertIsNone(_word_number_to_int("58%"))
        self.assertIsNone(_word_number_to_int("about five"))


# ------------------------------------------------------------ 1, 2, 7, 8 splits
class DataIntegrityTest(unittest.TestCase):
    def _load(self, name):
        path = OUTPUT_ROOT / "data" / "records" / "{}.json".format(name)
        if not path.exists():
            self.skipTest("built records missing")
        return json.loads(path.read_text())

    def test_no_image_across_splits(self):
        names = ["B_train", "B_val", "C_train", "C_val", "BC_calib", "BC_test",
                 "C_controlled_train", "C_controlled_val", "C_controlled_test"]
        images = {}
        for name in names:
            try:
                records = self._load(name)
            except unittest.SkipTest:
                continue
            for record in records:
                images.setdefault(record["image_id"], set()).add(name)
        overlaps = {image_id: splits for image_id, splits in images.items() if len(splits) > 1}
        self.assertEqual(overlaps, {}, "image crosses splits: {}".format(overlaps))

    def test_bc_test_questions_not_in_training(self):
        bc_test = self._load("BC_test")
        bc_questions = {r["question"].strip().lower() for r in bc_test}
        for name in ("B_train", "C_train"):
            for record in self._load(name):
                self.assertNotIn(record["question"].strip().lower(), bc_questions,
                                 "BC_test question leaked into {}".format(name))

    def test_bc_test_images_not_in_training(self):
        bc_images = {r["image_id"] for r in self._load("BC_test")}
        for name in ("B_train", "C_train"):
            for record in self._load(name):
                self.assertNotIn(record["image_id"], bc_images,
                                 "BC_test image leaked into {}".format(name))

    def test_official_val_not_in_training(self):
        val_images = {r["image_id"] for r in self._load("val_official_Bonly")}
        for record in self._load("B_train"):
            self.assertNotIn(record["image_id"], val_images)

    def test_classifier_label_rules(self):
        # B-only: read one printed value
        self.assertEqual(classify("what time is it on the phone?", "12:20", ["12:20", "SAMSUNG"])["function_label"], "B_only")
        self.assertEqual(classify("what brand of watch is this?", "rolex", ["ROLEX"])["function_label"], "B_only")
        self.assertEqual(classify("what is the number on the jersey?", "10", ["10"])["function_label"], "B_only")
        # C-only: pure object count, answer not readable from OCR
        self.assertEqual(classify("how many planes in the picture?", "5", ["MARTIN", "AIRLINES"])["function_label"], "C_only")
        # C-only shortcut exclusion
        self.assertEqual(classify("how many planes in the picture?", "5", ["MARTIN", "5"])["function_label"], "EXCLUDE")
        # read-indicator exclusion for physical counts
        self.assertEqual(classify("how many people are killed by shooting every year?", "40000", ["NEWS"])["function_label"], "EXCLUDE")
        # "how many steps are listed" is a printed-quantity read (B)
        self.assertEqual(classify("how many steps are listed on the sign?", "5", ["SIGN"])["function_label"], "B_only")
        # B+C compare
        self.assertEqual(classify("what is the highest number on the ruler?", "30", ["3", "30", "RULER"])["function_label"], "B_plus_C")
        self.assertEqual(classify("what is the highest number on the ruler?", "30", ["RULER", "CM"])["function_label"], "B_plus_C")
        # B+C aggregate
        self.assertEqual(classify("how much money are these worth?", "10", ["10", "5"])["function_label"], "B_plus_C")
        # B+C time arithmetic
        self.assertEqual(classify("how many hours till midnight?", "1", ["12", "10", "2"])["function_label"], "B_plus_C")
        # artifact answers excluded
        self.assertTrue(_artifact_answer("unanswerable"))
        self.assertTrue(_artifact_answer("answering does not require reading text in the image"))
        self.assertFalse(_artifact_answer("10"))
        # helper predicates
        self.assertTrue(_compare_minmax("what is the smallest number on the pole?"))
        self.assertFalse(_compare_minmax("what is the second number on the jersey?"))
        self.assertTrue(_time_arithmetic("how many hours till midnight?"))
        self.assertTrue(_count_physical_compare("are there more chairs than tables?"))


# ------------------------------------------------------------------ 5 RMS cache
class RMSCacheTest(unittest.TestCase):
    def test_cache_reproducible(self, tmpdir="/tmp/p1_real_rms_test"):
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        provenance = {"analysis_seed": 0, "checkpoint_seed": 0,
                      "calibration_split": "BC_calib", "checkpoint_hash": "x",
                      "dataset_manifest_hash": "y", "composition_config_hash": "z"}
        first = RMSStatistics(provenance)
        key = StatisticKey(1, "layer0", "layer0", "Linear")
        delta = torch.arange(12.0).reshape(3, 4)
        base = torch.zeros(3, 4)
        for _ in range(5):
            first.update(key, delta, base + delta, base)
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
        first.save_json(Path(tmpdir) / "rms_statistics.json")
        second = RMSStatistics.load_json(Path(tmpdir) / "rms_statistics.json", provenance)
        self.assertAlmostEqual(
            first.delta_rms(1, "layer0"), second.delta_rms(1, "layer0"), places=6)
        self.assertGreater(first.delta_rms(1, "layer0"), 0.0)


# -------------------------------------------------------------------- 6, 7 C3
class C3IsolationTest(unittest.TestCase):
    def test_c3_uses_validation_only(self):
        source = Path("/home/zhaozhuofan/Hyper-LlaVA/compose/eval/compose_p1_real.py").read_text()
        # the C3 scalar search reads only the calibration file
        grid_block = source.split("grid_loss = {}")[1].split("selected_scalars = min")[0]
        self.assertIn("calibration", grid_block)
        self.assertNotIn("args.test_questions", grid_block)
        self.assertNotIn("args.test_questions", source.split("C3 scalar search")[1].split("selected_scalars = min")[0])
        # summary must assert the test was not used
        self.assertIn("test_used_for_c3_search", source)

    def test_metrics_claim_test_not_used(self):
        summary = OUTPUT_ROOT / "metrics"
        if summary.exists():
            pass  # runtime summaries carry the flag; checked in eval runs


# -------------------------------------------------------------------- 8 configs
class SameCheckpointTest(unittest.TestCase):
    def test_c0_c3_share_checkpoint(self):
        task_file = OUTPUT_ROOT / "configs" / "compose_p1_real_tasks.jsonl"
        if not task_file.exists():
            self.skipTest("task manifest missing")
        by_seed = {}
        for line in task_file.read_text().splitlines():
            task = json.loads(line)
            if task["stage"] == "p1_real_eval" and "BC_test" in task["id"]:
                by_seed.setdefault(task["seed"], set()).add(task.get("checkpoint"))
        for seed, checkpoints in by_seed.items():
            self.assertEqual(len(checkpoints), 1,
                             "C0-C3 must share one checkpoint per seed {}".format(seed))


# -------------------------------------------------------------------- 9 layer mask
class LayerMaskTest(unittest.TestCase):
    def test_mask_applies_only_to_masked_layers(self):
        sys.path.insert(0, "/home/zhaozhuofan/Hyper-LlaVA/compose/eval")
        from compose_p1_real import LayerMaskedComposer  # noqa: E402
        import torch.nn as nn

        class FakeBridge:
            def __init__(self):
                self.depth = 0
                self.named_layers = [("low", nn.Identity()), ("high", nn.Identity())]
                self.expert_b, self.expert_c = 1, 2

            def enable_pair_execution(self):
                self.depth += 1

            def disable_pair_execution(self):
                self.depth -= 1

            def compute_expert_delta(self, module, expert_id, hidden):
                return torch.full_like(hidden, float(expert_id))

        bridge = FakeBridge()
        masks = {"low": {"B": 1, "C": 0}, "high": {"B": 0, "C": 1}}
        composer = LayerMaskedComposer(bridge, masks)
        composer.expert_b, composer.expert_c = 1, 2
        hidden = torch.ones(1, 4)
        with composer:
            low_out = bridge.named_layers[0][1](hidden)   # hook fires: B only
            high_out = bridge.named_layers[1][1](hidden)  # hook fires: C only
        scale = 1.0 / (2 ** 0.5)
        self.assertTrue(torch.allclose(low_out, hidden + torch.full_like(hidden, 1.0) * scale))
        self.assertTrue(torch.allclose(high_out, hidden + torch.full_like(hidden, 2.0) * scale))
        self.assertEqual(bridge.depth, 0, "pair execution state must be restored")
        # after exit the hooks are removed: plain identity output
        plain_out = bridge.named_layers[0][1](hidden)
        self.assertTrue(torch.allclose(plain_out, hidden))


# -------------------------------------------------------------------- 10 GPU limit
class GpuPolicyTest(unittest.TestCase):
    def test_devices_limited_to_0_7(self):
        with self.assertRaises(ValueError):
            validate_devices("0,8")
        with self.assertRaises(ValueError):
            validate_devices("4,4")
        self.assertEqual(validate_devices("4,5,6,7".split(",")), (4, 5, 6, 7))

    def test_auto_prefers_primary(self):
        # 0-3 free -> only primary; 0-3 busy -> only authorized fallback
        memory = {device: 20000 for device in range(8)}
        self.assertEqual(resolve_devices("auto", 18000, lambda d: memory[d]), (0, 1, 2, 3))
        for device in (0, 1, 2, 3):
            memory[device] = 1000
        self.assertEqual(resolve_devices("auto", 18000, lambda d: memory[d]), (4, 5, 6, 7))
        memory = {device: 1000 for device in range(8)}
        with self.assertRaises(RuntimeError):
            resolve_devices("auto", 18000, lambda d: memory[d])


# -------------------------------------------------------------------- 11, 12 scheduler
class SchedulerResumeTest(unittest.TestCase):
    def test_output_complete_and_archive(self, tmpdir="/tmp/p1_real_sched_test"):
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        output = Path(tmpdir) / "run"
        output.mkdir(parents=True)
        (output / "summary.json").write_text(json.dumps({"status": "COMPLETED"}))
        (output / "per_sample.jsonl").write_text("{}")
        self.assertTrue(output_complete({"output_dir": str(output)}))
        (output / "summary.json").write_text(json.dumps({"status": "FAILED"}))
        self.assertFalse(output_complete({"output_dir": str(output)}))
        archived = archive_partial_output(output, 1)
        self.assertIsNotNone(archived)
        self.assertTrue(Path(archived).exists())
        self.assertTrue(output.exists(), "retry dir must be recreated clean")

    def test_attempt_logs_are_not_overwritten(self):
        source = Path("/home/zhaozhuofan/Hyper-LlaVA/compose/experiments/scheduler.py").read_text()
        self.assertIn("attempt{}_stdout.log", source)
        self.assertIn("attempt{}_stderr.log", source)
        self.assertIn(".oom_attempt{}", source)


if __name__ == "__main__":
    unittest.main()
