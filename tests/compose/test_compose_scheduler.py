import json
import tempfile
import unittest
from pathlib import Path

from compose.experiments.scheduler import archive_partial_output, output_complete, resolve_devices, validate_devices


class ComposeSchedulerTest(unittest.TestCase):
    def test_scheduler_validates_devices_and_uses_fallback_only_when_primary_busy(self):
        self.assertEqual(validate_devices((0, 1, 2, 3)), (0, 1, 2, 3))
        self.assertEqual(validate_devices((4, 5, 6, 7)), (4, 5, 6, 7))
        with self.assertRaisesRegex(ValueError, "0-7"):
            validate_devices((8,))
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_devices((0, 0))
        primary_free = {device: (20000 if device == 2 else 10000) for device in range(8)}
        self.assertEqual(resolve_devices("auto", 18000, primary_free.__getitem__), (2,))
        fallback_free = {device: (24000 if device >= 4 else 10000) for device in range(8)}
        self.assertEqual(resolve_devices("auto", 18000, fallback_free.__getitem__), (4, 5, 6, 7))

    def test_resume_only_accepts_complete_nonempty_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            task = {"output_dir": str(output), "completion_files": ["summary.json", "per_sample.jsonl"]}
            (output / "summary.json").write_text(json.dumps({"status": "COMPLETED"}))
            (output / "per_sample.jsonl").write_text("{}\n")
            self.assertTrue(output_complete(task))
            (output / "summary.json").write_text(json.dumps({"status": "FAILED"}))
            self.assertFalse(output_complete(task))

    def test_oom_partial_output_is_archived_before_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "job"
            output.mkdir()
            (output / "rms_statistics.json").write_text("{}\n")
            archived = Path(archive_partial_output(output, 1))
            self.assertEqual((archived / "rms_statistics.json").read_text(), "{}\n")
            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
