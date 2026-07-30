import json
import tempfile
import unittest
from pathlib import Path

from compose.data.controlled_functional import FUNCTIONS, generate


class ControlledFunctionalDataTest(unittest.TestCase):
    def test_generation_is_deterministic_and_split_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first" / "controlled_v1"
            second = Path(directory) / "second" / "controlled_v1"
            sizes = {"train": 8, "val": 3, "test": 4}
            left = generate(str(first), seed=17, split_sizes=sizes)
            right = generate(str(second), seed=17, split_sizes=sizes)
            self.assertEqual(left["files"], right["files"])
            for function_name in FUNCTIONS:
                split_ids = []
                for split, count in sizes.items():
                    rows = json.loads((first / "instructions" / function_name / (split + ".json")).read_text())
                    self.assertEqual(len(rows), count)
                    split_ids.append({row["template_id"] for row in rows})
                    self.assertTrue(all(row["image"].startswith("controlled_v1/images/") for row in rows))
                self.assertFalse(split_ids[0] & split_ids[1])
                self.assertFalse(split_ids[0] & split_ids[2])
                self.assertFalse(split_ids[1] & split_ids[2])


if __name__ == "__main__":
    unittest.main()
