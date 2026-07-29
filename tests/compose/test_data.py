import unittest

import torch

from compose.train.data import DataCollatorForSupervisedDataset
from llava.constants import IGNORE_INDEX


class DummyTokenizer:
    pad_token_id = 0
    model_max_length = 4


class DataCollatorTest(unittest.TestCase):
    def test_records_supervised_token_summary_after_truncation(self):
        collator = DataCollatorForSupervisedDataset(DummyTokenizer())
        batch = collator([
            {
                "sample_id": "valid-a",
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([IGNORE_INDEX, 2, 3]),
            },
            {
                "sample_id": "valid-b",
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "labels": torch.tensor([IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5]),
            },
        ])
        self.assertEqual(tuple(batch["labels"].shape), (2, 4))
        self.assertEqual(
            collator.supervision_summary(),
            {"samples": 2, "min": 2, "mean": 2.0, "max": 2, "zero_supervision": 0},
        )

    def test_rejects_zero_supervision_created_by_truncation(self):
        collator = DataCollatorForSupervisedDataset(DummyTokenizer())
        instance = {
            "sample_id": "truncated-answer",
            "input_ids": torch.tensor([1, 2, 3, 4, 5]),
            "labels": torch.tensor(
                [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 5]
            ),
        }
        with self.assertRaisesRegex(ValueError, "truncated-answer") as caught:
            collator([instance])
        self.assertIn("batch_position", str(caught.exception))
        self.assertIn("original_length", str(caught.exception))
        self.assertEqual(collator.supervision_summary()["zero_supervision"], 1)
