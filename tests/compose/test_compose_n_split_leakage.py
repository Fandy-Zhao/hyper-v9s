"""Test N (spec §27): split leakage guards -- validation residuals never
enter cluster training; recall-miss diagnostics are recorded but never
become expert material; train/val residual sample ids are disjoint."""

import json
import tempfile
import unittest
from pathlib import Path

from compose.expansion.residual import (
    RESIDUAL_REASON_RECALL_MISS,
    build_residual_records,
    write_residual_split,
)


def _record(sample_id, teacher_loss, empty_loss, teacher_set=(0,), candidate_experts=(0,)):
    return {
        "sample_id": sample_id,
        "task_id": 1,
        "teacher_set": teacher_set,
        "teacher_loss": teacher_loss,
        "empty_loss": empty_loss,
        "candidate_experts": candidate_experts,
        "teacher_multi_hot": {},
    }


class SplitLeakageTest(unittest.TestCase):
    def test_train_and_val_residual_sets_are_disjoint(self):
        train_records = [
            _record("train-{:03d}".format(index), teacher_loss=3.0, empty_loss=3.1)
            for index in range(10)
        ]
        val_records = [
            _record("val-{:03d}".format(index), teacher_loss=3.0, empty_loss=3.1)
            for index in range(5)
        ]
        _, train_residual = build_residual_records(
            train_records, tau_res=2.0, split="train"
        )
        _, val_residual = build_residual_records(val_records, tau_res=2.0, split="val")
        train_ids = {record.sample_id for record in train_residual}
        val_ids = {record.sample_id for record in val_residual}
        self.assertTrue(train_ids.isdisjoint(val_ids))
        # Only train residuals are eligible cluster input.
        cluster_input = [record for record in train_residual if not record.retrieval_diagnostic]
        self.assertEqual(
            {record.sample_id for record in cluster_input}, train_ids
        )

    def test_validation_split_never_feeds_training_manifest(self):
        train_records = [
            _record("train-{:03d}".format(index), teacher_loss=3.0, empty_loss=3.1)
            for index in range(6)
        ]
        val_records = [
            _record("val-{:03d}".format(index), teacher_loss=2.5, empty_loss=2.6)
            for index in range(4)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, train_residual = build_residual_records(
                train_records, tau_res=2.0, split="train"
            )
            _, val_residual = build_residual_records(
                val_records, tau_res=2.0, split="val"
            )
            write_residual_split(root / "train", [], train_residual, tau_res=2.0)
            write_residual_split(root / "val", [], val_residual, tau_res=2.0)
            train_payload = json.loads(
                (root / "train" / "residual" / "residual.json").read_text(encoding="utf-8")
            )
            val_payload = json.loads(
                (root / "val" / "residual" / "residual.json").read_text(encoding="utf-8")
            )
        train_ids = {record["sample_id"] for record in train_payload}
        val_ids = {record["sample_id"] for record in val_payload}
        self.assertTrue(train_ids.isdisjoint(val_ids))
        for record in train_payload:
            self.assertEqual(record["split"], "train")
        for record in val_payload:
            self.assertEqual(record["split"], "val")

    def test_recall_miss_is_diagnostic_never_expert_material(self):
        records = [
            _record("miss-0", teacher_loss=0.5, empty_loss=2.0, teacher_set=(3,), candidate_experts=(0,)),
            _record("hit-0", teacher_loss=3.0, empty_loss=3.1, teacher_set=(0,), candidate_experts=(0,)),
        ]
        coverage = [False, True]  # Top-M missed expert 3 for the first sample
        reuse, residual = build_residual_records(
            records, tau_res=2.0, split="train", top_m_covered=coverage
        )
        # The recall miss is recorded as a diagnostic residual...
        self.assertEqual(len(residual), 2)
        diagnostics = [
            record for record in residual if record.retrieval_diagnostic
        ]
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].sample_id, "miss-0")
        self.assertEqual(diagnostics[0].residual_reason, RESIDUAL_REASON_RECALL_MISS)
        # ...but never fed to clustering/training.
        expert_material = [
            record for record in residual if not record.retrieval_diagnostic
        ]
        self.assertEqual(
            {record.sample_id for record in expert_material}, {"hit-0"}
        )

    def test_below_tau_samples_are_reused_not_residual(self):
        records = [
            _record("reuse-0", teacher_loss=0.5, empty_loss=2.0, teacher_set=(0,), candidate_experts=(0,)),
            _record("residual-0", teacher_loss=3.0, empty_loss=3.1, teacher_set=(0,), candidate_experts=(0,)),
        ]
        reuse, residual = build_residual_records(records, tau_res=2.0, split="train")
        self.assertEqual({record.sample_id for record in reuse}, {"reuse-0"})
        self.assertEqual({record.sample_id for record in residual}, {"residual-0"})


if __name__ == "__main__":
    unittest.main()
