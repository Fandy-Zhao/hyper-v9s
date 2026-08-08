"""Test B (spec §27): Task 0 runs against an empty registry -- a legal cold
start. There is no rejected/rebootstrap concept: every training sample is
residual material for the first task."""

import tempfile
import unittest
from pathlib import Path

from compose.experts.registry import ExpertRegistry
from compose.expansion.residual import (
    RESIDUAL_REASON_COLD_START,
    build_residual_records,
    is_residual,
    should_create_experts,
    write_residual_split,
)


def _record(sample_id, teacher_loss, empty_loss, teacher_set=(0,), candidate_experts=()):
    return {
        "sample_id": sample_id,
        "task_id": 0,
        "teacher_set": teacher_set,
        "teacher_loss": teacher_loss,
        "empty_loss": empty_loss,
        "candidate_experts": candidate_experts,
        "teacher_multi_hot": {},
    }


class Task0EmptyRegistryTest(unittest.TestCase):
    def test_fresh_registry_is_empty(self):
        registry = ExpertRegistry()
        self.assertEqual(registry.next_expert_id(), 0)
        self.assertEqual(registry.pool_version, 1)
        self.assertEqual(registry.get_active_experts(), [])
        self.assertEqual(registry.active_lifecycle_ids(), ())
        self.assertEqual(registry.get_rejected_candidates(), [])

    def test_empty_registry_round_trip_stays_empty(self):
        registry = ExpertRegistry()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expert_registry.json"
            registry.save_json(path)
            restored = ExpertRegistry.load_json(path)
        self.assertEqual(restored.next_expert_id(), 0)
        self.assertEqual(restored.pool_version, 1)

    def test_next_expert_id_is_consuming_and_persisted(self):
        # Regression: next_expert_id() must allocate distinct ids when
        # called repeatedly before any commit (one per cluster), and the
        # counter must survive a save/load round trip.
        registry = ExpertRegistry()
        first = registry.next_expert_id()
        second = registry.next_expert_id()
        third = registry.next_expert_id()
        self.assertEqual((first, second, third), (0, 1, 2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expert_registry.json"
            registry.save_json(path)
            restored = ExpertRegistry.load_json(path)
        self.assertEqual(restored.next_expert_id(), 3)
        self.assertNotEqual(restored.next_expert_id(), first)
        self.assertNotEqual(restored.next_expert_id(), second)

    def test_cold_start_is_residual_not_rejected(self):
        residual, reason = is_residual(0.0, tau_res=2.0, empty_pool=True)
        self.assertTrue(residual)
        self.assertEqual(reason, RESIDUAL_REASON_COLD_START)

    def test_all_samples_residual_under_empty_pool(self):
        records = [
            _record("a", teacher_loss=0.1, empty_loss=0.2),
            _record("b", teacher_loss=3.0, empty_loss=3.1),
            _record("c", teacher_loss=1.5, empty_loss=1.6),
        ]
        reuse, residual = build_residual_records(records, tau_res=2.0, split="train", empty_pool=True)
        self.assertEqual(reuse, [])
        self.assertEqual(len(residual), 3)
        for record in residual:
            self.assertEqual(record.residual_reason, RESIDUAL_REASON_COLD_START)
            self.assertFalse(record.retrieval_diagnostic)

    def test_count_gate_never_forces_experts(self):
        self.assertFalse(should_create_experts(7, 8))
        self.assertTrue(should_create_experts(8, 8))

    def test_cold_start_split_written(self):
        records = [_record("a", teacher_loss=0.1, empty_loss=0.2)]
        _, residual = build_residual_records(records, tau_res=2.0, split="train", empty_pool=True)
        with tempfile.TemporaryDirectory() as directory:
            summary = write_residual_split(
                Path(directory), [], residual, tau_res=2.0,
                summary={"task_id": 0, "empty_pool": True},
            )
            self.assertEqual(summary["residual_count"], 1)
            self.assertEqual(summary["cold_start_count"], 1)
            self.assertEqual(summary["reuse_count"], 0)
            self.assertTrue((Path(directory) / "residual" / "residual.json").is_file())


if __name__ == "__main__":
    unittest.main()
