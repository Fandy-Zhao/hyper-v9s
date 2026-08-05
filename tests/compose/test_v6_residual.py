"""V6 Stage E5: residual buffers, reuse/residual splitting and gates."""

import tempfile
import unittest
from pathlib import Path

from compose.expansion.v6_residual import (
    RESIDUAL_REASON_EMPTY_TEACHER,
    RESIDUAL_REASON_GAIN_FLOOR,
    RESIDUAL_REASON_RECALL_MISS,
    V6ResidualBuffer,
    V6ResidualRecord,
    build_residual_records,
    is_residual,
    router_calibration_subset,
    should_create_candidates,
)
from compose.teacher.types import AnswerNLL
from compose.teacher.v6_teacher import V6TeacherRecord


def _teacher(sample_id, teacher_set, empty_loss, teacher_loss, candidates=(0, 1)):
    return V6TeacherRecord(
        sample_id=sample_id,
        task_id=1,
        pool_version=1,
        router_version="r1",
        candidate_experts=tuple(candidates),
        empty_loss=empty_loss,
        single_losses={0: 1.0, 1: 1.0},
        pair_losses={},
        best_single=(teacher_set[0],) if teacher_set else (),
        best_pair=tuple(teacher_set) if len(teacher_set) == 2 else (),
        pair_gain=None,
        teacher_set=tuple(teacher_set),
        teacher_loss=teacher_loss,
        teacher_multi_hot={int(e): 1 for e in teacher_set},
        cache_key="ck-" + sample_id,
    )


class ResidualJudgmentTest(unittest.TestCase):
    def test_judgment_uses_answer_teacher_not_router(self):
        # Teacher picked expert 0 but gain 0.05 below floor 0.1: residual.
        record = _teacher("s1", teacher_set=(0,), empty_loss=1.0, teacher_loss=0.95)
        result, reason = is_residual(record, min_old_gain=0.1, recall_covered=True)
        self.assertTrue(result)
        self.assertEqual(reason, RESIDUAL_REASON_GAIN_FLOOR)

    def test_empty_teacher_never_residual(self):
        record = _teacher("s2", teacher_set=(), empty_loss=1.0, teacher_loss=1.0)
        result, reason = is_residual(record, min_old_gain=0.0, recall_covered=True)
        self.assertFalse(result)
        self.assertEqual(reason, RESIDUAL_REASON_EMPTY_TEACHER)

    def test_recall_miss_is_not_residual(self):
        record = _teacher("s3", teacher_set=(2,), empty_loss=1.0, teacher_loss=0.95)
        result, reason = is_residual(record, min_old_gain=0.1, recall_covered=False)
        self.assertFalse(result)
        self.assertEqual(reason, RESIDUAL_REASON_RECALL_MISS)

    def test_sufficient_teacher_is_reuse_not_residual(self):
        record = _teacher("s4", teacher_set=(0,), empty_loss=1.0, teacher_loss=0.8)
        result, reason = is_residual(record, min_old_gain=0.1, recall_covered=True)
        self.assertFalse(result)
        self.assertEqual(reason, "teacher_sufficient")


class BuildResidualRecordsTest(unittest.TestCase):
    def test_split_into_reuse_and_residual(self):
        records = [
            _teacher("a", teacher_set=(0,), empty_loss=1.0, teacher_loss=0.8),  # reuse
            _teacher("b", teacher_set=(0,), empty_loss=1.0, teacher_loss=0.95),  # residual
            _teacher("c", teacher_set=(), empty_loss=1.0, teacher_loss=1.0),  # skip
        ]
        reuse, residual = build_residual_records(
            records, query_feature_path="/feat/train.pt",
            min_old_gain=0.1, top_m_covered=[True, True, True], split="train",
        )
        self.assertEqual([record.sample_id for record in reuse], ["a"])
        self.assertEqual([record.sample_id for record in residual], ["b"])
        self.assertEqual(residual[0].residual_reason, RESIDUAL_REASON_GAIN_FLOOR)
        self.assertEqual(residual[0].split, "train")
        self.assertAlmostEqual(residual[0].old_gain, 0.05, places=6)

    def test_recall_miss_recorded_with_reason(self):
        records = [_teacher("b", teacher_set=(2,), empty_loss=1.0, teacher_loss=0.95)]
        reuse, residual = build_residual_records(
            records, query_feature_path="/f", min_old_gain=0.1,
            top_m_covered=[False], split="train",
        )
        self.assertEqual(reuse, [])
        self.assertEqual(residual[0].residual_reason, RESIDUAL_REASON_RECALL_MISS)

    def test_length_mismatch_rejected(self):
        records = [_teacher("a", teacher_set=(0,), empty_loss=1.0, teacher_loss=0.8)]
        with self.assertRaisesRegex(ValueError, "align"):
            build_residual_records(
                records, "/f", min_old_gain=0.1, top_m_covered=[True, True],
                split="train",
            )


class V6ResidualBufferTest(unittest.TestCase):
    def _record(self, sample_id, split="train"):
        return V6ResidualRecord(
            sample_id=sample_id,
            task_id=1,
            old_teacher_set=(0,),
            empty_loss=1.0,
            old_teacher_loss=0.95,
            old_gain=0.05,
            residual_reason=RESIDUAL_REASON_GAIN_FLOOR,
            query_feature_path="/f",
            split=split,
        )

    def test_add_and_dedup(self):
        buffer = V6ResidualBuffer()
        self.assertTrue(buffer.add(self._record("a")))
        self.assertFalse(buffer.add(self._record("a")))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            buffer.add(
                V6ResidualRecord(
                    sample_id="a", task_id=1, old_teacher_set=(0,),
                    empty_loss=2.0, old_teacher_loss=0.95, old_gain=1.05,
                    residual_reason=RESIDUAL_REASON_GAIN_FLOOR,
                    query_feature_path="/f", split="train",
                )
            )

    def test_supports_validation_split(self):
        buffer = V6ResidualBuffer()
        buffer.add(self._record("a", split="train"))
        buffer.add(self._record("b", split="val"))
        self.assertEqual(
            [r.sample_id for r in buffer.split_records("train")], ["a"]
        )
        self.assertEqual(
            [r.sample_id for r in buffer.split_records("val")], ["b"]
        )

    def test_shard_round_trip_and_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            left = V6ResidualBuffer()
            left.add(self._record("a"))
            left.write_shard(str(directory / "left.json"), rank=0)
            right = V6ResidualBuffer()
            right.add(self._record("b"))
            right.write_shard(str(directory / "right.json"), rank=1)
            output = directory / "merged.json"
            count = V6ResidualBuffer.merge_shards(
                [str(directory / "left.json"), str(directory / "right.json")],
                str(output),
                expected_sample_ids=["a", "b"],
            )
            self.assertEqual(count, 2)
            merged, rank = V6ResidualBuffer.load_shard(str(output))
            self.assertEqual(merged.sample_ids, ("a", "b"))
            self.assertEqual(rank, 0)

    def test_invalid_split_rejected(self):
        with self.assertRaisesRegex(ValueError, "split"):
            self._record("a", split="test")


class ResidualGatesTest(unittest.TestCase):
    def test_candidates_created_only_above_min(self):
        self.assertFalse(should_create_candidates(5, 10))
        self.assertTrue(should_create_candidates(10, 10))
        self.assertTrue(should_create_candidates(20, 10))

    def test_calibration_subset_is_seeded_and_bounded(self):
        records = [
            V6ResidualRecord(
                sample_id="s{}".format(index), task_id=1, old_teacher_set=(0,),
                empty_loss=1.0, old_teacher_loss=0.95, old_gain=0.05,
                residual_reason=RESIDUAL_REASON_GAIN_FLOOR,
                query_feature_path="/f",
                split=("train" if index % 2 == 0 else "val"),
            )
            for index in range(8)
        ]
        first = router_calibration_subset(records, capacity=4, seed=42)
        second = router_calibration_subset(records, capacity=4, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        # Stratified across splits.
        splits = set()
        for sample_id in first:
            for record in records:
                if record.sample_id == sample_id:
                    splits.add(record.split)
        self.assertEqual(splits, {"train", "val"})


if __name__ == "__main__":
    unittest.main()
