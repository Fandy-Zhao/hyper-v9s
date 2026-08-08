"""Residual boundary tests (pre-formal validation §10).

The capability-residual judgment is exactly

    is_residual = old_teacher_loss > tau_res

with ``<=`` being reuse and no gain-floor anywhere. A historical Top-M
recall miss is recorded as a retrieval diagnostic and must never be
wrapped into "need a new capability expert".
"""

import unittest

from compose.expansion.residual import (
    RESIDUAL_REASON_ABOVE_TAU,
    RESIDUAL_REASON_BELOW_TAU,
    RESIDUAL_REASON_RECALL_MISS,
    build_residual_records,
    is_residual,
)


def _record(sample_id, teacher_loss, empty_loss=None, candidate_experts=()):
    return {
        "sample_id": sample_id,
        "task_id": 1,
        "teacher_set": tuple(candidate_experts) or (0,),
        "teacher_loss": teacher_loss,
        "empty_loss": empty_loss if empty_loss is not None else teacher_loss,
        "candidate_experts": candidate_experts,
    }


class ResidualBoundaryTest(unittest.TestCase):
    def test_loss_below_tau_is_reuse(self):
        residual, reason = is_residual(1.4, tau_res=1.5)
        self.assertFalse(residual)
        self.assertEqual(reason, RESIDUAL_REASON_BELOW_TAU)

    def test_loss_equal_tau_is_reuse(self):
        residual, reason = is_residual(1.5, tau_res=1.5)
        self.assertFalse(residual)
        self.assertEqual(reason, RESIDUAL_REASON_BELOW_TAU)

    def test_loss_epsilon_above_tau_is_residual(self):
        residual, reason = is_residual(1.5 + 1e-9, tau_res=1.5)
        self.assertTrue(residual)
        self.assertEqual(reason, RESIDUAL_REASON_ABOVE_TAU)

    def test_recall_miss_is_diagnostic_not_capability(self):
        """A recall miss is recorded separately and must not create a
        capability expert, even when the loss is above tau_res."""
        residual, reason = is_residual(9.0, tau_res=1.5, recall_covered=False)
        self.assertFalse(residual)
        self.assertEqual(reason, RESIDUAL_REASON_RECALL_MISS)

    def test_build_splits_above_and_below(self):
        records = [
            _record("r-low", teacher_loss=0.5),
            _record("r-eq", teacher_loss=1.5),
            _record("r-high", teacher_loss=2.0),
        ]
        reuse, residual = build_residual_records(records, tau_res=1.5, split="train")
        self.assertEqual([r.sample_id for r in reuse], ["r-low", "r-eq"])
        self.assertEqual([r.sample_id for r in residual], ["r-high"])
        self.assertEqual(residual[0].residual_reason, RESIDUAL_REASON_ABOVE_TAU)
        self.assertFalse(residual[0].retrieval_diagnostic)

    def test_recall_miss_record_is_marked_and_not_trainable(self):
        records = [_record("r-miss", teacher_loss=9.0, candidate_experts=())]
        reuse, residual = build_residual_records(
            records, tau_res=1.5, split="train", top_m_covered=[False]
        )
        self.assertEqual(reuse, [])
        self.assertEqual(len(residual), 1)
        self.assertTrue(residual[0].retrieval_diagnostic)
        self.assertEqual(residual[0].residual_reason, RESIDUAL_REASON_RECALL_MISS)


if __name__ == "__main__":
    unittest.main()
