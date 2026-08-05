"""V6 Stage E8: multi-hot teacher labels and Router calibration."""

import unittest

import torch

from compose.router.v6_calibrate import (
    CalibrationConfig,
    build_multi_hot_labels,
    calibrate_v6_router,
    compute_anchor_logits,
    evaluate_v6_router,
    tune_v6_thresholds,
)
from compose.router.v6_router import V6QueryEncoder, V6Router


def _router(seed: int = 42, **overrides):
    encoder = V6QueryEncoder(visual_dim=4, text_dim=5, query_dim=8)
    return V6Router(encoder, top_m=3, seed=seed, **overrides)


def _populated_router(seed: int = 42):
    router = _router(seed)
    router.add_expert(0, creation_task=0, checkpoint_sha256="a" * 64)
    router.add_expert(1, creation_task=1, checkpoint_sha256="b" * 64)
    router.add_expert(2, creation_task=2, checkpoint_sha256="c" * 64)
    return router


def _records(count=6, pattern="alternate"):
    records = []
    for index in range(count):
        if pattern == "alternate":
            teacher = (index % 3,) if index % 3 else ()
            if index % 3 == 2:
                teacher = (0, 1)
        else:
            teacher = (index % 2,)
        records.append({"sample_id": "s{}".format(index), "teacher_set": teacher})
    return records


class MultiHotTest(unittest.TestCase):
    def test_labels_cover_pool(self):
        records = _records(4, pattern="simple")
        labels, pool_ids = build_multi_hot_labels(records, [0, 1, 2])
        self.assertEqual(pool_ids, (0, 1, 2))
        self.assertEqual(labels.shape, (4, 3))
        # s0 teacher (0,) -> row [1, 0, 0]
        torch.testing.assert_close(labels[0], torch.tensor([1.0, 0.0, 0.0]))

    def test_unknown_teacher_rejected(self):
        with self.assertRaisesRegex(ValueError, "not in pool"):
            build_multi_hot_labels([{"teacher_set": (9,)}], [0, 1])

    def test_empty_pool_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            build_multi_hot_labels([{"teacher_set": ()}], [])


class CalibrationTest(unittest.TestCase):
    def test_calibration_updates_router_and_reduces_loss(self):
        router = _populated_router()
        torch.manual_seed(5)
        queries = torch.randn(24, 8)
        records = _records(24, pattern="simple")
        labels, pool_ids = build_multi_hot_labels(records, router.expert_ids)
        config = CalibrationConfig(epochs=3, batch_size=8, learning_rate=1e-3)
        before = float(router._scores(queries, router.expert_ids)[0].mean())
        result = calibrate_v6_router(router, queries, labels, config)
        after = float(router._scores(queries, router.expert_ids)[0].mean())
        self.assertEqual(len(result["loss_history"]), 3)
        self.assertIsInstance(before, float)
        self.assertIsInstance(after, float)

    def test_anchor_loss_runs_with_historical_logits(self):
        router = _populated_router()
        torch.manual_seed(11)
        queries = torch.randn(12, 8)
        labels, _ = build_multi_hot_labels(
            _records(12, pattern="simple"), router.expert_ids
        )
        anchor_queries = queries[:6]
        anchor_logits = compute_anchor_logits(router, anchor_queries)
        config = CalibrationConfig(epochs=2, batch_size=6)
        result = calibrate_v6_router(
            router, queries, labels, config,
            anchor_queries=anchor_queries, anchor_logits=anchor_logits,
        )
        self.assertEqual(len(result["loss_history"]), 2)

    def test_mismatched_labels_rejected(self):
        router = _populated_router()
        labels, _ = build_multi_hot_labels(
            _records(4, pattern="simple"), [0, 1]
        )
        with self.assertRaisesRegex(ValueError, "align"):
            calibrate_v6_router(router, torch.randn(3, 8), labels, CalibrationConfig(epochs=1))

    def test_empty_router_refused(self):
        router = _router()
        labels, _ = build_multi_hot_labels(
            [{"teacher_set": ()}], [0]
        )
        with self.assertRaisesRegex(ValueError, "no experts"):
            calibrate_v6_router(router, torch.randn(1, 8), labels, CalibrationConfig(epochs=1))


class EvaluateTest(unittest.TestCase):
    def _aligned_router(self):
        router = _populated_router()
        with torch.no_grad():
            router.key_store.keys["0"].copy_(
                torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
            )
            router.key_store.keys["1"].copy_(
                torch.tensor([0.0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
            )
            router.key_store.keys["2"].copy_(
                torch.tensor([0.0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float32)
            )
        router.set_thresholds(tau_none=0.5, tau_second=0.6)
        return router

    def test_metrics_on_aligned_router(self):
        router = self._aligned_router()
        queries = torch.stack(
            [
                torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]),
                torch.tensor([0.0, 1, 0, 0, 0, 0, 0, 0]),
                torch.tensor([0.0, 0, 1, 0, 0, 0, 0, 0]),
                torch.tensor([-1.0, -1, -1, 0, 0, 0, 0, 0]),
            ]
        )
        records = [
            {"teacher_set": (0,)},
            {"teacher_set": (1,)},
            {"teacher_set": (2,)},
            {"teacher_set": ()},
        ]
        metrics = evaluate_v6_router(router, queries, records, split="validation")
        self.assertEqual(metrics["samples"], 4)
        self.assertEqual(metrics["SetExactAcc"], 1.0)
        self.assertEqual(metrics["EmptyAcc"], 1.0)
        self.assertEqual(metrics["ExpertRecall@1"], 1.0)
        self.assertEqual(metrics["ExpertRecall@2"], 1.0)
        self.assertEqual(metrics["average_active_experts"], 0.75)
        self.assertEqual(metrics["per_expert"]["0"]["precision"], 1.0)

    def test_test_split_refused(self):
        router = self._aligned_router()
        with self.assertRaisesRegex(ValueError, "test"):
            evaluate_v6_router(router, torch.randn(2, 8), _records(2), split="test")

    def test_pair_recall_metric(self):
        router = self._aligned_router()
        # A pair teacher where both experts appear in top-2.
        queries = torch.stack(
            [
                torch.tensor([0.7, 0.7, 0.7, 0, 0, 0, 0, 0]),
            ]
        )
        records = [{"teacher_set": (0, 1)}]
        metrics = evaluate_v6_router(router, queries, records, split="validation")
        self.assertEqual(metrics["PairRecall"], 1.0)


class ThresholdTuningTest(unittest.TestCase):
    def test_tuning_picks_best_thresholds_on_validation(self):
        router = _populated_router()
        with torch.no_grad():
            router.key_store.keys["0"].copy_(
                torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
            )
        queries = torch.stack(
            [
                torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]),
                torch.tensor([-1.0, 0, 0, 0, 0, 0, 0, 0]),
            ]
        )
        records = [{"teacher_set": (0,)}, {"teacher_set": ()}]
        candidates = [(0.3, 0.3), (0.5, 0.5), (0.8, 0.8)]
        best, exact = tune_v6_thresholds(
            router, queries, records, candidates, split="validation"
        )
        self.assertGreaterEqual(exact, 0.5)

    def test_test_split_refused_in_tuning(self):
        router = _populated_router()
        with self.assertRaisesRegex(ValueError, "test"):
            tune_v6_thresholds(router, torch.randn(2, 8), _records(2),
                               [(0.5, 0.5)], split="test")


class CalibrationConfigTest(unittest.TestCase):
    def test_defaults_from_task_book(self):
        config = CalibrationConfig()
        self.assertEqual(config.epochs, 5)
        self.assertAlmostEqual(config.learning_rate, 2.0e-4)
        self.assertTrue(config.lambda_sparse < 0.1)

    def test_invalid_epochs_rejected(self):
        with self.assertRaisesRegex(ValueError, "epochs"):
            CalibrationConfig(epochs=0)


if __name__ == "__main__":
    unittest.main()
