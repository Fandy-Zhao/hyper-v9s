import unittest

import torch

from compose.router.multilabel_router import (
    MultiLabelQueryKeyRouter,
    RouterThresholds,
    router_loss,
    tune_thresholds,
)


class MultiLabelRouterTest(unittest.TestCase):
    def test_router_selects_at_most_two(self):
        router = MultiLabelQueryKeyRouter(4, 5, 3)
        selections, probabilities = router.select(torch.randn(8, 4), range(5), RouterThresholds(0.0, 0.0))
        self.assertEqual(probabilities.shape, (8, 5))
        self.assertTrue(all(len(value) <= 2 for value in selections))

    def test_loss_has_all_terms_and_gradients(self):
        logits = torch.randn(3, 4, requires_grad=True)
        targets = torch.tensor([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
        losses = router_loss(logits, targets, anchor_logits=torch.zeros_like(logits))
        self.assertEqual(set(losses), {"total", "bce", "ranking", "sparse", "anchor"})
        losses["total"].backward()
        self.assertIsNotNone(logits.grad)

    def test_test_split_cannot_tune_thresholds(self):
        rows = [{"probabilities": [0.8, 0.2], "teacher_set": [0]}]
        candidates = [RouterThresholds(0.5, 0.5)]
        with self.assertRaisesRegex(ValueError, "test"):
            tune_thresholds(rows, candidates, "test")
        selected, score = tune_thresholds(rows, candidates, "validation")
        self.assertEqual(selected, candidates[0])
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
