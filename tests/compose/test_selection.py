import math
import unittest

import torch

from compose.adapters import ExpertManager, inject_compose_adapters
from compose.adapters.types import ComposeSelection
from compose.config import ComposeAdapterConfig
from compose.experts import ExpertPool
from test_injection import TinyModel


class ComposeSelectionTest(unittest.TestCase):
    def test_none_preserves_user_gates(self):
        gates = torch.tensor([[2.0, 3.0, 0.0, 0.0], [0.25, 0.75, 0.0, 0.0]])
        selection = ComposeSelection(
            torch.tensor([[0, 1, -1, -1], [1, 2, -1, -1]]), gates, normalization="none"
        )
        torch.testing.assert_close(selection.gates, gates)

    def test_l1_normalizes_each_row(self):
        selection = ComposeSelection(
            torch.tensor([[0, 1, -1, -1], [1, 2, -1, -1]]),
            torch.tensor([[1.0, 3.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0]]),
            normalization="l1",
        )
        torch.testing.assert_close(selection.gates.sum(dim=1), torch.ones(2))

    def test_l2_normalizes_each_row(self):
        selection = ComposeSelection(
            torch.tensor([[0, 1, -1, -1], [1, 2, -1, -1]]),
            torch.tensor([[3.0, 4.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
            normalization="l2",
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(selection.gates, ord=2, dim=1), torch.ones(2)
        )
        torch.testing.assert_close(
            selection.gates[1, :2], torch.full((2,), 1.0 / math.sqrt(2))
        )
        self.assertEqual(float(selection.gates[1, 2]), 0.0)
        self.assertEqual(float(selection.gates[1, 3]), 0.0)

    def test_manager_default_l2_pair_uses_inverse_sqrt_two(self):
        model = TinyModel(layer_count=1)
        inject_compose_adapters(model, ComposeAdapterConfig())
        pool = ExpertPool(ExpertManager(model))
        pool.register(0)
        pool.register(1)
        selection = pool.make_selection(
            [0, 1], batch_size=2, normalization="l2"
        )
        torch.testing.assert_close(
            selection.gates[:, :2],
            torch.full((2, 2), 1.0 / math.sqrt(2)),
        )
        torch.testing.assert_close(selection.gates[:, 2], torch.zeros(2))

    def test_single_expert_default_is_one(self):
        selection = ComposeSelection(
            torch.tensor([[3, -1, -1, -1]]),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            normalization="none",
        )
        torch.testing.assert_close(
            selection.gates, torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        )

    def test_duplicate_expert_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate expert IDs"):
            ComposeSelection(
                torch.tensor([[1, 1, -1, -1]]),
                torch.tensor([[0.5, 0.5, 0.0, 0.0]]),
            )

    def test_invalid_normalization_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "normalization"):
            ComposeSelection(
                torch.tensor([[0, -1, -1, -1]]),
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                normalization="softmax",
            )

    def test_gate_validation_is_independent_of_normalization(self):
        invalid_gates = (
            torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]),
            torch.tensor([[-1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        )
        for gates in invalid_gates:
            with self.subTest(gates=gates), self.assertRaises(ValueError):
                ComposeSelection(
                    torch.tensor([[0, -1, -1, -1]]), gates, normalization="none"
                )

    def test_to_preserves_normalization(self):
        selection = ComposeSelection(
            torch.tensor([[0, 1, -1, -1]]),
            torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            normalization="l2",
        )
        moved = selection.to(torch.device("cpu"))
        self.assertEqual(moved.normalization, "l2")
        torch.testing.assert_close(moved.gates, selection.gates)
