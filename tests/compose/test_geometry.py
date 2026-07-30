import unittest

import torch

from compose.experts.analyze_geometry import factorized_delta_stats


class GeometryTest(unittest.TestCase):
    def test_factorized_statistics_match_dense_deltas(self):
        torch.manual_seed(13)
        a1, b1 = torch.randn(2, 5), torch.randn(7, 2)
        a2, b2 = torch.randn(3, 5), torch.randn(7, 3)
        result = factorized_delta_stats((a1, b1, 2.0), (a2, b2, 0.5))
        delta1 = 2.0 * b1 @ a1
        delta2 = 0.5 * b2 @ a2
        expected_cosine = torch.nn.functional.cosine_similarity(
            delta1.flatten(), delta2.flatten(), dim=0
        ).item()
        self.assertAlmostEqual(result["first_delta_rms"], delta1.square().mean().sqrt().item(), places=6)
        self.assertAlmostEqual(result["second_delta_rms"], delta2.square().mean().sqrt().item(), places=6)
        self.assertAlmostEqual(result["cosine"], expected_cosine, places=6)

    def test_zero_delta_has_finite_zero_cosine(self):
        a = torch.ones(1, 2)
        b = torch.zeros(3, 1)
        result = factorized_delta_stats((a, b, 1.0), (a, b, 1.0))
        self.assertEqual(result["cosine"], 0.0)
