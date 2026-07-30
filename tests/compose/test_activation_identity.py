import unittest

import torch

from compose.experts.activation_identity import vector_cosine


class ActivationIdentityTest(unittest.TestCase):
    def test_vector_cosine(self):
        self.assertAlmostEqual(vector_cosine(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])), 1.0)
        self.assertAlmostEqual(vector_cosine(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])), -1.0)
        self.assertEqual(vector_cosine(torch.zeros(2), torch.ones(2)), 0.0)
