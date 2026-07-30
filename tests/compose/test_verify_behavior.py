import unittest

import torch

from compose.experts.verify_behavior import tensor_sha256


class VerifyBehaviorTest(unittest.TestCase):
    def test_tensor_hash_is_exact_and_shape_sensitive(self):
        first = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
        self.assertEqual(tensor_sha256(first), tensor_sha256(first.clone()))
        self.assertNotEqual(tensor_sha256(first), tensor_sha256(first.reshape(2, 1)))
        changed = first.clone()
        changed[0, 0] = 3.0
        self.assertNotEqual(tensor_sha256(first), tensor_sha256(changed))
