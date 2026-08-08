"""Test A (spec §27): per-sample Top-M retrieval is a batch x M expert-id
matrix; row i holds sample i's OWN Top-M ids, never a broadcast of row 0."""

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from compose.router.router import ComposeRouter, PAD_EXPERT_ID


class _StubEncoder(nn.Module):
    query_dim = 128

    def forward(self, z_v, z_s):
        raise NotImplementedError("stub; retrieve() never touches the encoder")


class PerSampleTopMTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.router = ComposeRouter(_StubEncoder(), top_m=8, tau_none=0.5, tau_second=0.5)
        # Four keys on the standard basis: e0..e3. Basis keys keep
        # cross-similarities exactly 0 so sortedness assertions are not
        # perturbed by random-direction noise.
        for expert_id in range(4):
            key = torch.eye(128)[expert_id]
            self.router.add_expert(
                expert_id, creation_task=0, checkpoint_sha256="sha", key=key
            )
        self.keys = [self.router.key_store.keys[str(i)].detach() for i in range(4)]

    def test_rows_differ_across_batch(self):
        # Each sample is most similar to a DIFFERENT expert.
        queries = torch.stack(
            [
                F.normalize(2.0 * self.keys[0] + torch.randn(128) * 0.05, dim=0),
                F.normalize(2.0 * self.keys[1] + torch.randn(128) * 0.05, dim=0),
                F.normalize(2.0 * self.keys[2] + torch.randn(128) * 0.05, dim=0),
            ]
        )
        result = self.router.retrieve(queries, self.router.expert_ids)
        ids = result.expert_ids
        self.assertEqual(ids.shape, (3, 8))
        self.assertEqual(ids.dtype, torch.long)
        # Per-sample top-1: row 1 and row 2 are NOT row 0 broadcast.
        self.assertEqual(int(ids[0, 0]), 0)
        self.assertEqual(int(ids[1, 0]), 1)
        self.assertEqual(int(ids[2, 0]), 2)
        self.assertFalse(torch.equal(ids[0], ids[1]))
        self.assertFalse(torch.equal(ids[0], ids[2]))

    def test_similarity_sorted_descending_per_row(self):
        queries = torch.stack([self.keys[1] * 0.8 + self.keys[0] * 0.2] * 3)
        result = self.router.retrieve(queries, self.router.expert_ids)
        row = result.similarities[0]
        self.assertTrue(
            bool(torch.all(row[1:] <= row[:-1])), "Top-M must be sorted descending"
        )

    def test_padding_when_pool_smaller_than_top_m(self):
        queries = self.keys[0].unsqueeze(0).repeat(2, 1)
        result = self.router.retrieve(queries, self.router.expert_ids)
        ids = result.expert_ids
        # 4 real experts, then 4 pad slots.
        self.assertTrue(bool(torch.all(ids[:, :4] >= 0)))
        self.assertTrue(bool(torch.all(ids[:, 4:] == PAD_EXPERT_ID)))
        self.assertEqual(result.visible_ids, (0, 1, 2, 3))

    def test_empty_visible_pool_pads_everything(self):
        empty = ComposeRouter(_StubEncoder(), top_m=8)
        result = empty.retrieve(torch.randn(2, 128), ())
        self.assertEqual(result.expert_ids.shape, (2, 8))
        self.assertTrue(bool(torch.all(result.expert_ids == PAD_EXPERT_ID)))
        self.assertEqual(result.similarities.shape, (2, 0))


if __name__ == "__main__":
    unittest.main()
