"""Regression test for the dead-expert training bug (pre-formal validation).

Real-scale symptom (compose_ucit_smoke3 task0): the cluster LoRA training
log reported ``Finite-gradient LoRA-B count: 0`` and the committed
checkpoints had ``lora_B.weight`` exactly zero for every layer, so the
experts contributed nothing, RMS raw deltas were zero everywhere, and the
runtime kappa calibration collapsed to ``kappa_min`` for every layer.

Root cause: reentrant activation checkpointing (the torch default) runs
each checkpointed segment again during ``loss.backward()``; multithreaded
autograd (on by default) dispatches that recomputation to engine worker
threads. ``ContextVar`` selections (``compose.adapters.runtime.
use_selection``) do not propagate to those threads, so the recomputed
graph is backbone-only and the cluster expert never receives gradients.

Fix: ``compose.train.train_compose._prepare_cluster_expert_backward``
pins the backward to the calling thread (``set_multithreading_enabled``)
and refuses multi-GPU DataParallel, where the forward itself runs on
worker threads. These tests assert the resulting contract: a checkpointed
``ComposeLinear`` trained under a context selection must accumulate
nonzero ``lora_B`` gradients.
"""

import unittest
from unittest import mock

import torch
from torch import nn

from compose.adapters.lora import ComposeLinear
from compose.adapters.runtime import use_selection
from compose.adapters.types import PAD_EXPERT_ID, ComposeSelection
from compose.train.train_compose import _prepare_cluster_expert_backward


def _selection(batch_size: int) -> ComposeSelection:
    ids = torch.full((batch_size, 3), PAD_EXPERT_ID, dtype=torch.long)
    gates = torch.zeros(batch_size, 3)
    ids[:, 0] = 7
    gates[:, 0] = 1.0
    return ComposeSelection(ids, gates)


def _build():
    torch.manual_seed(0)
    layer = ComposeLinear(nn.Linear(64, 64, bias=False), rank=4, alpha=8.0)
    layer.add_expert(7)
    with torch.no_grad():
        layer.experts["7"].lora_B.weight.normal_(0.0, 0.05)
    return layer


class CheckpointSelectionGradTest(unittest.TestCase):
    def test_checkpointed_selection_backward_reaches_lora_b(self):
        """With the backward pinned to the calling thread (the fix), the
        reentrant-checkpoint recompute sees the selection and lora_B
        receives a nonzero gradient."""
        with mock.patch("torch.cuda.device_count", return_value=1):
            _prepare_cluster_expert_backward()
        self.assertFalse(torch.autograd.is_multithreading_enabled())
        layer = _build()
        x = torch.randn(4, 64, requires_grad=True)
        with use_selection(_selection(4)):
            y = torch.utils.checkpoint.checkpoint(
                layer, x, use_reentrant=True
            )
            y.square().mean().backward()
        grad = layer.experts["7"].lora_B.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_single_gpu_guard_refuses_multi_gpu(self):
        """DataParallel would hide the selection from the forward entirely;
        the setup must refuse to train in that configuration."""
        with mock.patch("torch.cuda.device_count", return_value=2):
            with self.assertRaisesRegex(ValueError, "exactly one visible GPU"):
                _prepare_cluster_expert_backward()

    def test_single_gpu_guard_accepts_one_gpu(self):
        with mock.patch("torch.cuda.device_count", return_value=1):
            _prepare_cluster_expert_backward()
        self.assertFalse(torch.autograd.is_multithreading_enabled())


if __name__ == "__main__":
    unittest.main()
