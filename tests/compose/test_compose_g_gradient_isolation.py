"""Test G (spec §27): conditional-residual gradient isolation -- when the
cluster expert is trained, only that expert receives gradient; historical
experts stay frozen with no grads."""

import unittest

import torch
from torch import nn

from compose.adapters.lora import ComposeLinear
from compose.adapters.manager import ExpertManager
from compose.adapters.runtime import use_selection
from compose.adapters.types import PAD_EXPERT_ID, ComposeSelection
from compose.experts import ExpertPool


def _layer(rank=4, alpha=8.0):
    torch.manual_seed(23)
    layer = ComposeLinear(nn.Linear(16, 8, bias=False), rank=rank, alpha=alpha)
    for expert_id in range(3):
        layer.add_expert(expert_id)
    # LoRAExpert zero-initializes lora_B, so expert deltas (and lora_A
    # gradients) are zero until trained. Perturb B so gradient flow through
    # A is observable; the train-only flags are independent of the weights.
    with torch.no_grad():
        for expert in layer.experts.values():
            expert.lora_B.weight.normal_(0.0, 0.05)
    return layer


def _selection(batch_size, *rows):
    ids = torch.full((batch_size, 4), PAD_EXPERT_ID, dtype=torch.long)
    gates = torch.zeros(batch_size, 4)
    for index, (expert_ids, expert_gates) in enumerate(rows):
        for slot, (expert_id, gate) in enumerate(zip(expert_ids, expert_gates)):
            ids[index, slot] = expert_id
            gates[index, slot] = gate
    return ComposeSelection(ids, gates)


class GradientIsolationTest(unittest.TestCase):
    def test_train_only_freezes_historical_experts(self):
        layer = _layer()
        manager = ExpertManager(nn.ModuleDict({"layer": layer}))
        pool = ExpertPool(manager)
        pool.train_only([2])  # only the new cluster expert is trainable
        self.assertTrue(layer.experts["2"].lora_A.weight.requires_grad)
        self.assertTrue(layer.experts["2"].lora_B.weight.requires_grad)
        for expert_id in (0, 1):
            self.assertFalse(layer.experts[str(expert_id)].lora_A.weight.requires_grad)
            self.assertFalse(layer.experts[str(expert_id)].lora_B.weight.requires_grad)
        # The base layer is frozen too.
        self.assertFalse(layer.base_layer.weight.requires_grad)

    def test_backward_writes_grads_only_for_selected_expert(self):
        layer = _layer()
        inputs = torch.randn(2, 16, requires_grad=False)
        selection = _selection(
            2,
            ([2, -1, -1, -1], [1.0, 0.0, 0.0, 0.0]),
            ([2, 1, -1, -1], [1.0, 1.0, 0.0, 0.0]),  # cluster expert + one historical
        )
        with use_selection(selection):
            loss = layer(inputs).sum()
        loss.backward()
        # The new expert accumulates gradient from both samples.
        self.assertIsNotNone(layer.experts["2"].lora_A.weight.grad)
        self.assertGreater(
            float(layer.experts["2"].lora_A.weight.grad.abs().sum()), 0.0
        )
        # Expert 1 is active for sample 2 only (pair with 2) -> grads exist.
        self.assertIsNotNone(layer.experts["1"].lora_A.weight.grad)
        self.assertIsNotNone(layer.experts["1"].lora_B.weight.grad)
        # Expert 0 is never selected -> its forward never executes and the
        # parameters are absent from the graph: no grads at all.
        self.assertIsNone(layer.experts["0"].lora_A.weight.grad)
        self.assertIsNone(layer.experts["0"].lora_B.weight.grad)

    def test_unselected_expert_gets_no_grad_even_when_trainable(self):
        layer = _layer()
        manager = ExpertManager(nn.ModuleDict({"layer": layer}))
        pool = ExpertPool(manager)
        pool.train_only([0, 1, 2])  # all trainable, but selection gates which run
        inputs = torch.randn(1, 16)
        with use_selection(_selection(1, ([1, -1, -1, -1], [1.0, 0.0, 0.0, 0.0]))):
            loss = layer(inputs).sum()
        loss.backward()
        # Expert 2 is trainable but not selected: forward never touches it.
        self.assertIsNone(layer.experts["2"].lora_B.weight.grad)
        self.assertIsNotNone(layer.experts["1"].lora_B.weight.grad)


if __name__ == "__main__":
    unittest.main()
