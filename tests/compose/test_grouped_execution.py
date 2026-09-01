import copy
import math
import unittest

import torch
import torch.nn as nn

from compose.adapters.lora import ComposeLinear
from compose.adapters.runtime import use_selection
from compose.adapters.types import ComposeSelection


def _layer(expert_count=5, dtype=torch.float32):
    torch.manual_seed(17)
    layer = ComposeLinear(nn.Linear(3, 2, bias=False), rank=2, alpha=2.0)
    for expert_id in range(expert_count):
        layer.add_expert(expert_id)
    return layer.to(dtype=dtype)


def _reference_forward(layer, inputs, selection):
    """Reference mirror of ComposeLinear.forward (incl. the composition
    rule: a sample with 2 active experts scales its delta sum by 1/sqrt(2),
    a 3-expert cluster-training selection by 1/sqrt(3))."""
    result = layer.base_layer(inputs)
    delta = torch.zeros_like(result)
    gate_shape = [1] + [1] * (result.ndim - 1)
    for sample_index in range(inputs.shape[0]):
        sample_input = inputs[sample_index : sample_index + 1]
        active = 0
        for slot in range(selection.top_k):
            gate = selection.gates[sample_index, slot]
            if bool(gate > 0):
                expert_id = int(selection.expert_ids[sample_index, slot])
                expert_delta = layer.experts[str(expert_id)](sample_input).to(result.dtype)
                delta[sample_index : sample_index + 1] += (
                    expert_delta * gate.to(result.dtype).reshape(gate_shape)
                )
                active += 1
        if active == 2:
            delta[sample_index] = delta[sample_index] / math.sqrt(2.0)
        elif active == 3:
            delta[sample_index] = delta[sample_index] / math.sqrt(3.0)
    return result + delta


def _mixed_selection():
    # The unified Compose selection is MAX_ACTIVE_EXPERTS=4 slots wide; a
    # zero-weight slot is expressed as the -1 pad.
    return ComposeSelection(
        expert_ids=torch.tensor(
            [[0, 1, -1, -1], [1, -1, -1, -1], [2, 3, -1, -1]],
            dtype=torch.long,
        ),
        gates=torch.tensor(
            [[0.6, 0.8, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]
        ),
        normalization="none",
    )


class GroupedExecutionTest(unittest.TestCase):
    def test_mixed_batch_matches_per_sample_reference(self):
        grouped = _layer()
        reference = copy.deepcopy(grouped)
        inputs = torch.randn(3, 4, 3)
        selection = _mixed_selection()
        with use_selection(selection):
            actual = grouped(inputs)
        expected = _reference_forward(reference, inputs, selection)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_each_expert_receives_only_active_rows(self):
        layer = _layer()
        observed_batch_sizes = {expert_id: [] for expert_id in range(5)}
        handles = []
        for expert_id in range(5):
            handles.append(
                layer.experts[str(expert_id)].register_forward_pre_hook(
                    lambda module, args, value=expert_id: observed_batch_sizes[value].append(
                        args[0].shape[0]
                    )
                )
            )
        try:
            with use_selection(_mixed_selection()):
                layer(torch.randn(3, 2, 3))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(observed_batch_sizes[0], [1])
        self.assertEqual(observed_batch_sizes[1], [2])
        self.assertEqual(observed_batch_sizes[2], [1])
        self.assertEqual(observed_batch_sizes[3], [1])
        self.assertEqual(observed_batch_sizes[4], [])

    def test_grouped_backward_matches_reference(self):
        grouped = _layer()
        reference = copy.deepcopy(grouped)
        grouped_inputs = torch.randn(3, 3, requires_grad=True)
        reference_inputs = grouped_inputs.detach().clone().requires_grad_(True)
        selection = _mixed_selection()

        with use_selection(selection):
            grouped_loss = grouped(grouped_inputs).square().sum()
        reference_loss = _reference_forward(reference, reference_inputs, selection).square().sum()
        grouped_loss.backward()
        reference_loss.backward()

        torch.testing.assert_close(
            grouped_inputs.grad, reference_inputs.grad, rtol=1e-6, atol=1e-6
        )
        for (grouped_name, grouped_parameter), (reference_name, reference_parameter) in zip(
            grouped.named_parameters(), reference.named_parameters()
        ):
            self.assertEqual(grouped_name, reference_name)
            torch.testing.assert_close(
                grouped_parameter.grad,
                reference_parameter.grad,
                rtol=1e-6,
                atol=1e-6,
            )

    def test_bfloat16_forward_backward_matches_reference(self):
        grouped = _layer(dtype=torch.bfloat16)
        reference = copy.deepcopy(grouped)
        grouped_inputs = torch.randn(3, 2, 3, dtype=torch.bfloat16, requires_grad=True)
        reference_inputs = grouped_inputs.detach().clone().requires_grad_(True)
        selection = _mixed_selection()
        with use_selection(selection):
            actual = grouped(grouped_inputs)
        expected = _reference_forward(reference, reference_inputs, selection)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        actual.float().sum().backward()
        expected.float().sum().backward()
        torch.testing.assert_close(
            grouped_inputs.grad, reference_inputs.grad, rtol=2e-2, atol=2e-2
        )

    def test_rank_two_input_is_supported(self):
        layer = _layer()
        inputs = torch.randn(3, 3)
        selection = _mixed_selection()
        with use_selection(selection):
            actual = layer(inputs)
        expected = _reference_forward(copy.deepcopy(layer), inputs, selection)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
