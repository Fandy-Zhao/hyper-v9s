"""Equivalence gates for the V8-Exact-Accelerated execution flags.

Every flag in this file is *execution only*: it must not change a single
produced number.  The tests therefore compare the accelerated path against the
frozen baseline path on the same tensors and demand bit-identical forward
outputs and backward gradients -- ``torch.equal``, not ``assert_close``.
"""

import unittest

import torch
import torch.nn as nn

from compose.adapters.lora import (
    ComposeLinear,
    fast_selection_enabled,
    set_fast_selection,
)
from compose.adapters.runtime import use_selection
from compose.adapters.types import PAD_EXPERT_ID, ComposeSelection


def _layer(seed=0, rank=4):
    generator = torch.Generator().manual_seed(seed)
    base = nn.Linear(6, 5, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.randn(5, 6, generator=generator))
    layer = ComposeLinear(base, rank=rank, alpha=16.0)
    for expert_id in range(4):
        expert = layer.add_expert(expert_id)
        with torch.no_grad():
            expert.lora_A.weight.copy_(torch.randn(rank, 6, generator=generator))
            expert.lora_B.weight.copy_(torch.randn(5, rank, generator=generator))
    return layer


def _selection(rows, gates, normalization="none"):
    padded_ids = []
    padded_gates = []
    for ids, weights in zip(rows, gates):
        padded_ids.append(list(ids) + [PAD_EXPERT_ID] * (4 - len(ids)))
        padded_gates.append(list(weights) + [0.0] * (4 - len(weights)))
    return ComposeSelection(
        torch.tensor(padded_ids, dtype=torch.long),
        torch.tensor(padded_gates, dtype=torch.float32),
        normalization=normalization,
    )


#: Single, pair, triple, empty and mixed rows in one batch.
CASES = (
    ((((0,), (1,), (2,), (3,))), ((1.0,), (1.0,), (1.0,), (1.0,))),
    ((((0, 2), (1, 3), (0, 1), (2, 3))), ((1.0, 1.0),) * 4),
    ((((0, 1, 2), (1, 2, 3), (0, 2, 3), (0, 1, 3))), ((1.0, 0.5, 0.25),) * 4),
    ((((), (0,), (1, 2, 3), (3,))), ((0.0,), (0.7,), (1.0, 0.5, 0.25), (2.0,))),
    ((((3,), (3,), (2,), (2,))), ((0.1,), (0.9,), (0.4,), (0.6,))),
)


class SelectionPlanEquivalenceTest(unittest.TestCase):
    def setUp(self):
        self._original = fast_selection_enabled()
        set_fast_selection(False)

    def tearDown(self):
        set_fast_selection(self._original)

    def _run(self, enabled, case, normalization="none", kappa=None, batched=False):
        ids, gates = case
        generator = torch.Generator().manual_seed(11)
        batch = len(ids)
        inputs = torch.randn(batch, 7, 6, generator=generator)
        layer = _layer(seed=3)
        if kappa is not None:
            layer.set_expert_calibration(kappa)
        layer.zero_grad(set_to_none=True)
        for name, parameter in layer.named_parameters():
            if ".experts." in name:
                parameter.requires_grad_(True)
        selection = _selection(ids, gates, normalization=normalization)
        set_fast_selection(enabled)
        try:
            with use_selection(selection):
                output = layer(inputs)
            output.square().sum().backward()
        finally:
            set_fast_selection(self._original)
        grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in layer.named_parameters()
            if parameter.grad is not None
        }
        return output.detach().clone(), grads

    def test_forward_and_backward_are_bit_identical(self):
        for case in CASES:
            for normalization in ("none", "l1", "l2"):
                for kappa in (None, {0: 0.5, 1: 2.0, 2: 0.25, 3: 1.5}):
                    with self.subTest(case=case, normalization=normalization, kappa=kappa):
                        baseline, baseline_grads = self._run(
                            False, case, normalization, kappa
                        )
                        accelerated, accelerated_grads = self._run(
                            True, case, normalization, kappa
                        )
                        self.assertTrue(torch.equal(baseline, accelerated))
                        self.assertEqual(set(baseline_grads), set(accelerated_grads))
                        self.assertTrue(baseline_grads)
                        for name, value in baseline_grads.items():
                            self.assertTrue(
                                torch.equal(value, accelerated_grads[name]), name
                            )

    def test_empty_selection_matches(self):
        # One batch row, no active expert: expert_ids row is all pad, gate 0.
        empty_case = (((),), ((0.0,),))
        baseline, _ = self._run(False, empty_case)
        accelerated, _ = self._run(True, empty_case)
        self.assertTrue(torch.equal(baseline, accelerated))

    def test_plan_is_rebuilt_when_the_selection_changes(self):
        layer = _layer(seed=5)
        inputs = torch.randn(1, 3, 6, generator=torch.Generator().manual_seed(2))
        set_fast_selection(True)
        try:
            first = _selection(((0,),), ((1.0,),))
            with use_selection(first):
                out_first = layer(inputs).detach().clone()
            second = _selection(((2,),), ((1.0,),))
            with use_selection(second):
                out_second = layer(inputs).detach().clone()
            # Re-using the first selection must reproduce the first output even
            # though a different selection was executed in between.
            with use_selection(first):
                out_first_again = layer(inputs).detach().clone()
            with use_selection(second):
                out_second_again = layer(inputs).detach().clone()
        finally:
            set_fast_selection(self._original)
        self.assertTrue(torch.equal(out_first, out_first_again))
        self.assertTrue(torch.equal(out_second, out_second_again))
        self.assertFalse(torch.equal(out_first, out_second))

    def test_unregistered_expert_still_raises(self):
        layer = _layer(seed=7)
        inputs = torch.randn(1, 3, 6, generator=torch.Generator().manual_seed(4))
        selection = _selection(((9,),), ((1.0,),))
        for enabled in (False, True):
            set_fast_selection(enabled)
            try:
                with use_selection(selection):
                    with self.assertRaisesRegex(KeyError, "expert 9 is not registered"):
                        layer(inputs)
            finally:
                set_fast_selection(self._original)

    def test_batch_mismatch_still_raises(self):
        layer = _layer(seed=9)
        inputs = torch.randn(2, 3, 6, generator=torch.Generator().manual_seed(6))
        selection = _selection(((0,),), ((1.0,),))
        for enabled in (False, True):
            set_fast_selection(enabled)
            try:
                with use_selection(selection):
                    with self.assertRaisesRegex(ValueError, "does not match input"):
                        layer(inputs)
            finally:
                set_fast_selection(self._original)


class FlagDefaultTest(unittest.TestCase):
    def test_defaults_to_the_frozen_baseline_path(self):
        import os

        self.assertNotEqual(os.environ.get("COMPOSE_SELECTION_PLAN"), "1")
        self.assertFalse(fast_selection_enabled())


if __name__ == "__main__":
    unittest.main()
