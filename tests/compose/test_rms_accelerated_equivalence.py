"""``compute_expert_rms_accelerated`` must be bit-identical to the baseline.

The accelerated collector is an *execution* change only: it reduces the shared
module output once per layer instead of once per expert, and it defers the
per-moment ``.item()`` into one batched device transfer per batch.  Both are
claims about floating-point value preservation, so they are tested the only way
such a claim can be tested -- by comparing the raw fp64 accumulators for exact
equality, not by comparing the rounded RMS report.

The comparison is on ``RMSStatistics.state_dict()`` because that is the exact
object that gets persisted and later consumed by ``build_kappa_calibration``;
any drift in the merge sequence shows up here as a differing float.
"""

import math
import os
import sys
import unittest

import torch
from torch import nn

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from compose.adapters.lora import ComposeLinear  # noqa: E402
from compose.lora.rms import (  # noqa: E402
    ComposeRMSConfig,
    RMSStatistics,
    compute_expert_rms,
    compute_expert_rms_accelerated,
)

PROVENANCE = {
    "calibration_split": "validation",
    "checkpoint_hash": "unit-test",
    "dataset_manifest_hash": "unit-test",
    "composition_config_hash": "unit-test",
}


class _TinyModel(nn.Module):
    """Two ComposeLinear layers, so ordering across layers is exercised too."""

    def __init__(self, experts, rank, seed):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.proj_a = ComposeLinear(nn.Linear(16, 12, bias=False), rank=rank, alpha=2 * rank)
        self.proj_b = ComposeLinear(nn.Linear(12, 8, bias=False), rank=rank, alpha=2 * rank)
        for module in (self.proj_a, self.proj_b):
            for expert_id in experts:
                expert = module.add_expert(expert_id)
                with torch.no_grad():
                    expert.lora_A.weight.copy_(
                        torch.randn(expert.lora_A.weight.shape, generator=generator)
                    )
                    expert.lora_B.weight.copy_(
                        torch.randn(expert.lora_B.weight.shape, generator=generator)
                    )

    def forward(self, inputs):
        return self.proj_b(self.proj_a(inputs))


def _collect(collector, model, expert_ids, batches, config):
    return collector(
        model,
        expert_ids,
        batches,
        PROVENANCE,
        config,
        prepare_batch=lambda batch: batch,
        forward_fn=lambda inputs: model(inputs),
        device="cpu",
    )


class AcceleratedCollectorEquivalenceTest(unittest.TestCase):
    def _run_pair(self, experts, rank=4, seed=7, batches=3, length=5, batch_size=1):
        model = _TinyModel(experts, rank=rank, seed=seed)
        generator = torch.Generator().manual_seed(seed + 100)
        data = [
            torch.randn(batch_size, length, 16, generator=generator) for _ in range(batches)
        ]
        config = ComposeRMSConfig(calibration_split="validation")
        baseline, base_pairs = _collect(compute_expert_rms, model, experts, data, config)
        accelerated, fast_pairs = _collect(
            compute_expert_rms_accelerated, model, experts, data, config
        )
        return baseline, base_pairs, accelerated, fast_pairs, model

    def test_accumulators_are_bit_identical(self):
        baseline, _, accelerated, _, _model = self._run_pair([0, 1, 2])
        left = baseline.state_dict()
        right = accelerated.state_dict()
        self.assertEqual(sorted(left["entries"]), sorted(right["entries"]))
        for token in sorted(left["entries"]):
            with self.subTest(token=token):
                self.assertEqual(left["entries"][token], right["entries"][token])

    def test_more_experts_than_samples_still_matches(self):
        """The output reduction is shared by experts that never co-occur."""
        baseline, _, accelerated, _, _model = self._run_pair([3, 5, 8, 13], length=1)
        self.assertEqual(baseline.state_dict(), accelerated.state_dict())

    def test_every_accumulator_carries_the_full_element_count(self):
        """Element counts pin the *identity* of the tensors being merged.

        The delta lane accumulates the expert delta (``out_features`` per
        token) while the output lane accumulates the module output, so a
        swapped or reused tensor shows up as a wrong count even when the
        arithmetic happens to agree.
        """
        baseline, _, accelerated, _, model = self._run_pair([0, 1, 2], batches=3, length=5)
        widths = {"proj_a": model.proj_a.out_features, "proj_b": model.proj_b.out_features}
        tokens = 3 * 5
        for token, entry in baseline.state_dict()["entries"].items():
            layer = entry["key"]["layer_name"]
            with self.subTest(token=token):
                self.assertEqual(entry["sample_count"], 3)
                self.assertEqual(entry["delta"]["count"], tokens * widths[layer])
                self.assertEqual(
                    entry["output"]["count"],
                    tokens * widths[layer],
                    "output lane must count module-output elements",
                )
                self.assertEqual(
                    entry["output"]["count"],
                    accelerated.state_dict()["entries"][token]["output"]["count"],
                )

    def test_derived_kappa_calibration_is_identical(self):
        from compose.lora.rms import build_kappa_calibration

        experts = [0, 1, 2]
        baseline, _, accelerated, _, _model = self._run_pair(experts)
        config = ComposeRMSConfig(calibration_split="validation")
        self.assertEqual(
            build_kappa_calibration(baseline, experts, config),
            build_kappa_calibration(accelerated, experts, config),
        )

    def test_pair_diagnostics_match_for_a_two_expert_pool(self):
        """The pair branch fires only for a two-expert pool; it must agree."""
        baseline, base_pairs, accelerated, fast_pairs, _model = self._run_pair([0, 1])
        self.assertEqual(sorted(base_pairs), sorted(fast_pairs))
        for layer in sorted(base_pairs):
            with self.subTest(layer=layer):
                self.assertEqual(base_pairs[layer].state_dict(), fast_pairs[layer].state_dict())

    def test_empty_pool_is_rejected_by_both(self):
        model = _TinyModel([0], rank=2, seed=1)
        for collector in (compute_expert_rms, compute_expert_rms_accelerated):
            with self.subTest(collector=collector.__name__):
                with self.assertRaises(ValueError):
                    _collect(collector, model, [], [torch.randn(1, 2, 16)], None)

    def test_missing_expert_is_rejected_by_both(self):
        model = _TinyModel([0], rank=2, seed=1)
        for collector in (compute_expert_rms, compute_expert_rms_accelerated):
            with self.subTest(collector=collector.__name__):
                with self.assertRaises(KeyError):
                    _collect(
                        collector, model, [0, 4], [torch.randn(1, 2, 16)],
                        ComposeRMSConfig(calibration_split="validation"),
                    )

    def test_device_moments_reproduces_update_exactly(self):
        """The on-device helper must equal the three ``.item()`` reductions.

        This is the whole premise of the accelerated path: ``device_moments``
        left on the device and drained later has to produce the same fp64
        moments as ``update`` draining each reduction immediately.
        """
        from compose.lora.statistics import OnlineMoments, StatisticKey

        generator = torch.Generator().manual_seed(11)
        statistic_key = StatisticKey(
            expert_id=0,
            layer_name="layer",
            module_name="layer",
            target_module_type="ComposeLinear",
        )
        for shape in ((1, 7), (3, 5, 4), (64,), (1, 1)):
            with self.subTest(shape=shape):
                values = torch.randn(*shape, generator=generator) * 3.5
                reference = RMSStatistics(dict(PROVENANCE))
                reference.update(statistic_key, values, values)

                numel, mean, m2, sum_squares = OnlineMoments.device_moments(values)
                deferred = OnlineMoments()
                deferred.merge(numel, mean.item(), m2.item(), sum_squares.item())

                entry = reference.entries[statistic_key.token()]
                self.assertEqual(entry["delta"].state_dict(), deferred.state_dict())
                self.assertEqual(entry["output"].state_dict(), deferred.state_dict())

    def test_device_moments_of_an_empty_tensor_is_none(self):
        from compose.lora.statistics import OnlineMoments

        self.assertIsNone(OnlineMoments.device_moments(torch.zeros(0)))
        self.assertIsNone(OnlineMoments.device_moments(torch.zeros(1, 0, 3)))

    def test_rms_value_itself_is_unchanged(self):
        """Guard against a swap that compares equal only in aggregate."""
        baseline, _, accelerated, _, _model = self._run_pair([0, 1, 2])
        for token, entry in baseline.state_dict()["entries"].items():
            fast = accelerated.state_dict()["entries"][token]
            with self.subTest(token=token):
                self.assertEqual(
                    math.sqrt(entry["delta"]["sum_squares"] / entry["delta"]["count"]),
                    math.sqrt(fast["delta"]["sum_squares"] / fast["delta"]["count"]),
                )


if __name__ == "__main__":
    unittest.main()
