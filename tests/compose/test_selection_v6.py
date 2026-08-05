"""V6 Stage E1: unified empty / single / pair ComposeSelection.

Verifies the contract that one selection structure covers all three
cardinalities with a shared forward path, batch grouping, per-sample set
recording, checkpoint round trip and DDP-consistent construction.

Numerical equivalence uses output-difference identities (explicit incremental
summation): for any two experts A, B,
    out({A,B}) - out({}) == (out({A}) - out({})) + (out({B}) - out({}))
"""

import hashlib
import math
import tempfile
import unittest
from pathlib import Path

import torch

from compose.adapters import ExpertManager, inject_compose_adapters
from compose.adapters.runtime import use_selection
from compose.adapters.types import ComposeSelection
from compose.config import ComposeAdapterConfig
from compose.experts import ExpertPool, load_expert_checkpoint, save_expert_checkpoint
from test_injection import TinyModel


def _model_and_pool(seed: int = 7):
    torch.manual_seed(seed)
    model = TinyModel(layer_count=2)
    inject_compose_adapters(model, ComposeAdapterConfig(rank=1, alpha=2))
    return model, ExpertPool(ExpertManager(model))


def _decoder_output(model, inputs):
    hidden = inputs
    layer = model.model.layers[0]
    hidden = layer.self_attn.o_proj(
        layer.self_attn.q_proj(hidden)
        + layer.self_attn.k_proj(hidden)
        + layer.self_attn.v_proj(hidden)
    )
    return layer.mlp.down_proj(
        layer.mlp.gate_proj(hidden) + layer.mlp.up_proj(hidden)
    )


def _first_layer(pool):
    return list(pool.manager.layers.values())[0]


class ComposeSelectionV6Test(unittest.TestCase):
    def test_empty_equals_backbone_only(self):
        model, pool = _model_and_pool()
        pool.register(0)
        inputs = torch.randn(3, 4, 3)
        with pool.manager.selection_context(pool.make_selection([], batch_size=3)):
            selected_output = _decoder_output(model, inputs)
        base_output = _decoder_output(model, inputs)
        torch.testing.assert_close(selected_output, base_output)

    def test_single_equals_original_single_adapter_api(self):
        # The old API was manager.set_default_selection([x], [w]); the new
        # padded selection must produce bit-identical output.
        model, pool = _model_and_pool()
        pool.register(0)
        layer = _first_layer(pool)
        with torch.no_grad():
            layer.experts["0"].lora_A.weight.fill_(0.5)
            layer.experts["0"].lora_B.weight.fill_(-0.25)
        inputs = torch.randn(2, 4, 3)

        pool.manager.set_default_selection([0], [1.0], normalization="none")
        legacy_output = _decoder_output(model, inputs)
        pool.manager.clear_default_selection()

        with pool.manager.selection_context(pool.make_selection([0], batch_size=2)):
            unified_output = _decoder_output(model, inputs)
        torch.testing.assert_close(unified_output, legacy_output)

    def test_pair_equals_explicit_incremental_sum(self):
        # out({A,B}) - out({}) == (out({A}) - out({})) + (out({B}) - out({}))
        model, pool = _model_and_pool()
        pool.register(0)
        pool.register(1)
        layer = _first_layer(pool)
        with torch.no_grad():
            layer.experts["0"].lora_A.weight.fill_(1.0)
            layer.experts["0"].lora_B.weight.fill_(2.0)
            layer.experts["1"].lora_A.weight.fill_(0.5)
            layer.experts["1"].lora_B.weight.fill_(-0.5)
        inputs = torch.randn(2, 4, 3)
        weight = 1.0 / math.sqrt(2)

        def forward_with(expert_ids, gates, normalization="none"):
            with pool.manager.selection_context(
                pool.make_selection(
                    expert_ids, batch_size=2, gates=gates, normalization=normalization
                )
            ):
                return _decoder_output(model, inputs)

        base = forward_with([], [])
        single_a = forward_with([0], [1.0])
        single_b = forward_with([1], [1.0])
        # Pair with unit gates equals the explicit incremental sum.
        pair = forward_with([0, 1], [1.0, 1.0])
        torch.testing.assert_close(pair, base + (single_a - base) + (single_b - base))
        # Pair with l2-normalized gates scales the incremental sum by 1/sqrt(2).
        pair_l2 = forward_with([0, 1], [1.0, 1.0], normalization="l2")
        torch.testing.assert_close(
            pair_l2, base + ((single_a - base) + (single_b - base)) * weight
        )

    def test_heterogeneous_batch(self):
        model, pool = _model_and_pool()
        pool.register(0)
        pool.register(1)
        layer = _first_layer(pool)
        with torch.no_grad():
            layer.experts["0"].lora_A.weight.fill_(1.0)
            layer.experts["0"].lora_B.weight.fill_(2.0)
            layer.experts["1"].lora_A.weight.fill_(1.0)
            layer.experts["1"].lora_B.weight.fill_(4.0)
        inputs = torch.randn(4, 4, 3)
        selection = ComposeSelection(
            torch.tensor([[-1, -1], [0, -1], [0, 1], [1, -1]], dtype=torch.long),
            torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0]],
                dtype=torch.float32,
            ),
            normalization="none",
        )
        with use_selection(selection):
            output = _decoder_output(model, inputs)
        base = _decoder_output(model, inputs)

        # Per-row equivalence via the incremental-sum identity.
        rows = [[], [0], [0, 1], [1]]
        gates_by_row = [[], [1.0], [1.0, 1.0], [1.0]]
        for index, (expert_ids, row_gates) in enumerate(zip(rows, gates_by_row)):
            expected = base[index].clone()
            for expert_id, gate in zip(expert_ids, row_gates):
                with pool.manager.selection_context(
                    pool.make_selection([expert_id], batch_size=1, gates=[gate])
                ):
                    single_output = _decoder_output(model, inputs[index: index + 1])
                expected = expected + (single_output[0] - base[index])
            torch.testing.assert_close(output[index], expected)

    def test_batch_expert_dedup_loading(self):
        model, pool = _model_and_pool()
        pool.register(0)
        layer = _first_layer(pool)
        calls = {"count": 0}
        original_forward = layer.experts["0"].forward

        def counted_forward(inputs):
            calls["count"] += 1
            return original_forward(inputs)

        layer.experts["0"].forward = counted_forward
        inputs = torch.randn(4, 4, 3)
        selection = ComposeSelection(
            torch.tensor([[0, -1], [0, -1], [0, -1], [0, -1]], dtype=torch.long),
            torch.tensor(
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                dtype=torch.float32,
            ),
        )
        with use_selection(selection):
            _decoder_output(model, inputs)
        # One grouped call per unique expert, not one per sample.
        self.assertEqual(calls["count"], 1)

    def test_old_expert_plus_candidate_forward(self):
        model, pool = _model_and_pool()
        pool.register(0)  # committed old expert
        pool.manager.add_expert(1)  # candidate
        layer = _first_layer(pool)
        with torch.no_grad():
            layer.experts["0"].lora_A.weight.fill_(1.0)
            layer.experts["0"].lora_B.weight.fill_(2.0)
            layer.experts["1"].lora_A.weight.fill_(0.5)
            layer.experts["1"].lora_B.weight.fill_(0.5)
        inputs = torch.randn(2, 4, 3)
        selection = ComposeSelection(
            torch.tensor([[0, 1], [-1, -1]], dtype=torch.long),
            torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32),
        )
        with use_selection(selection):
            output = _decoder_output(model, inputs)
        base = _decoder_output(model, inputs)
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(torch.allclose(output[0], base[0]))
        # Empty row stays backbone-only even when other rows select experts.
        torch.testing.assert_close(output[1], base[1])

    def test_unselected_expert_not_computed(self):
        model, pool = _model_and_pool()
        pool.register(0)
        pool.register(1)
        layer = _first_layer(pool)
        calls = {"count": 0}
        original_forward = layer.experts["1"].forward

        def counted_forward(inputs):
            calls["count"] += 1
            return original_forward(inputs)

        layer.experts["1"].forward = counted_forward
        inputs = torch.randn(2, 4, 3)
        with pool.manager.selection_context(pool.make_selection([0], batch_size=2)):
            _decoder_output(model, inputs)
        self.assertEqual(calls["count"], 0)

    def test_selected_candidate_has_gradient_old_expert_frozen(self):
        model, pool = _model_and_pool()
        pool.register(0)  # committed old expert
        pool.manager.add_expert(1)  # candidate
        pool.train_only([1])
        for layer in pool.manager.layers.values():
            self.assertFalse(layer.experts["0"].lora_A.weight.requires_grad)
            self.assertFalse(layer.experts["0"].lora_B.weight.requires_grad)
            self.assertTrue(layer.experts["1"].lora_A.weight.requires_grad)
            self.assertTrue(layer.experts["1"].lora_B.weight.requires_grad)
        inputs = torch.randn(2, 4, 3)
        selection = ComposeSelection(
            torch.tensor([[0, 1], [1, -1]], dtype=torch.long),
            torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
        )
        with use_selection(selection):
            output = _decoder_output(model, inputs)
        loss = output.square().mean()
        loss.backward()
        # Only the first decoder layer participates in _decoder_output; the
        # second layer is never executed and therefore has no gradients.
        executed_layers = list(pool.manager.layers.values())[:7]
        for layer in executed_layers:
            for name in ("lora_A", "lora_B"):
                old_weight = getattr(layer.experts["0"], name).weight
                self.assertIsNone(old_weight.grad)
                cand_weight = getattr(layer.experts["1"], name).weight
                self.assertIsNotNone(cand_weight.grad)
                self.assertTrue(torch.isfinite(cand_weight.grad).all())

    def test_per_sample_sets_recording(self):
        selection = ComposeSelection(
            torch.tensor([[-1, -1], [0, -1], [0, 1]], dtype=torch.long),
            torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32
            ),
            normalization="l2",
        )
        records = selection.per_sample_sets()
        self.assertEqual(records[0], {"expert_ids": [], "gates": []})
        self.assertEqual(records[1], {"expert_ids": [0], "gates": [1.0]})
        # float32 storage: compare with tolerance.
        self.assertEqual(records[2]["expert_ids"], [0, 1])
        for gate in records[2]["gates"]:
            self.assertAlmostEqual(gate, 1.0 / math.sqrt(2), places=6)
        canonical = selection.canonical_sets()
        self.assertEqual(canonical[2][0], (0, 1))
        for gate in canonical[2][1]:
            self.assertAlmostEqual(gate, 1.0 / math.sqrt(2), places=6)

    def test_save_reload_roundtrip_preserves_selection_output(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory)
            source_model, source = _model_and_pool()
            source.register(0)
            source.register(1)
            for layer in source.manager.layers.values():
                with torch.no_grad():
                    layer.experts["0"].lora_A.weight.fill_(1.0)
                    layer.experts["0"].lora_B.weight.fill_(2.0)
                    layer.experts["1"].lora_A.weight.fill_(0.5)
                    layer.experts["1"].lora_B.weight.fill_(-0.5)
            save_expert_checkpoint(source, str(checkpoint_path))

            target_model, target = _model_and_pool()
            load_expert_checkpoint(target, str(checkpoint_path))
            inputs = torch.randn(3, 4, 3)
            for expert_ids, gates in [([], []), ([0], [1.0]), ([0, 1], [1.0, 1.0])]:
                with source.manager.selection_context(
                    source.make_selection(expert_ids, batch_size=3, gates=gates)
                ):
                    source_output = _decoder_output(source_model, inputs)
                with target.manager.selection_context(
                    target.make_selection(expert_ids, batch_size=3, gates=gates)
                ):
                    target_output = _decoder_output(target_model, inputs)
                torch.testing.assert_close(target_output, source_output)

    def test_ddp_consistent_construction(self):
        # Selections are constructed from pure integers on CPU and moved to
        # the device afterwards; identical on every rank by construction
        # (same pattern as the DDP state-fingerprint tests).
        model, pool = _model_and_pool()
        pool.register(0)
        pool.register(1)
        first = pool.make_selection([0, 1], batch_size=4, gates=[1.0, 1.0])
        second = pool.make_selection([0, 1], batch_size=4, gates=[1.0, 1.0])
        torch.testing.assert_close(first.expert_ids, second.expert_ids)
        torch.testing.assert_close(first.gates, second.gates)
        self.assertEqual(first.per_sample_sets(), second.per_sample_sets())
        # CPU-constructed selection is valid on any device.
        moved = first.to(torch.device("cpu"))
        torch.testing.assert_close(moved.expert_ids, first.expert_ids)

    def test_frozen_old_expert_hash_unchanged(self):
        model, pool = _model_and_pool()
        pool.register(0)
        pool.manager.add_expert(1)

        def weights_hash():
            payload = b""
            for layer in pool.manager.layers.values():
                for name in ("lora_A", "lora_B"):
                    payload += (
                        getattr(layer.experts["0"], name)
                        .weight.detach()
                        .cpu()
                        .numpy()
                        .tobytes()
                    )
            return hashlib.sha256(payload).hexdigest()

        before = weights_hash()
        pool.train_only([1])
        inputs = torch.randn(2, 4, 3)
        selection = ComposeSelection(
            torch.tensor([[0, 1], [1, -1]], dtype=torch.long),
            torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
        )
        with use_selection(selection):
            output = _decoder_output(model, inputs)
        output.square().mean().backward()
        self.assertEqual(before, weights_hash())


if __name__ == "__main__":
    unittest.main()
