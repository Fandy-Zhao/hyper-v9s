"""Tests for the inference-only expert-subset load.

``load_expert_checkpoint(..., keep_ids=...)`` exists so an evaluation on a
shared GPU can skip instantiating experts a routing manifest never selects.
That is only legitimate if it is *exactly* a memory saving -- so the load
tests here are about the two ways it could go wrong instead: silently
weakening the checkpoint validation, and silently producing a different
output.  The tests that the pool is still described in full and that a
subset pool cannot be saved are the ones that keep the lever honest.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from compose.adapters import ExpertManager, inject_compose_adapters
from compose.config import ComposeAdapterConfig
from compose.experts import ExpertPool, load_expert_checkpoint, save_expert_checkpoint
from test_injection import TinyModel
from test_checkpoint import _decoder_output


def _model_and_pool():
    torch.manual_seed(7)
    model = TinyModel(layer_count=32)
    inject_compose_adapters(model, ComposeAdapterConfig(rank=1, alpha=2))
    return model, ExpertPool(ExpertManager(model))


def _saved_checkpoint(directory, expert_ids=(3, 5, 9)):
    """A checkpoint whose experts all differ, so mixing them up is visible."""
    model, pool = _model_and_pool()
    for expert_id in expert_ids:
        pool.register(expert_id, name="expert-{}".format(expert_id))
        for layer in pool.manager.layers.values():
            with torch.no_grad():
                layer.experts[str(expert_id)].lora_A.weight.fill_(0.1 * expert_id)
                layer.experts[str(expert_id)].lora_B.weight.fill_(0.5 - 0.01 * expert_id)
    save_expert_checkpoint(pool, str(directory))
    return model


class SubsetLoadTest(unittest.TestCase):
    def test_subset_load_matches_full_load_output(self):
        """Selecting only kept experts must be bit-identical to the full pool."""
        inputs = torch.randn(2, 4, 3)
        with tempfile.TemporaryDirectory() as directory:
            _saved_checkpoint(Path(directory))

            full_model, full = _model_and_pool()
            load_expert_checkpoint(full, str(directory))
            full.manager.set_default_selection([3], [1.0])
            full_output = _decoder_output(full_model, inputs)

            subset_model, subset = _model_and_pool()
            load_expert_checkpoint(subset, str(directory), keep_ids=[3])
            subset.manager.set_default_selection([3], [1.0])
            subset_output = _decoder_output(subset_model, inputs)

        torch.testing.assert_close(subset_output, full_output)

    def test_subset_load_keeps_metadata_and_validation_for_the_whole_pool(self):
        """Metadata and tensor accounting still describe every expert."""
        with tempfile.TemporaryDirectory() as directory:
            _saved_checkpoint(Path(directory))
            model, pool = _model_and_pool()
            manifest = load_expert_checkpoint(pool, str(directory), keep_ids=[5])

        # Only expert 5 has modules; all three are still described.
        self.assertEqual(pool.manager.expert_ids(), [5])
        self.assertEqual(pool.expert_ids(), [3, 5, 9])
        self.assertEqual(pool.get(3).name, "expert-3")
        # The count still covers the whole checkpoint, so a truncated weights
        # file would be caught instead of passing against a smaller key set.
        self.assertEqual(
            manifest["load_summary"]["expected_tensor_count"],
            manifest["metrics"]["adapter_tensor_count"],
        )
        self.assertEqual(manifest["load_summary"]["instantiated_experts"], [5])
        self.assertEqual(manifest["load_summary"]["declared_experts"], 3)

    def test_selecting_an_uninstantiated_expert_fails_loudly(self):
        """A keep set that omits a selected expert must raise, not misroute."""
        with tempfile.TemporaryDirectory() as directory:
            _saved_checkpoint(Path(directory))
            model, pool = _model_and_pool()
            load_expert_checkpoint(pool, str(directory), keep_ids=[3])
            with self.assertRaises(KeyError):
                pool.manager.set_default_selection([5], [1.0])

    def test_subset_pool_refuses_to_save(self):
        """A partial pool must never overwrite a checkpoint on disk."""
        with tempfile.TemporaryDirectory() as directory:
            _saved_checkpoint(Path(directory))
            model, pool = _model_and_pool()
            load_expert_checkpoint(pool, str(directory), keep_ids=[3])
            with self.assertRaises(ValueError):
                save_expert_checkpoint(pool, str(Path(directory) / "partial"))

    def test_default_none_loads_every_expert(self):
        with tempfile.TemporaryDirectory() as directory:
            _saved_checkpoint(Path(directory))
            model, pool = _model_and_pool()
            manifest = load_expert_checkpoint(pool, str(directory))
        self.assertEqual(pool.manager.expert_ids(), [3, 5, 9])
        self.assertEqual(manifest["load_summary"]["instantiated_experts"], [3, 5, 9])


class ManifestUnionTest(unittest.TestCase):
    def test_union_covers_every_id_any_sample_selects(self):
        """The keep set must be a superset of what the run can select."""
        import json

        from compose.eval.eval_task import manifest_expert_union

        payload = {
            "v7_t0_val_0": {"global_top2": [3, 7]},
            "v7_t0_val_1": {"global_top2": [7, 2]},
            "v7_t0_val_2": {"global_top2": [0, 0]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            union = manifest_expert_union(str(path))
        self.assertEqual(union, [0, 2, 3, 7])
        for row in payload.values():
            for expert_id in row["global_top2"]:
                self.assertIn(expert_id, union)


if __name__ == "__main__":
    unittest.main()
