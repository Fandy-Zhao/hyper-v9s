import json
import tempfile
import unittest
from pathlib import Path

import torch

from compose.adapters import ExpertManager, inject_compose_adapters
from compose.config import ComposeAdapterConfig
from compose.experts import ExpertPool, load_expert_checkpoint, save_expert_checkpoint
from test_injection import TinyModel


def _model_and_pool():
    torch.manual_seed(7)
    model = TinyModel(layer_count=32)
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


class CheckpointTest(unittest.TestCase):
    def test_checkpoint_round_trip_is_exact_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory)
            source_model, source = _model_and_pool()
            source.register(3, name="image-net-r", tags=["task1"])
            source.mark_steps(3, 2)
            for layer in source.manager.layers.values():
                with torch.no_grad():
                    layer.experts["3"].lora_A.weight.fill_(1.25)
                    layer.experts["3"].lora_B.weight.fill_(-0.5)
            source.manager.set_default_selection([3])
            inputs = torch.randn(2, 4, 3)
            source_output = _decoder_output(source_model, inputs)
            save_expert_checkpoint(source, str(checkpoint_path))

            target_model, target = _model_and_pool()
            manifest = load_expert_checkpoint(target, str(checkpoint_path))
            target.manager.set_default_selection([3])
            target_output = _decoder_output(target_model, inputs)

            self.assertEqual(manifest["format_version"], 1)
            self.assertEqual(manifest["metrics"]["adapter_tensor_count"], 448)
            self.assertNotIn("normalization", manifest)
            self.assertGreater(manifest["metrics"]["adapter_parameter_count"], 0)
            self.assertGreater(manifest["metrics"]["checkpoint_bytes"], 0)
            self.assertEqual(manifest["load_summary"]["loaded_tensor_count"], 448)
            self.assertEqual(target.get(3).trained_steps, 2)
            self.assertEqual(target.get(3).tags, ["task1"])
            torch.testing.assert_close(target_output, source_output)
            with open(checkpoint_path / "compose_experts.json", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["adapter"]["rank"], 1)

    def test_checkpoint_rejects_missing_or_unexpected_tensors(self):
        for mutation in ("missing", "unexpected"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                _, source = _model_and_pool()
                source.register(3)
                save_expert_checkpoint(source, directory)
                weights_path = Path(directory) / "compose_experts.bin"
                state = torch.load(weights_path, map_location="cpu")
                if mutation == "missing":
                    state.pop(next(iter(state)))
                else:
                    state["vision_tower.experts.3.lora_A.weight"] = torch.zeros(1)
                torch.save(state, weights_path)
                _, target = _model_and_pool()
                with self.assertRaisesRegex(ValueError, "keys do not match exactly"):
                    load_expert_checkpoint(target, directory)
