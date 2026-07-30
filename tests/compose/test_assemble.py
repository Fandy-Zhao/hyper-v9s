import tempfile
import unittest
from pathlib import Path

import torch

from compose.adapters import ExpertManager, inject_compose_adapters
from compose.config import ComposeAdapterConfig
from compose.experts import ExpertPool, load_expert_checkpoint, save_expert_checkpoint
from compose.experts.assemble import assemble_expert_checkpoint
from test_injection import TinyModel


def _pool():
    model = TinyModel(layer_count=32)
    inject_compose_adapters(model, ComposeAdapterConfig(rank=1, alpha=2))
    return ExpertPool(ExpertManager(model))


class AssembleCheckpointTest(unittest.TestCase):
    def test_assembles_selected_experts_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a, source_b, output = root / "a", root / "b", root / "output"
            first = _pool()
            first.register(0, name="a")
            for layer in first.manager.layers.values():
                layer.experts["0"].lora_A.weight.data.fill_(1.0)
                layer.experts["0"].lora_B.weight.data.fill_(2.0)
            save_expert_checkpoint(first, str(source_a))
            second = _pool()
            second.register(1, name="b")
            for layer in second.manager.layers.values():
                layer.experts["1"].lora_A.weight.data.fill_(3.0)
                layer.experts["1"].lora_B.weight.data.fill_(4.0)
            save_expert_checkpoint(second, str(source_b))

            manifest = assemble_expert_checkpoint(
                [(str(source_a), 0), (str(source_b), 1)], str(output)
            )
            self.assertEqual([item["expert_id"] for item in manifest["experts"]], [0, 1])
            self.assertEqual(manifest["metrics"]["adapter_tensor_count"], 896)
            target = _pool()
            loaded = load_expert_checkpoint(target, str(output))
            self.assertEqual(loaded["load_summary"]["loaded_tensor_count"], 896)
            for layer in target.manager.layers.values():
                torch.testing.assert_close(
                    layer.experts["0"].lora_B.weight, torch.full_like(layer.experts["0"].lora_B.weight, 2.0)
                )
                torch.testing.assert_close(
                    layer.experts["1"].lora_A.weight, torch.full_like(layer.experts["1"].lora_A.weight, 3.0)
                )

    def test_rejects_duplicate_ids_and_incompatible_adapters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _pool()
            first.register(0)
            save_expert_checkpoint(first, str(root / "a"))
            with self.assertRaisesRegex(ValueError, "duplicate expert id"):
                assemble_expert_checkpoint(
                    [(str(root / "a"), 0), (str(root / "a"), 0)],
                    str(root / "duplicate"),
                )

            model = TinyModel(layer_count=32)
            inject_compose_adapters(model, ComposeAdapterConfig(rank=2, alpha=2))
            second = ExpertPool(ExpertManager(model))
            second.register(1)
            save_expert_checkpoint(second, str(root / "b"))
            with self.assertRaisesRegex(ValueError, "incompatible adapter metadata"):
                assemble_expert_checkpoint(
                    [(str(root / "a"), 0), (str(root / "b"), 1)],
                    str(root / "incompatible"),
                )
