import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from compose.adapters import ExpertManager, inject_compose_adapters
from compose.config import ComposeAdapterConfig
from compose.experts import ExpertPool, load_expert_checkpoint, save_expert_checkpoint


def _pool():
    model = nn.Sequential(nn.Linear(2, 2, bias=False))
    inject_compose_adapters(
        model, ComposeAdapterConfig(rank=1, alpha=2, target_modules=["0"])
    )
    return ExpertPool(ExpertManager(model))


class CheckpointTest(unittest.TestCase):
    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory)
            source = _pool()
            source.register(3, name="image-net-r", tags=["task1"])
            source.mark_steps(3, 2)
            source_layer = next(iter(source.manager.layers.values()))
            with torch.no_grad():
                source_layer.experts["3"].lora_A.weight.fill_(1.25)
                source_layer.experts["3"].lora_B.weight.fill_(-0.5)
            save_expert_checkpoint(source, str(checkpoint_path))

            target = _pool()
            manifest = load_expert_checkpoint(target, str(checkpoint_path))
            target_layer = next(iter(target.manager.layers.values()))
            self.assertEqual(manifest["format_version"], 1)
            self.assertEqual(target.get(3).trained_steps, 2)
            self.assertEqual(target.get(3).tags, ["task1"])
            torch.testing.assert_close(
                target_layer.experts["3"].lora_A.weight,
                source_layer.experts["3"].lora_A.weight,
            )
            with open(checkpoint_path / "compose_experts.json", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["adapter"]["rank"], 1)
