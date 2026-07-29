import unittest

import torch
import torch.nn as nn

from compose.adapters.inject import inject_compose_adapters, validate_compose_injection
from compose.adapters.lora import ComposeLinear
from compose.config import ComposeAdapterConfig


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.Linear(3, 3, bias=False))


class TinyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, name, nn.Linear(3, 3, bias=False))


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()
        self.mlp = TinyMlp()


class TinyDecoder(nn.Module):
    def __init__(self, layer_count=2):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer() for _ in range(layer_count)])
        self.mm_projector = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
        self.vision_tower = TinyAttention()


class TinyModel(nn.Module):
    def __init__(self, layer_count=2):
        super().__init__()
        self.model = TinyDecoder(layer_count)
        self.lm_head = nn.Linear(3, 3)

    def get_model(self):
        return self.model


class InjectionTest(unittest.TestCase):
    def test_injection_targets_decoder_projections_only(self):
        model = TinyModel(layer_count=2)
        original_weight = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
        matches = inject_compose_adapters(model, ComposeAdapterConfig(rank=2, alpha=4))
        summary = validate_compose_injection(model, matches)

        self.assertEqual(len(matches), 14)
        self.assertEqual(summary["decoder_layers"], 2)
        self.assertEqual(summary["injected_layers"], 14)
        self.assertEqual(summary["expected_layers"], 14)
        self.assertEqual(summary["vision_tower_injected"], 0)
        self.assertEqual(summary["mm_projector_injected"], 0)
        self.assertEqual(summary["lm_head_injected"], 0)
        self.assertIsInstance(model.model.layers[0].self_attn.q_proj, ComposeLinear)
        self.assertIsInstance(model.model.vision_tower.q_proj, nn.Linear)
        self.assertIsInstance(model.model.mm_projector[0], nn.Linear)
        self.assertIsInstance(model.lm_head, nn.Linear)
        torch.testing.assert_close(
            model.model.layers[0].self_attn.q_proj.weight, original_weight
        )

    def test_duplicate_injection_fails(self):
        model = TinyModel(layer_count=1)
        config = ComposeAdapterConfig()
        inject_compose_adapters(model, config)
        with self.assertRaisesRegex(ValueError, "already injected"):
            inject_compose_adapters(model, config)

    def test_32_layer_decoder_injects_exactly_224_projections(self):
        model = TinyModel(layer_count=32)
        matches = inject_compose_adapters(model, ComposeAdapterConfig())
        self.assertEqual(len(matches), 224)
        self.assertEqual(validate_compose_injection(model)["expected_layers"], 224)

    def test_noncanonical_target_set_fails(self):
        with self.assertRaisesRegex(ValueError, "requires exactly"):
            inject_compose_adapters(
                TinyModel(), ComposeAdapterConfig(target_modules=["q_proj"])
            )
