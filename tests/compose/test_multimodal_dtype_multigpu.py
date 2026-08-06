"""Regression: encode_images dtype mismatch under multi-GPU DataParallel.

DataParallel replica threads lose the autocast context, so an fp32 vision
tower output feeding a bf16 mm_projector raised
``RuntimeError: mat1 and mat2 must have the same dtype`` on 4-GPU runs.
The fix casts features to the projector weight dtype in encode_images.
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from compose.model.multimodal_arch import ComposeLlavaMetaForCausalLM


class _FakeVisionTower:
    hidden_size = 4

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        # Emit fp32 regardless of input dtype (CLIP output behavior).
        return images.float()


class _FakeMMProjector(torch.nn.Module):
    """Mirrors the real mm_projector: a Sequential (mlp2x_gelu), so it has
    no ``.weight`` attribute; dtype must come from parameters()."""

    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4, 4, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(4, 4, bias=False),
        ).to(torch.bfloat16)

    def forward(self, features):
        return self.mlp(features)


class _FakeModel:
    def __init__(self):
        self.mm_projector = _FakeMMProjector()
        self._vision_tower = _FakeVisionTower()

    def get_vision_tower(self):
        return self._vision_tower


class _FakeLLaVA(ComposeLlavaMetaForCausalLM):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self._model = _FakeModel()
        self.device = torch.device("cpu")
        self.dtype = dtype

    def get_model(self):
        return self._model

    def get_vision_tower(self):
        return self._model.get_vision_tower()


def test_encode_images_fp32_features_bf16_projector():
    model = _FakeLLaVA()
    images = torch.randn(2, 4, dtype=torch.bfloat16)
    # Pre-fix this raised "mat1 and mat2 must have the same dtype".
    output = model.encode_images(images)
    assert output.dtype == torch.bfloat16
    assert output.shape == (2, 4)


def test_encode_images_bf16_features_bf16_projector():
    model = _FakeLLaVA()
    images = torch.randn(3, 4, dtype=torch.bfloat16)
    output = model.encode_images(images)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


def test_encode_images_fp32_features_fp32_projector():
    model = _FakeLLaVA(dtype=torch.float32)
    model._model.mm_projector.mlp = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(4, 4, bias=False),
    ).to(torch.float32)
    images = torch.randn(2, 4, dtype=torch.float32)
    output = model.encode_images(images)
    assert output.dtype == torch.float32


def test_encode_images_dtype_from_model_not_projector_params():
    """DataParallel replicas keep frozen module weights as plain tensor
    attributes (not Parameters in ``_parameters``); dtype must come from
    the model, not from ``projector.parameters()``."""
    model = _FakeLLaVA(dtype=torch.bfloat16)
    # Simulate the replica: move weights out of _parameters into plain
    # tensor attributes (the torch replicate behavior for frozen params).
    for module in model._model.mm_projector.mlp.modules():
        if isinstance(module, torch.nn.Linear):
            weight = module._parameters.pop("weight")
            module.weight = weight.detach()
            if "bias" in module._parameters:
                bias = module._parameters.pop("bias")
                if bias is not None:
                    module.bias = bias.detach()
                else:
                    module.register_parameter("bias", None)
    assert sum(1 for _ in model._model.mm_projector.parameters()) == 0
    images = torch.randn(2, 4, dtype=torch.bfloat16)
    output = model.encode_images(images)
    assert output.dtype == torch.bfloat16
    assert output.shape == (2, 4)
