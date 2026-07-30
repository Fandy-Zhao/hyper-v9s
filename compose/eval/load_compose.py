import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional

import torch
import transformers

from compose.adapters import (
    ExpertManager,
    decoder_projection_names,
    inject_compose_adapters,
    validate_compose_injection,
)
from compose.config import ComposeAdapterConfig
from compose.experts import ExpertPool, load_expert_checkpoint
from compose.experts.checkpoint import MANIFEST_NAME
from compose.model import ComposeLlavaForCausalLM, load_compose_config


@dataclass
class EvaluationBundle:
    model: torch.nn.Module
    tokenizer: transformers.PreTrainedTokenizer
    image_processor: object
    context_length: int
    load_summary: Dict[str, object]
    expert_pool: Optional[ExpertPool] = None


def _foundation(
    model_path: str,
    vision_tower: str,
    projector_path: str,
    device: str,
    dtype: torch.dtype,
    model_max_length: int,
):
    if not os.path.isdir(model_path):
        raise FileNotFoundError("model path does not exist: {}".format(model_path))
    if not os.path.isdir(vision_tower):
        raise FileNotFoundError("vision tower does not exist: {}".format(vision_tower))
    if not os.path.isfile(projector_path):
        raise FileNotFoundError("projector checkpoint does not exist: {}".format(projector_path))

    config = load_compose_config(model_path)
    config.mm_vision_tower = vision_tower
    config.mm_vision_select_layer = -2
    config.mm_vision_select_feature = "patch"
    config.mm_projector_type = "mlp2x_gelu"
    model = ComposeLlavaForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=dtype
    )
    model.get_model().initialize_vision_modules(
        SimpleNamespace(
            vision_tower=vision_tower,
            mm_vision_select_layer=-2,
            mm_vision_select_feature="patch",
            mm_projector_type="mlp2x_gelu",
            pretrain_mm_mlp_adapter=projector_path,
        )
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    model.config.image_aspect_ratio = "pad"
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_use_im_start_end = False
    model.config.mm_use_im_patch_token = False
    model.initialize_vision_tokenizer(
        SimpleNamespace(mm_use_im_start_end=False, mm_use_im_patch_token=False),
        tokenizer,
    )
    model.to(device=torch.device(device), dtype=dtype)
    model.eval()
    return model, tokenizer, model.get_vision_tower().image_processor


def _read_compose_manifest(checkpoint_dir: str) -> Dict[str, object]:
    path = os.path.join(checkpoint_dir, MANIFEST_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError("Compose manifest does not exist: {}".format(path))
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    adapter = manifest.get("adapter")
    if not isinstance(adapter, dict):
        raise ValueError("Compose manifest is missing adapter configuration")
    for field in ("rank", "alpha", "dropout", "layers"):
        if field not in adapter:
            raise ValueError("Compose manifest adapter is missing {!r}".format(field))
    return manifest


def load_compose_model(
    model_path: str,
    checkpoint_dir: str,
    vision_tower: str,
    projector_path: str,
    expert_id: int = 0,
    gate: float = 1.0,
    normalization: str = "none",
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    model_max_length: int = 2048,
) -> EvaluationBundle:
    manifest = _read_compose_manifest(checkpoint_dir)
    adapter = manifest["adapter"]
    model, tokenizer, image_processor = _foundation(
        model_path, vision_tower, projector_path, device, dtype, model_max_length
    )
    adapter_config = ComposeAdapterConfig(
        rank=int(adapter["rank"]),
        alpha=float(adapter["alpha"]),
        dropout=float(adapter["dropout"]),
    )
    injected = inject_compose_adapters(model, adapter_config)
    injection_summary = validate_compose_injection(model, injected)
    manager = ExpertManager(model)
    pool = ExpertPool(manager)
    loaded_manifest = load_expert_checkpoint(pool, checkpoint_dir)
    if expert_id not in pool.expert_ids():
        raise KeyError("expert {} is not present in checkpoint".format(expert_id))
    pool.train_only([])
    manager.set_default_selection([expert_id], [gate], normalization=normalization)
    model.eval()
    return EvaluationBundle(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        context_length=model_max_length,
        expert_pool=pool,
        load_summary={
            "adapter_kind": "compose",
            "checkpoint": checkpoint_dir,
            "expert_id": expert_id,
            "gate": gate,
            "normalization": normalization,
            "injection": injection_summary,
            "checkpoint_load": loaded_manifest["load_summary"],
            "adapter_parameter_count": loaded_manifest["metrics"][
                "adapter_parameter_count"
            ],
        },
    )


def load_peft_model(
    model_path: str,
    checkpoint_dir: str,
    vision_tower: str,
    projector_path: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    model_max_length: int = 2048,
) -> EvaluationBundle:
    from peft import PeftConfig, PeftModel

    config = PeftConfig.from_pretrained(checkpoint_dir)
    expected_targets = None
    model, tokenizer, image_processor = _foundation(
        model_path, vision_tower, projector_path, device, dtype, model_max_length
    )
    expected_targets = set(decoder_projection_names(model))
    actual_targets = set(config.target_modules or [])
    if actual_targets != expected_targets:
        raise ValueError(
            "PEFT checkpoint target boundary mismatch; missing={}, unexpected={}".format(
                sorted(expected_targets - actual_targets),
                sorted(actual_targets - expected_targets),
            )
        )
    model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=False)
    model.to(device=torch.device(device), dtype=dtype)
    model.eval()
    adapter_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    )
    return EvaluationBundle(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        context_length=model_max_length,
        load_summary={
            "adapter_kind": "peft",
            "checkpoint": checkpoint_dir,
            "rank": int(config.r),
            "alpha": float(config.lora_alpha),
            "dropout": float(config.lora_dropout),
            "target_count": len(actual_targets),
            "adapter_parameter_count": adapter_parameters,
        },
    )
