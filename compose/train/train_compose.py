import os
from typing import List, Optional

import torch
import transformers

from llava import conversation as conversation_lib

from compose.adapters import ExpertManager, inject_compose_adapters
from compose.config import ComposeAdapterConfig
from compose.experts import ExpertPool, load_expert_checkpoint, save_expert_checkpoint
from compose.model import ComposeLlavaConfig, ComposeLlavaForCausalLM

from .arguments import DataArguments, ModelArguments, TrainingArguments
from .data import make_supervised_data_module
from .trainer import ComposeTrainer


def _csv_ints(value: str) -> List[int]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("compose_expert_ids must not be empty")
    return [int(item) for item in values]


def _csv_floats(value: str) -> Optional[List[float]]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    return [float(item) for item in values] if values else None


def train() -> None:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if model_args.vision_tower is None:
        raise ValueError("Compose foundation requires --vision_tower")

    config = ComposeLlavaConfig.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir
    )
    config.mm_vision_tower = model_args.vision_tower
    config.mm_vision_select_layer = model_args.mm_vision_select_layer
    config.mm_vision_select_feature = model_args.mm_vision_select_feature
    config.mm_projector_type = model_args.mm_projector_type
    model = ComposeLlavaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    model.config.use_cache = False
    model.get_model().initialize_vision_modules(model_args, fsdp=training_args.fsdp)
    vision_tower = model.get_vision_tower()
    vision_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    vision_tower.to(device=training_args.device, dtype=vision_dtype)

    adapter_config = ComposeAdapterConfig(
        rank=model_args.compose_rank,
        alpha=model_args.compose_alpha,
        dropout=model_args.compose_dropout,
        target_modules=[
            item.strip()
            for item in model_args.compose_target_modules.split(",")
            if item.strip()
        ],
    )
    injected = inject_compose_adapters(model, adapter_config)
    manager = ExpertManager(model)
    pool = ExpertPool(manager)
    selected_experts = _csv_ints(model_args.compose_expert_ids)
    gates = _csv_floats(model_args.compose_gates)
    if len(selected_experts) not in (1, 2):
        raise ValueError("Compose foundation trains one or two fixed experts")

    if model_args.compose_checkpoint:
        load_expert_checkpoint(pool, model_args.compose_checkpoint)
    for expert_id in selected_experts:
        if expert_id not in pool.expert_ids():
            pool.register(expert_id, name="task1-expert-{}".format(expert_id))
    pool.train_only(selected_experts)
    manager.set_default_selection(selected_experts, gates)

    if model_args.tune_mm_mlp_adapter:
        model.get_model().mm_projector.requires_grad_(True)
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    conversation_lib.default_conversation = conversation_lib.conv_templates.get(
        model_args.version, conversation_lib.conv_templates["vicuna_v1"]
    )
    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True
    data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.initialize_vision_tokenizer(model_args, tokenizer)

    data_module = make_supervised_data_module(tokenizer, data_args)
    trainer = ComposeTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        expert_pool=pool,
        **data_module
    )
    checkpoints = [
        name
        for name in os.listdir(training_args.output_dir)
        if name.startswith("checkpoint-")
    ] if os.path.isdir(training_args.output_dir) else []
    trainer.train(resume_from_checkpoint=bool(checkpoints))
    trainer.save_state()
    model.config.use_cache = True
    if training_args.should_save:
        pool.sync_training_step(trainer.state.global_step)
        model.config.save_pretrained(training_args.output_dir)
        save_expert_checkpoint(pool, training_args.output_dir)
    if training_args.local_rank in (-1, 0):
        print("Injected {} Compose layers; experts={}".format(len(injected), pool.expert_ids()))


if __name__ == "__main__":
    train()
