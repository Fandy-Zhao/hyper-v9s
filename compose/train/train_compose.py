"""Compose conditional-residual expert training (single unified entry).

``--compose-mode`` selects the routing strategy:

- ``fixed``: one or two fixed experts compose the forward for every sample
  (foundation / baseline training);
- ``cluster_expert``: per-sample cluster-wise conditional-residual training.
  Each sample routes to ``old_teacher_set + cluster_expert`` via the
  ``compose_selections`` batch key (ComposeSelectionDataset /
  ComposeSelectionCollator); only the cluster expert receives gradient; the
  base model, vision encoder, projector and all historical experts stay
  frozen. Old experts are loaded from ``--compose-checkpoint``, new cluster
  experts are registered as trainable, and the training job ends by writing
  a standalone per-expert state dict (``expert_<id>.pt``) for the commit
  transaction.

This module is the only formal Compose trainer: the legacy candidate-only
entry is merged here and removed.
"""

import json
import os
import sys
from typing import List, Optional

import torch
import transformers

from llava import conversation as conversation_lib

from compose.adapters import (
    ExpertManager,
    inject_compose_adapters,
    validate_compose_injection,
)
from compose.config import ComposeAdapterConfig
from compose.experts import ExpertPool, load_expert_checkpoint, save_expert_checkpoint
from compose.model import ComposeLlavaForCausalLM, load_compose_config

from .arguments import DataArguments, ModelArguments, TrainingArguments
from .data import (
    ComposeSelectionCollator,
    ComposeSelectionDataset,
    DataCollatorForSupervisedDataset,
    LazySupervisedDataset,
    make_supervised_data_module,
)
from .trainer import ComposeTrainer


def _csv_ints(value: str) -> List[int]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expert id list must not be empty")
    return [int(item) for item in values]


def _csv_floats(value: str) -> Optional[List[float]]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    return [float(item) for item in values] if values else None


def _expert_origin_mapping(value: str):
    mapping = {}
    for item in (entry.strip() for entry in value.split(",") if entry.strip()):
        expert_id, separator, origin_task_id = item.partition("=")
        if not separator or not origin_task_id.strip():
            raise ValueError(
                "compose_existing_expert_origins entries must use EXPERT_ID=TASK_ID"
            )
        mapping[int(expert_id.strip())] = origin_task_id.strip()
    return mapping


def _load_selection_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    manifest = {}
    for row in rows:
        manifest[str(row["sample_id"])] = row
    return manifest


def _resolve_expert_roles(active_experts, trainable_value: str):
    active = [int(value) for value in active_experts]
    trainable = _csv_ints(trainable_value) if trainable_value.strip() else list(active)
    if not set(trainable).issubset(set(active)):
        raise ValueError("compose_trainable_expert_ids must be a subset of active experts")
    return active, trainable


def _prepare_cluster_expert_backward() -> None:
    """Make per-sample selections visible to checkpoint recomputation.

    Reentrant activation checkpointing (the torch default used by LLaVA)
    recomputes each checkpointed segment during ``loss.backward()``.
    Multithreaded autograd (on by default) dispatches that recomputation
    to autograd-engine worker threads, and ``ContextVar`` values do not
    propagate to those threads: ``use_selection`` becomes invisible, the
    recomputed graph is backbone-only, and the cluster expert receives no
    gradients (``lora_B`` stays at its zero initialization). The real
    smoke observed exactly this: ``Finite-gradient LoRA-B count: 0`` and
    committed experts whose ``lora_B.weight`` was exactly zero everywhere.

    Two guards, in order of importance:

    1. Pin the backward to the calling thread so the recompute runs with
       the selection context active.
    2. Refuse single-process multi-GPU DataParallel: ``nn.DataParallel``
       runs the forward on its own worker threads, where the selection
       context is invisible in *both* passes, silently training
       backbone-only experts. Distributed DDP (world_size > 1 via
       torchrun) is safe: each rank is its own process, its forward and
       backward run on the main thread, and the selection context stays
       visible in both passes (verified by the 4-GPU gradient-audit
       smoke, spec §9).
    """
    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    if not distributed and torch.cuda.device_count() > 1:
        raise ValueError(
            "cluster_expert mode requires exactly one visible GPU per "
            "process (CUDA_VISIBLE_DEVICES with a single device), or a "
            "distributed launch (torchrun, one GPU per rank): "
            "nn.DataParallel runs the model forward on worker threads "
            "where the per-sample selection context is invisible, which "
            "silently trains backbone-only experts. Got {} visible GPUs.".format(
                torch.cuda.device_count()
            )
        )
    torch.autograd.set_multithreading_enabled(False)


def _build_model(model_args, training_args):
    config = load_compose_config(
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
    return model, vision_tower


def _inject_and_pool(model, model_args):
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
    injection_summary = validate_compose_injection(model, injected)
    manager = ExpertManager(model)
    pool = ExpertPool(manager)
    return injected, injection_summary, manager, pool


def _load_old_checkpoint(pool, model_args, training_args):
    if not model_args.compose_checkpoint:
        return None
    load_summary = load_expert_checkpoint(pool, model_args.compose_checkpoint)[
        "load_summary"
    ]
    for expert_id, origin_task_id in _expert_origin_mapping(
        model_args.compose_existing_expert_origins
    ).items():
        metadata = pool.get(expert_id)
        if metadata.origin_task_id not in (None, origin_task_id):
            raise ValueError(
                "expert {} origin task mismatch; checkpoint={!r}, requested={!r}".format(
                    expert_id, metadata.origin_task_id, origin_task_id
                )
            )
        metadata.origin_task_id = origin_task_id
    if training_args.local_rank in (-1, 0):
        print("Compose checkpoint load summary: {}".format(load_summary))
    return load_summary


def _register_new_experts(pool, expert_ids, model_args):
    for expert_id in expert_ids:
        if expert_id not in pool.expert_ids():
            pool.register(
                expert_id,
                name=(model_args.compose_expert_name or "expert-{:04d}".format(expert_id)),
                origin_task_id=model_args.compose_origin_task_id,
                source_checkpoint=model_args.compose_checkpoint,
                tags=[
                    value.strip()
                    for value in model_args.compose_expert_tags.split(",")
                    if value.strip()
                ],
            )


def _check_expected_parameters(model, model_args):
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if (
        model_args.expected_adapter_parameters is not None
        and trainable_parameter_count != model_args.expected_adapter_parameters
    ):
        raise ValueError(
            "trainable parameter count mismatch; expected={}, actual={}".format(
                model_args.expected_adapter_parameters, trainable_parameter_count
            )
        )
    return trainable_parameter_count


def _normalize_compose_argv(argv):
    """Map the hyphenated Compose CLI contract (spec §12, e.g.
    ``--compose-mode``) onto the underscore flags that transformers 4.33's
    HfArgumentParser registers (``--compose_mode``). Only ``--compose-*``
    tokens are rewritten; every other argument passes through untouched."""
    return [
        "--" + token[2:].replace("-", "_") if token.startswith("--compose-") else token
        for token in argv
    ]


def train() -> None:
    sys.argv = _normalize_compose_argv(sys.argv)
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if model_args.vision_tower is None:
        raise ValueError("Compose training requires --vision_tower")

    mode = model_args.compose_mode
    if mode not in ("fixed", "cluster_expert"):
        raise ValueError("--compose-mode must be one of fixed, cluster_expert")
    if mode == "cluster_expert":
        if not model_args.compose_selection_manifest:
            raise ValueError(
                "cluster_expert mode requires --compose-selection-manifest"
            )
        if not model_args.compose_cluster_expert_ids.strip():
            raise ValueError(
                "cluster_expert mode requires --compose-cluster-expert-ids"
            )
    else:
        if not model_args.compose_expert_ids.strip():
            raise ValueError("fixed mode requires --compose-expert-ids")

    model, vision_tower = _build_model(model_args, training_args)
    injected, injection_summary, manager, pool = _inject_and_pool(model, model_args)

    if mode == "cluster_expert":
        _prepare_cluster_expert_backward()
        _load_old_checkpoint(pool, model_args, training_args)
        cluster_expert_ids = _csv_ints(model_args.compose_cluster_expert_ids)
        _register_new_experts(pool, cluster_expert_ids, model_args)
        pool.train_only(cluster_expert_ids)
        # Reentrant activation checkpointing (torch default) re-runs each
        # layer under torch.no_grad() in backward and only connects the graph
        # when at least one checkpoint input requires grad.  The whole base
        # is frozen (embeddings included), so re-enable grad on the
        # embeddings only: they stay out of the optimizer (LoRA-only update)
        # but give each checkpointed layer a grad-requiring input, the
        # standard HF LoRA + gradient-checkpointing arrangement.
        model.gradient_checkpointing_enable()
        model.model.embed_tokens.weight.requires_grad_(True)
    else:
        _load_old_checkpoint(pool, model_args, training_args)
        selected_experts, trainable_experts = _resolve_expert_roles(
            _csv_ints(model_args.compose_expert_ids),
            model_args.compose_trainable_expert_ids,
        )
        gates = _csv_floats(model_args.compose_gates)
        if len(selected_experts) not in (1, 2):
            raise ValueError("Compose fixed mode trains one or two experts")
        _register_new_experts(pool, selected_experts, model_args)
        pool.train_only(trainable_experts)
        manager.set_default_selection(
            selected_experts,
            gates,
            normalization=model_args.compose_gate_normalization,
        )

    trainable_parameter_count = _check_expected_parameters(model, model_args)
    if model_args.tune_mm_mlp_adapter:
        model.get_model().mm_projector.requires_grad_(True)
    if training_args.gradient_checkpointing and mode == "fixed":
        model.enable_input_require_grads()

    if training_args.local_rank in (-1, 0):
        print("Compose mode: {}".format(mode))
        print("Compose injection summary: {}".format(injection_summary))
        print("Compose expert count: {}".format(len(pool.expert_ids())))
        print("Trainable expert IDs: {}".format(sorted(pool.trainable_expert_ids)))
        print("Trainable parameter count: {}".format(trainable_parameter_count))

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

    if mode == "cluster_expert":
        selections = _load_selection_manifest(model_args.compose_selection_manifest)
        dataset = ComposeSelectionDataset(
            data_args.data_path, tokenizer, data_args, selections
        )
        if model_args.max_samples and len(dataset.records) > model_args.max_samples:
            kept_ids = {
                str(record.get("id", index))
                for index, record in enumerate(dataset.records[: model_args.max_samples])
            }
            dataset.records = dataset.records[: model_args.max_samples]
            dataset.selections = {
                sample_id: selection
                for sample_id, selection in dataset.selections.items()
                if sample_id in kept_ids
            }
        data_collator = ComposeSelectionCollator(tokenizer)
        data_module = {
            "train_dataset": dataset,
            "eval_dataset": None,
            "data_collator": data_collator,
        }
    else:
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
    if checkpoints:
        raise ValueError(
            "output_dir contains checkpoint-* entries; automatic resume is disabled "
            "because Compose checkpoints are adapter-only: {}".format(sorted(checkpoints))
        )
    trainer.train()
    trainer.save_state()
    model.config.use_cache = True
    if training_args.should_save:
        pool.sync_training_step(trainer.state.global_step)
        pool.train_only([])
        model.config.save_pretrained(training_args.output_dir)
        save_expert_checkpoint(pool, training_args.output_dir)
        if mode == "cluster_expert":
            # Standalone per-expert state dicts for the commit transaction.
            for expert_id in cluster_expert_ids:
                payload = {}
                for layer_name, layer in manager.layers.items():
                    expert = layer.experts[str(expert_id)]
                    payload["{}.lora_A.weight".format(layer_name)] = expert.lora_A.weight
                    payload["{}.lora_B.weight".format(layer_name)] = expert.lora_B.weight
                torch.save(
                    {"expert_id": expert_id, "state_dict": payload},
                    os.path.join(training_args.output_dir, "expert_{:04d}.pt".format(expert_id)),
                )
    if training_args.local_rank in (-1, 0):
        supervision_summary = data_module["data_collator"].supervision_summary()
        if training_args.dataloader_num_workers:
            supervision_summary["scope"] = "main-process-only"
            supervision_summary["note"] = (
                "worker collators enforce zero-supervision errors but do not share counters; "
                "use compose.train.audit_supervision for dataset-wide statistics"
            )
        print("Compose supervision summary: {}".format(supervision_summary))
        print("Trainable LoRA-B count: {}".format(trainer.trainable_lora_b_count))
        print("Finite-gradient LoRA-B count: {}".format(
            trainer.max_finite_gradient_lora_b_count
        ))
        print("Injected {} Compose layers; experts={}".format(len(injected), pool.expert_ids()))


if __name__ == "__main__":
    train()
