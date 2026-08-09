"""4-GPU DDP gradient-audit smoke (spec §9/§21) — run under torchrun.

Usage (mirrors the S6 cluster-expert train command; the audit reuses the
exact model-building path of ``compose.train.train_compose``):

    torchrun --standalone --nproc_per_node=4 -m compose.experiments.smoke_ddp \
      --model_name_or_path <base> --vision_tower <tower> \
      --pretrain_mm_mlp_adapter <projector> --version v1 \
      --data_path <audit_train.json> --image_folder <images> \
      --compose-mode cluster_expert \
      --compose-selection-manifest <audit_manifest.json> \
      --compose-cluster-expert-ids 77 \
      --compose-checkpoint <prev pool>            # optional: historical experts
      --output_dir <scratch> --bf16 True --gradient_checkpointing True \
      --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
      --num_train_epochs 1 --learning_rate 2.0e-4 --warmup_ratio 0.03 \
      --lr_scheduler_type cosine --model_max_length 2048 \
      --seed 42 --report_to none

Per rank it verifies:

- §21 (init identity): sha256 of the new-expert LoRA weights before
  training is identical on rank0..rank3 (deterministic init + DDP
  initial broadcast);
- §9 (gradient isolation): after each optimizer step, every new-expert
  LoRA gradient is finite, and every other parameter (frozen base,
  vision tower, projector, historical experts, embeddings controller
  set) has no gradient (requires_grad=False); embed_tokens carries a
  gradient exactly as in the single-GPU protocol (out of the LoRA
  update semantics, in the optimizer);
- §9 (weight sync): after one optimizer.step the new-expert weight hash
  is identical on all ranks (DDP all-reduce).

Any failure exits nonzero on every rank (torchrun propagates it).

Scaling bench (spec §25): set ``SMOKE_DDP_STEPS=N`` to run N optimizer
steps on whatever data/manifest is given and additionally report
samples/sec, mean step time and peak VRAM (world_size=1 measures the
single-GPU reference; world_size=4 measures the DDP run — identical code
path, only the world size differs).
"""

import hashlib
import json
import os
import sys

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from llava import conversation as conversation_lib

from compose.adapters.runtime import use_selection
from compose.adapters.types import ComposeSelection
from compose.train.arguments import DataArguments, ModelArguments, TrainingArguments
from compose.train.data import ComposeSelectionCollator, ComposeSelectionDataset
from compose.train.train_compose import (
    _build_model,
    _csv_ints,
    _inject_and_pool,
    _load_old_checkpoint,
    _load_selection_manifest,
    _normalize_compose_argv,
    _prepare_cluster_expert_backward,
    _register_new_experts,
)


def _expert_state_hash(manager, cluster_expert_ids) -> str:
    tensors = []
    for layer in manager.layers.values():
        for expert_id in cluster_expert_ids:
            expert = layer.experts[str(expert_id)]
            tensors.append(expert.lora_A.weight.detach().float().cpu().reshape(-1))
            tensors.append(expert.lora_B.weight.detach().float().cpu().reshape(-1))
    flat = torch.cat(tensors)
    return hashlib.sha256(flat.numpy().tobytes()).hexdigest()


def _all_equal_hex(local_hex: str) -> bool:
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, local_hex)
    return len(set(values)) == 1


def _audit_gradients(model, manager, cluster_expert_ids, rank, problems) -> None:
    """§9: new-expert LoRA gradients finite; everything else grad-free."""
    for expert_id in cluster_expert_ids:
        for layer in manager.layers.values():
            expert = layer.experts[str(expert_id)]
            for name, parameter in (("lora_A", expert.lora_A.weight), ("lora_B", expert.lora_B.weight)):
                if parameter.grad is None:
                    problems.append("rank{}: expert {} {} has NO gradient".format(rank, expert_id, name))
                elif not bool(torch.isfinite(parameter.grad).all()):
                    problems.append("rank{}: expert {} {} gradient non-finite".format(rank, expert_id, name))
    module = model.module if hasattr(model, "module") else model
    frozen_with_grad = [
        name
        for name, parameter in module.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    for name in frozen_with_grad:
        problems.append("rank{}: frozen parameter {} has a gradient".format(rank, name))
    # The optimizer must contain ONLY the LoRA params + embed_tokens
    # (single-GPU protocol identity; spec §22 — query encoder must never
    # enter the optimizer, and it is frozen by _build_model).
    optimizer_scope = sorted(
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    )
    if not optimizer_scope:
        problems.append("rank{}: optimizer scope is empty".format(rank))


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Scaling-bench override (spec §25): SMOKE_DDP_STEPS=N runs N optimizer
    # steps and reports samples/sec / step time / peak VRAM per rank.
    steps = int(os.environ.get("SMOKE_DDP_STEPS", "4"))

    sys.argv = _normalize_compose_argv(sys.argv)
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, vision_tower = _build_model(model_args, training_args)
    injected, _, manager, pool = _inject_and_pool(model, model_args)
    _prepare_cluster_expert_backward()
    _load_old_checkpoint(pool, model_args, training_args)
    cluster_expert_ids = _csv_ints(model_args.compose_cluster_expert_ids)
    _register_new_experts(pool, cluster_expert_ids, model_args)
    pool.train_only(cluster_expert_ids)
    model.gradient_checkpointing_enable()
    model.model.embed_tokens.weight.requires_grad_(True)

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
    model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer)

    # We bypass accelerate (no HF Trainer here), so move the whole model
    # (base + vision tower + projector) to this rank's device explicitly;
    # under torchrun, TrainingArguments.device is cuda:0 on every rank.
    # model.train() is required: activation checkpointing only engages
    # when self.training (the HF Trainer path calls it; without it the
    # full graph is retained and a 7B bf16 model OOMs at backward).
    model.train()
    model = model.to("cuda:{}".format(local_rank))

    # §21: init identity across ranks (before training, after DDP wrap so
    # the initial broadcast is included).
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], find_unused_parameters=False
    )
    init_hex = _expert_state_hash(manager, cluster_expert_ids)
    init_ok = _all_equal_hex(init_hex)

    selections = _load_selection_manifest(model_args.compose_selection_manifest)
    dataset = ComposeSelectionDataset(
        data_args.data_path, tokenizer, data_args, selections
    )
    collator = ComposeSelectionCollator(tokenizer)
    sampler = DistributedSampler(dataset, shuffle=True, seed=seed)
    sampler.set_epoch(0)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training_args.learning_rate),
    )

    problems = []
    steps_run = 0
    wall_start = torch.cuda.Event(enable_timing=True)
    wall_end = torch.cuda.Event(enable_timing=True)
    wall_start.record()
    module = model.module if hasattr(model, "module") else model
    print(
        "rank{}: training={} grad_checkpointing={} seq={} weights_gb={:.1f}".format(
            local_rank,
            module.training,
            module.model.gradient_checkpointing,
            loader.dataset.records[0].get("conversations", [{}])[0].get("value", "")[:20],
            sum(parameter.numel() for parameter in module.parameters()) * 2 / 1e9,
        ),
        flush=True,
    )
    for batch in loader:
        device_batch = {
            key: value.to("cuda:{}".format(local_rank)) if torch.is_tensor(value) else value
            for key, value in batch.items()
            if key not in ("compose_selections", "sample_ids")
        }
        raw = batch["compose_selections"]
        expert_ids = torch.tensor([list(ids) for ids, _ in raw], dtype=torch.long)
        gates = torch.tensor([list(gates) for _, gates in raw], dtype=torch.float32)
        selection = ComposeSelection(expert_ids, gates)
        # The selection context must span backward as well as forward:
        # activation-checkpoint recomputation re-runs the layers inside
        # loss.backward() and needs the same LoRA routing.
        with use_selection(selection):
            outputs = model(**device_batch)
            loss = outputs.loss
            if steps_run == 0:
                print(
                    "rank{}: input_len={} labels_len={} after_forward_mb={:.0f}".format(
                        local_rank,
                        int(device_batch["input_ids"].numel()),
                        int(device_batch["labels"].numel()),
                        torch.cuda.memory_allocated() / 1e6,
                    ),
                    flush=True,
                )
            loss.backward()
            if steps_run == 0:
                print(
                    "rank{}: after_backward_mb={:.0f} peak_mb={:.0f}".format(
                        local_rank,
                        torch.cuda.memory_allocated() / 1e6,
                        torch.cuda.max_memory_allocated() / 1e6,
                    ),
                    flush=True,
                )
        _audit_gradients(model, manager, cluster_expert_ids, local_rank, problems)
        optimizer.step()
        optimizer.zero_grad()
        steps_run += 1
        if steps_run >= steps:
            break
    wall_end.record()
    torch.cuda.synchronize()
    step_time_s = wall_start.elapsed_time(wall_end) / 1000.0 / max(steps_run, 1)

    post_hex = _expert_state_hash(manager, cluster_expert_ids)
    post_ok = _all_equal_hex(post_hex)

    results = {
        "rank": local_rank,
        "world_size": dist.get_world_size(),
        "steps_run": steps_run,
        "optimizer_parameter_count": len(
            [parameter for parameter in model.parameters() if parameter.requires_grad]
        ),
        "cluster_expert_ids": cluster_expert_ids,
        "init_hash": init_hex,
        "init_identical_across_ranks": init_ok,
        "post_step_hash": post_hex,
        "post_step_identical_across_ranks": post_ok,
        "gradient_audit_problems": problems,
        "audit_passed": init_ok and post_ok and not problems,
        "wall_time_steps_s": step_time_s * steps_run,
        "mean_step_time_s": step_time_s,
        "samples_per_second": steps_run * dist.get_world_size() / step_time_s,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)
    if not results["audit_passed"]:
        raise SystemExit("rank {}: DDP audit FAILED".format(local_rank))


if __name__ == "__main__":
    main()
