"""V6 candidate conditional-residual training (Stage E6/E12 runner entry).

Trains candidate LoRA experts under frozen old experts:

- the base model, vision encoder, multimodal projector and all committed
  old experts are frozen;
- each sample carries its answer-supervised old-teacher set plus its
  assigned candidate slot; the training set is
  ``old_teacher_set + selected_candidate`` via a per-sample unified
  ComposeSelection (E1 semantics, -1 padded);
- only the selected candidate receives gradient; unselected candidates
  stay at zero gradient (ExpertManager.train_only + sparse forward).

Task 1 cold start uses one candidate slot and backbone-only teacher sets.

Inputs: UCIT-format LLaVA JSON (``--data-path``) plus a selection manifest
(``--selection-manifest``) mapping sample ids to
``{"teacher_ids": [...], "slot": n}``. Outputs a compose expert checkpoint
per candidate slot.
"""

import argparse
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import transformers
from torch.utils.data import Dataset

from llava import conversation as conversation_lib

from compose.adapters import ExpertManager, inject_compose_adapters, validate_compose_injection
from compose.adapters.runtime import use_selection
from compose.adapters.types import ComposeSelection, PAD_EXPERT_ID
from compose.config import ComposeAdapterConfig
from compose.experts import ExpertPool, load_expert_checkpoint, save_expert_checkpoint
from compose.model import ComposeLlavaForCausalLM, load_compose_config
from compose.train.arguments import DataArguments, ModelArguments, TrainingArguments
from compose.train.data import DataCollatorForSupervisedDataset, LazySupervisedDataset
from compose.train.trainer import ComposeTrainer


class SelectionTaggedDataset(LazySupervisedDataset):
    """LazySupervisedDataset with per-sample (teacher_ids, slot) routing."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        data_args: DataArguments,
        selections: Dict[str, Dict[str, Any]],
    ) -> None:
        super().__init__(data_path, tokenizer, data_args)
        self.selections = selections

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(index)
        sample_id = str(self.records[index].get("id", index))
        selection = self.selections.get(sample_id)
        if selection is None:
            raise KeyError(
                "sample {} missing from the selection manifest".format(sample_id)
            )
        item["sample_id"] = sample_id
        item["teacher_ids"] = list(selection.get("teacher_ids", []))
        item["slot"] = int(selection["slot"])
        return item


class SelectionCollator:
    """Formal padding plus per-sample unified selections."""

    def __init__(self, tokenizer) -> None:
        self._shim = DataCollatorForSupervisedDataset(tokenizer)

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sample_ids = [str(instance["sample_id"]) for instance in instances]
        selections = []
        for instance in instances:
            teacher_ids = list(instance["teacher_ids"])
            slot = int(instance["slot"])
            ids = teacher_ids + [slot]
            gates = [1.0] * len(ids)
            while len(ids) < 2:
                ids.append(PAD_EXPERT_ID)
                gates.append(0.0)
            selections.append((ids, gates))
        stripped = [
            {k: v for k, v in instance.items()
             if k not in ("sample_id", "teacher_ids", "slot")}
            for instance in instances
        ]
        batch = self._shim(stripped)
        batch["sample_ids"] = sample_ids
        batch["v6_selections"] = selections
        return batch


class V6CandidateTrainer(ComposeTrainer):
    """ComposeTrainer that applies per-sample selections during training."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _apply_selections(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raw = inputs.pop("v6_selections")
        expert_ids = torch.tensor(
            [[ids[0], ids[1]] for ids, _ in raw], dtype=torch.long
        )
        gates = torch.tensor(
            [[gates[0], gates[1]] for _, gates in raw], dtype=torch.float32
        )
        inputs["compose_selection"] = ComposeSelection(expert_ids, gates)
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = self._apply_selections(dict(inputs))
        return super().compute_loss(model, inputs, return_outputs=return_outputs)


def _load_selection_manifest(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    manifest = {}
    for row in rows:
        manifest[str(row["sample_id"])] = row
    return manifest


def _max_length(records) -> int:
    lengths = []
    for sample in records:
        length = sum(len(message["value"].split()) for message in sample["conversations"])
        lengths.append(length + (128 if "image" in sample else 0))
    return max(lengths) if lengths else 128


def train() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--projector-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--candidate-ids", required=True)
    parser.add_argument("--old-expert-checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--global-batch-size", type=int, default=24)
    parser.add_argument("--per-device-batch-size", type=int, default=6)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--deepspeed", default=None)
    # dataloader 预取并行度。正式全量任务（23998 样本）在 num_workers=0 时
    # 每次 __getitem__ 串行重载图像+预处理，成为训练瓶颈（~13s/step）；
    # 多 worker 预取可恢复 ~3.3s/step 的纯计算速度。
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.local_rank is None:
        args.local_rank = int(
            os.environ.get("LOCAL_RANK", os.environ.get("RANK", "-1"))
        )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    transformers.set_seed(args.seed)

    candidate_ids = [int(value) for value in args.candidate_ids.split(",") if value]
    if not candidate_ids:
        raise ValueError("at least one candidate id is required")

    config = load_compose_config(args.model_path)
    config.attn_implementation = "flash_attention_2"
    config.mm_vision_tower = args.vision_tower
    config.mm_vision_select_layer = -2
    config.mm_vision_select_feature = "patch"
    config.mm_projector_type = "mlp2x_gelu"
    model = ComposeLlavaForCausalLM.from_pretrained(
        args.model_path, config=config, torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False
    model.get_model().initialize_vision_modules(
        SimpleNamespace(
            vision_tower=args.vision_tower,
            mm_vision_select_layer=-2,
            mm_vision_select_feature="patch",
            mm_projector_type="mlp2x_gelu",
            pretrain_mm_mlp_adapter=args.projector_path,
        )
    )
    vision_tower = model.get_vision_tower()
    vision_tower.to(device="cuda", dtype=torch.bfloat16)

    adapter_config = ComposeAdapterConfig(
        rank=args.rank, alpha=args.alpha, dropout=0.0
    )
    injected = inject_compose_adapters(model, adapter_config)
    validate_compose_injection(model, injected)
    manager = ExpertManager(model)
    pool = ExpertPool(manager)
    if args.old_expert_checkpoint:
        load_summary = load_expert_checkpoint(pool, args.old_expert_checkpoint)[
            "load_summary"
        ]
        if args.local_rank in (-1, 0):
            print("old expert checkpoint load: {}".format(load_summary))
    for candidate_id in candidate_ids:
        pool.register(candidate_id, name="candidate-{}".format(candidate_id))
    pool.train_only(candidate_ids)
    # Reentrant activation checkpointing (torch default) re-runs each layer
    # under torch.no_grad() in backward and only connects the graph when at
    # least one checkpoint input requires grad.  train_only froze the whole
    # base (embeddings included), which would leave every layer's checkpoint
    # detached ("element 0 of tensors does not require grad").  Re-enable
    # grad on the embeddings only: they stay out of the optimizer (LoRA-only
    # update) but give each checkpointed layer a grad-requiring input, the
    # standard HF LoRA + gradient-checkpointing arrangement.
    model.gradient_checkpointing_enable()
    model.model.embed_tokens.weight.requires_grad_(True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path, model_max_length=2048, padding_side="right", use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

    selections = _load_selection_manifest(args.selection_manifest)
    data_args = DataArguments(
        data_path=args.data_path,
        lazy_preprocess=False,
        is_multimodal=True,
        image_folder=args.image_folder,
        image_aspect_ratio="pad",
    )
    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True
    data_args.mm_use_im_start_end = False
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    dataset = SelectionTaggedDataset(
        args.data_path, tokenizer, data_args, selections,
    )
    if args.max_samples and len(dataset.records) > args.max_samples:
        kept_ids = {
            str(record.get("id", index))
            for index, record in enumerate(dataset.records[: args.max_samples])
        }
        dataset.records = dataset.records[: args.max_samples]
        dataset.selections = {
            sample_id: selection
            for sample_id, selection in dataset.selections.items()
            if sample_id in kept_ids
        }
    data_collator = SelectionCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        logging_steps=20,
        save_steps=100000,
        save_total_limit=1,
        seed=args.seed,
        data_seed=args.seed,
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        dataloader_num_workers=args.dataloader_num_workers,
        model_max_length=2048,
    )
    trainer = V6CandidateTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=dataset,
        expert_pool=pool,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    if args.local_rank in (-1, 0):
        save_expert_checkpoint(pool, args.output_dir)
        # Per-candidate standalone state dicts for the commit transaction.
        for candidate_id in candidate_ids:
            payload = {}
            for layer_name, layer in manager.layers.items():
                expert = layer.experts[str(candidate_id)]
                payload["{}.lora_A.weight".format(layer_name)] = expert.lora_A.weight
                payload["{}.lora_B.weight".format(layer_name)] = expert.lora_B.weight
            torch.save(
                {"candidate_id": candidate_id, "state_dict": payload},
                os.path.join(args.output_dir, "candidate_{}.pt".format(candidate_id)),
            )
        print("V6 candidate training done: {}".format(args.output_dir))


if __name__ == "__main__":
    train()
