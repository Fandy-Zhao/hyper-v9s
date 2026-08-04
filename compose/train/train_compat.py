"""Stage 04: minimal compatibility-aware training variants (B2-B5).

Trains a frozen Expert A with a new B expert under several regimes, all
sharing the controlled recipe (AdamW lr 2e-4, cosine, warmup 0.03, bf16,
gradient checkpointing, per-device batch 8, deepspeed zero2, one epoch of
the combined data):

  B2  residual + standalone preservation:
        L = L_AB(A+B) + lambda_single * L_B(B) + lambda_anchor * L_preserve
      L_preserve = MSE between the current B-only answer-position logits and
      the frozen anchor (independent_b checkpoint) logits, on B_only samples.
  B3  alternating conditional training: B_only and A_plus_B samples are
      interleaved 1:1 in the same batches; per-sample expert routing (B-only
      samples activate expert 1 alone; A_plus_B samples activate 0+1).
  B4  layer-structured residual: expert 1 is trainable only inside the
      configured module/layer scope (--b4-modules, --b4-layer-range).
  B5  joint dual rank-8 diagnostic upper bound:
        L = L_A(A) + L_B(B) + lambda_pair * L_AB(A+B)
      both experts trainable; diagnostic only (not a formal method).

Per-sample expert routing uses the existing ComposeSelection mechanism with
a zero-weight null expert (id 100) filling the unused top-2 slot, so one
batch can mix single-expert and dual-expert samples.

B_plus_C data NEVER enters this script; the unseen B+C test is performed
only in evaluation.

Usage (deepspeed, mirrors run_train_one.sh):
  deepspeed --include localhost:4,5,6,7 --master_port 29800 --module compose.train.train_compat \
    --config <run>.json
"""

import argparse
import json
import os
import random
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import transformers

from llava import conversation as conversation_lib

from compose.adapters import ExpertManager, inject_compose_adapters, validate_compose_injection
from compose.adapters.runtime import use_selection
from compose.adapters.types import ComposeSelection
from compose.config import ComposeAdapterConfig
from compose.experts import ExpertPool, load_expert_checkpoint, save_expert_checkpoint
from compose.model import ComposeLlavaForCausalLM, load_compose_config
from compose.train.arguments import DataArguments, TrainingArguments
from compose.train.data import DataCollatorForSupervisedDataset, LazySupervisedDataset
from compose.train.trainer import ComposeTrainer

NULL_EXPERT_ID = 100  # zero-weight filler for the unused top-2 slot

MODES = ("b2", "b3", "b4", "b5")

TAG_SLOTS = {
    "A_only": ([0, NULL_EXPERT_ID], [1.0, 0.0]),
    "B_only": ([1, NULL_EXPERT_ID], [1.0, 0.0]),
    "A_plus_B": ([0, 1], [1.0, 1.0]),
}


@dataclass
class CompatConfig:
    mode: str
    data_paths: Dict[str, str]                 # tag -> dataset json
    lambda_single: float = 0.5                 # B2 L_B weight
    lambda_anchor: float = 0.1                 # B2 L_preserve weight
    lambda_pair: float = 1.0                   # B5 L_AB weight
    anchor_checkpoint: Optional[str] = None    # B2 independent_b checkpoint
    anchor_logits_path: Optional[str] = None   # B2 precomputed anchor logits
    b4_modules: str = ""                       # B4 comma-separated module suffixes
    b4_layer_range: str = ""                   # B4 "lo,hi" inclusive layer range
    seed: int = 42

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError("unknown compat mode: {}".format(self.mode))
        if not self.data_paths:
            raise ValueError("at least one dataset required")
        if self.mode in ("b2", "b3") and "A_plus_B" not in self.data_paths:
            raise ValueError("{} requires an A_plus_B dataset".format(self.mode))
        if self.mode in ("b2", "b3", "b5") and "B_only" not in self.data_paths:
            raise ValueError("{} requires a B_only dataset".format(self.mode))
        if self.mode == "b5" and "A_only" not in self.data_paths:
            raise ValueError("b5 requires an A_only dataset")
        if self.mode == "b2" and not self.anchor_logits_path:
            raise ValueError("b2 requires anchor_logits_path (precomputed B-only anchor logits)")
        if self.mode == "b4" and not (self.b4_modules or self.b4_layer_range):
            raise ValueError("b4 requires b4_modules and/or b4_layer_range")


def load_config(path: str) -> CompatConfig:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return CompatConfig(**payload)


class TaggedSupervisedDataset(LazySupervisedDataset):
    """LazySupervisedDataset with a per-sample routing tag."""

    def __init__(self, data_path: str, tokenizer, data_args: DataArguments,
                 tag: str) -> None:
        super().__init__(data_path, tokenizer, data_args)
        self.tag = tag

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(index)
        item["compose_tag"] = self.tag
        return item


class CombinedDataset(torch.utils.data.Dataset):
    """1:1 interleaved combination of tagged datasets with per-tag shuffling."""

    def __init__(self, datasets: Sequence[TaggedSupervisedDataset], seed: int) -> None:
        if not datasets:
            raise ValueError("no datasets")
        self.datasets = list(datasets)
        self.order = [index % len(self.datasets) for index in range(len(self.datasets) * max(len(d) for d in self.datasets))]
        rng = random.Random(seed)
        self.per_dataset = []
        for dataset in self.datasets:
            indices = list(range(len(dataset)))
            rng.shuffle(indices)
            self.per_dataset.append(indices)

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        dataset_index = self.order[index]
        local = self.per_dataset[dataset_index][
            (index // len(self.datasets)) % len(self.per_dataset[dataset_index])
        ]
        return self.datasets[dataset_index][local]

    @property
    def lengths(self) -> List[int]:
        return [
            sum(len(message["value"].split()) for message in sample["conversations"])
            + (128 if "image" in sample else 0)
            for sample in self._iter_samples()
        ]

    @property
    def modality_lengths(self) -> List[int]:
        lengths = []
        for dataset in self.datasets:
            for sample in dataset.records:
                length = sum(len(message["value"].split()) for message in sample["conversations"])
                lengths.append(length if "image" in sample else -length)
        return lengths

    def _iter_samples(self):
        for index in range(len(self)):
            dataset_index = self.order[index]
            local = self.per_dataset[dataset_index][
                (index // len(self.datasets)) % len(self.per_dataset[dataset_index])
            ]
            yield self.datasets[dataset_index].records[local]


class CompatCollator:
    """Formal padding + per-sample tags/sample ids."""

    def __init__(self, tokenizer) -> None:
        self._shim = DataCollatorForSupervisedDataset(tokenizer)

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        tags = [str(instance["compose_tag"]) for instance in instances]
        sample_ids = [str(instance["sample_id"]) for instance in instances]
        stripped = [
            {k: v for k, v in instance.items() if k not in ("compose_tag", "sample_id")}
            for instance in instances
        ]
        batch = self._shim(stripped)
        batch["compose_tags"] = tags
        batch["sample_ids"] = sample_ids
        return batch


class CompatTrainer(ComposeTrainer):
    def __init__(self, *args, compat: CompatConfig,
                 anchor_logits: Optional[Dict[str, torch.Tensor]] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.compat = compat
        self.anchor_logits = anchor_logits

    def _make_selection(self, tags: Sequence[str], device: torch.device) -> ComposeSelection:
        ids, gates = [], []
        for tag in tags:
            if tag not in TAG_SLOTS:
                raise ValueError("unknown compose tag: {}".format(tag))
            ids.append(TAG_SLOTS[tag][0])
            gates.append(TAG_SLOTS[tag][1])
        return ComposeSelection(
            torch.tensor(ids, dtype=torch.long, device=device),
            torch.tensor(gates, dtype=torch.float32, device=device),
            normalization="none",
        )

    def _sample_weight(self, tag: str) -> float:
        if self.compat.mode in ("b2", "b3"):
            return 1.0 if tag == "A_plus_B" else float(self.compat.lambda_single)
        if self.compat.mode == "b5":
            return float(self.compat.lambda_pair) if tag == "A_plus_B" else 1.0
        return 1.0

    def _weighted_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                       tags: Sequence[str]) -> torch.Tensor:
        vocab = logits.shape[-1]
        shifted = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        mask = shifted_labels.ne(-100)
        per_token = torch.nn.functional.cross_entropy(
            shifted.reshape(-1, vocab), shifted_labels.reshape(-1), reduction="none"
        ).view(shifted_labels.shape)
        per_sample = (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        weights = torch.tensor([self._sample_weight(tag) for tag in tags],
                               dtype=per_sample.dtype, device=per_sample.device)
        return (per_sample * weights).mean()

    def _anchor_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                     tags: Sequence[str], sample_ids: Sequence[str]) -> torch.Tensor:
        if self.compat.mode != "b2" or self.anchor_logits is None:
            return torch.zeros((), dtype=logits.dtype, device=logits.device)
        total = torch.zeros((), dtype=logits.dtype, device=logits.device)
        count = 0
        for index, (tag, sid) in enumerate(zip(tags, sample_ids)):
            if tag != "B_only" or sid not in self.anchor_logits:
                continue
            positions = [i for i, value in enumerate(labels[index].tolist()) if value != -100]
            if not positions:
                continue
            answer_pos = positions[0] - 1
            if answer_pos < 0:
                continue
            current = logits[index, answer_pos].float()
            anchor = self.anchor_logits[sid].to(device=current.device, dtype=current.dtype)
            if anchor.shape != current.shape:
                continue
            total = total + torch.nn.functional.mse_loss(current, anchor)
            count += 1
        return (total / count) if count else torch.zeros((), dtype=logits.dtype, device=logits.device)

    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        tags = list(inputs.pop("compose_tags", ["A_plus_B"] * inputs["input_ids"].shape[0]))
        sample_ids = list(inputs.pop("sample_ids", []))
        selection = self._make_selection(tags, inputs["input_ids"].device)
        with use_selection(selection):
            outputs = model(**inputs)
        loss = self._weighted_loss(outputs.logits, inputs["labels"], tags)
        anchor = self._anchor_loss(outputs.logits, inputs["labels"], tags, sample_ids)
        total = loss + float(self.compat.lambda_anchor) * anchor
        if self.do_grad_scaling:
            self.scaler.scale(total).backward()
        elif self.use_apex:
            raise NotImplementedError("apex is not supported by train_compat")
        else:
            self.accelerator.backward(total)
        return total.detach() / self.args.gradient_accumulation_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON CompatConfig")
    parser.add_argument("--model-name-or-path", default="/data/ckpt/zhaozhuofan/models/llava-v1.5-7b")
    parser.add_argument("--vision-tower", default="/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336")
    parser.add_argument("--pretrain-mm-mlp-adapter", default="")
    parser.add_argument("--image-folder", default="experiments/data/controlled_format_v1")
    parser.add_argument("--seed-checkpoint", required=True, help="seed B checkpoint (independent_b)")
    parser.add_argument("--expert-a-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    compat = load_config(args.config)
    compat.validate()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1

    config = load_compose_config(args.model_name_or_path)
    config.mm_vision_tower = args.vision_tower
    config.mm_vision_select_layer = -2
    config.mm_vision_select_feature = "patch"
    config.mm_projector_type = "mlp2x_gelu"
    model = ComposeLlavaForCausalLM.from_pretrained(
        args.model_name_or_path, config=config, torch_dtype=torch.bfloat16
    )
    model.get_model().initialize_vision_modules(
        argparse.Namespace(
            vision_tower=args.vision_tower, mm_vision_select_layer=-2,
            mm_vision_select_feature="patch", mm_projector_type="mlp2x_gelu",
            pretrain_mm_mlp_adapter=args.pretrain_mm_mlp_adapter,
        )
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path, model_max_length=args.max_length,
        padding_side="right", use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    model.config.image_aspect_ratio = "pad"
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_use_im_start_end = False
    model.config.mm_use_im_patch_token = False
    model.initialize_vision_tokenizer(
        argparse.Namespace(mm_use_im_start_end=False, mm_use_im_patch_token=False), tokenizer
    )
    adapter_config = ComposeAdapterConfig(rank=args.rank, alpha=args.alpha, dropout=0.0)
    inject_compose_adapters(model, adapter_config)
    validate_compose_injection(model)
    manager = ExpertManager(model)
    pool = ExpertPool(manager)
    load_expert_checkpoint(pool, args.expert_a_checkpoint)   # expert 0 (frozen A)
    load_expert_checkpoint(pool, args.seed_checkpoint)       # expert 1 (seed B)
    for layer in manager.layers.values():
        layer.add_expert(NULL_EXPERT_ID)  # zero-weight filler for the top-2 slot

    trainable = (0, 1) if compat.mode == "b5" else (1,)
    pool.train_only(trainable)
    if compat.mode == "b4":
        selected_modules = {m.strip() for m in compat.b4_modules.split(",") if m.strip()}
        lo, hi = 0, 31
        if compat.b4_layer_range:
            parts = compat.b4_layer_range.split(",")
            lo, hi = int(parts[0]), int(parts[1])
        for name, parameter in model.named_parameters():
            if ".experts.1." not in name:
                continue
            parts = name.split(".")
            layer_index = int(parts[2])
            module_name = parts[3]
            keep = (not selected_modules or module_name in selected_modules) and (lo <= layer_index <= hi)
            if not keep:
                parameter.requires_grad_(False)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank in (-1, 0):
        print("trainable parameters: {}".format(trainable_count))

    anchor_logits = None
    if compat.mode == "b2" and compat.anchor_logits_path:
        anchor_logits = torch.load(compat.anchor_logits_path, map_location="cpu")

    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    data_args = DataArguments(
        data_path=compat.data_paths.get("A_plus_B", ""), is_multimodal=True,
        image_folder=args.image_folder, image_aspect_ratio="pad",
    )
    datasets = [TaggedSupervisedDataset(path, tokenizer, data_args, tag)
                for tag, path in compat.data_paths.items()]
    combined = CombinedDataset(datasets, seed=compat.seed)
    training_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio, lr_scheduler_type="cosine",
        logging_steps=1, save_strategy="epoch", save_total_limit=2,
        bf16=True, tf32=True, gradient_checkpointing=True,
        dataloader_num_workers=args.num_workers, report_to="none", seed=compat.seed,
        model_max_length=args.max_length, remove_unused_columns=False,
        optim="adamw_torch", weight_decay=0.0,
    )
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    trainer = CompatTrainer(
        model=model, args=training_args, tokenizer=tokenizer,
        train_dataset=combined, data_collator=CompatCollator(tokenizer),
        expert_pool=pool, compat=compat, anchor_logits=anchor_logits,
    )
    trainer.train()
    if rank in (-1, 0):
        save_expert_checkpoint(pool, args.output_dir, [0, 1])
        manifest = {
            "mode": compat.mode, "config": asdict(compat),
            "seed": compat.seed, "output_dir": args.output_dir,
            "trainable_parameters": trainable_count,
            "trainable_expert_ids": list(trainable),
            "null_expert_id": NULL_EXPERT_ID,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        }
        with open(os.path.join(args.output_dir, "compat_manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
