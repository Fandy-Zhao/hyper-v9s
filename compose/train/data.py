import copy
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import torch
import transformers
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from llava import conversation as conversation_lib
from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IGNORE_INDEX,
)
from llava.mm_utils import tokenizer_image_token

from compose.adapters.types import MAX_ACTIVE_EXPERTS, pad_selection

from .arguments import DataArguments


ImageFile.LOAD_TRUNCATED_IMAGES = True
_KNOWN_MISSING_IMAGES = {
    "OCR-VQA/images/1421539896.jpg",
    "OCR-VQA/images/141393394.jpg",
    "OCR-VQA/images/316881791.jpg",
    "OCR-VQA/images/140445692.jpg",
    "OCR-VQA/images/142153990X.jpg",
    "OCR-VQA/images/689852649.jpg",
}


def preprocess_multimodal(sources: Sequence, data_args: DataArguments):
    if not data_args.is_multimodal:
        return sources
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                value = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = (DEFAULT_IMAGE_TOKEN + "\n" + value).strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN,
                        "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>",
                    )
            replacement = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replacement = DEFAULT_IM_START_TOKEN + replacement + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replacement
            )
    return sources


def preprocess_v1(sources, tokenizer, has_image: bool = False) -> Dict[str, torch.Tensor]:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []
    for source_index, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for message_index, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if role != conv.roles[message_index % 2]:
                raise ValueError("conversation roles are not alternating at {}".format(source_index))
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    targets = input_ids.clone()
    separator = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_length = int(target.ne(tokenizer.pad_token_id).sum())
        current_length = 1
        target[:current_length] = IGNORE_INDEX
        for round_text in conversation.split(conv.sep2):
            if not round_text:
                break
            parts = round_text.split(separator)
            if len(parts) != 2:
                break
            instruction = parts[0] + separator
            if has_image:
                round_length = len(tokenizer_image_token(round_text, tokenizer))
                instruction_length = len(tokenizer_image_token(instruction, tokenizer)) - 2
            else:
                round_length = len(tokenizer(round_text).input_ids)
                instruction_length = len(tokenizer(instruction).input_ids) - 2
            target[current_length : current_length + instruction_length] = IGNORE_INDEX
            current_length += round_length
        target[current_length:] = IGNORE_INDEX
        if current_length < tokenizer.model_max_length and current_length != total_length:
            target[:] = IGNORE_INDEX
    return {"input_ids": input_ids, "labels": targets}


def preprocess(sources, tokenizer, has_image: bool = False) -> Dict[str, torch.Tensor]:
    if not conversation_lib.default_conversation.version.startswith("v1"):
        raise ValueError("Compose foundation currently supports the LLaVA v1 template")
    return preprocess_v1(sources, tokenizer, has_image=has_image)


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, data_args: DataArguments) -> None:
        super().__init__()
        with open(data_path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        self.records = [
            record
            for record in records
            if "image" not in record or record["image"] not in _KNOWN_MISSING_IMAGES
        ]
        if data_args.memory_data_path:
            with open(data_args.memory_data_path, "r", encoding="utf-8") as handle:
                self.records.extend(json.load(handle))
            random.shuffle(self.records)
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self) -> int:
        return len(self.records)

    @property
    def lengths(self):
        return [
            sum(len(message["value"].split()) for message in sample["conversations"])
            + (128 if "image" in sample else 0)
            for sample in self.records
        ]

    @property
    def modality_lengths(self):
        lengths = []
        for sample in self.records:
            length = sum(
                len(message["value"].split()) for message in sample["conversations"]
            )
            lengths.append(length if "image" in sample else -length)
        return lengths

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.records[index]
        sources = [copy.deepcopy(sample["conversations"])]
        if "image" in sample:
            image_path = os.path.join(self.data_args.image_folder, sample["image"])
            image = Image.open(image_path).convert("RGB")
            processor = self.data_args.image_processor
            if self.data_args.image_aspect_ratio == "pad":
                width, height = image.size
                size = max(width, height)
                square = Image.new(
                    image.mode,
                    (size, size),
                    tuple(int(value * 255) for value in processor.image_mean),
                )
                square.paste(image, ((size - width) // 2, (size - height) // 2))
                image = square
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            sources = preprocess_multimodal(sources, self.data_args)
        data = preprocess(sources, self.tokenizer, has_image="image" in sample)
        item = {
            "input_ids": data["input_ids"][0],
            "labels": data["labels"][0],
            "sample_id": str(sample.get("id", index)),
        }
        if "image" in sample:
            item["image"] = image
        elif self.data_args.is_multimodal:
            crop = self.data_args.image_processor.crop_size
            item["image"] = torch.zeros(3, crop["height"], crop["width"])
        return item


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer
    sample_count: int = field(default=0, init=False)
    supervised_token_total: int = field(default=0, init=False)
    supervised_token_min: Optional[int] = field(default=None, init=False)
    supervised_token_max: int = field(default=0, init=False)
    zero_supervision_count: int = field(default=0, init=False)

    def supervision_summary(self) -> Dict[str, object]:
        mean = (
            float(self.supervised_token_total) / self.sample_count
            if self.sample_count
            else 0.0
        )
        return {
            "samples": self.sample_count,
            "min": self.supervised_token_min if self.supervised_token_min is not None else 0,
            "mean": mean,
            "max": self.supervised_token_max,
            "zero_supervision": self.zero_supervision_count,
        }

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )[:, : self.tokenizer.model_max_length]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )[:, : self.tokenizer.model_max_length]
        supervised_counts = labels.ne(IGNORE_INDEX).sum(dim=1)
        count_values = [int(value) for value in supervised_counts.tolist()]
        self.sample_count += len(count_values)
        self.supervised_token_total += sum(count_values)
        self.supervised_token_min = min(
            count_values
            + ([self.supervised_token_min] if self.supervised_token_min is not None else [])
        )
        self.supervised_token_max = max([self.supervised_token_max] + count_values)
        zero_positions = [index for index, value in enumerate(count_values) if value == 0]
        self.zero_supervision_count += len(zero_positions)
        if zero_positions:
            details = [
                {
                    "batch_position": position,
                    "sample_id": str(instances[position].get("sample_id", "unknown")),
                    "original_length": int(instances[position]["labels"].numel()),
                    "truncated_length": int(labels.shape[1]),
                }
                for position in zero_positions
            ]
            raise ValueError(
                "samples have zero supervised tokens after truncation: {}".format(details)
            )
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            batch["images"] = (
                torch.stack(images)
                if all(image.shape == images[0].shape for image in images)
                else images
            )
        return batch


class ComposeSelectionDataset(LazySupervisedDataset):
    """LazySupervisedDataset with per-sample (teacher set + cluster
    expert) routing for cluster-wise conditional-residual training.

    ``selections`` maps sample ids to manifest rows with either
    ``expert_ids`` (the union of the old-teacher set and the new cluster
    expert, the canonical training selection) or the pair of
    ``teacher_ids`` / ``cluster_expert_id`` (joined here).
    """

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
        expert_ids = selection.get("expert_ids")
        if expert_ids is None:
            teacher_ids = list(selection.get("teacher_ids", []))
            expert_ids = sorted(set(teacher_ids) | {int(selection["cluster_expert_id"])})
        item["expert_ids"] = list(expert_ids)
        return item


class ComposeSelectionCollator:
    """Formal padding plus per-sample unified Compose selections.

    Every sample routes to ``old_teacher_set + cluster_expert`` (1-4
    experts; four slots are the widest legal
    selection). Selections are padded to ``MAX_ACTIVE_EXPERTS`` slots
    with ``PAD_EXPERT_ID`` / zero gates and stored under the
    ``compose_selections`` batch key consumed by ``ComposeTrainer``.
    """

    def __init__(self, tokenizer) -> None:
        self._shim = DataCollatorForSupervisedDataset(tokenizer)

    def supervision_summary(self) -> Dict[str, object]:
        """Zero-supervision audit, delegated to the wrapped collator."""
        return self._shim.supervision_summary()

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sample_ids = [str(instance["sample_id"]) for instance in instances]
        selections = []
        for instance in instances:
            ids = [int(value) for value in instance["expert_ids"]]
            if len(ids) > MAX_ACTIVE_EXPERTS:
                raise ValueError(
                    "selection {} exceeds {} slots (old teacher pair + cluster "
                    "expert is the legal maximum)".format(ids, MAX_ACTIVE_EXPERTS)
                )
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate expert ids in sample selection: {}".format(ids))
            gates = [1.0] * len(ids)
            selections.append(pad_selection(tuple(ids), tuple(gates)))
        stripped = [
            {k: v for k, v in instance.items()
             if k not in ("sample_id", "expert_ids")}
            for instance in instances
        ]
        batch = self._shim(stripped)
        batch["sample_ids"] = sample_ids
        batch["compose_selections"] = selections
        return batch


def make_supervised_data_module(tokenizer, data_args: DataArguments) -> Dict:
    return {
        "train_dataset": LazySupervisedDataset(data_args.data_path, tokenizer, data_args),
        "eval_dataset": (
            LazySupervisedDataset(data_args.eval_data_path, tokenizer, data_args)
            if data_args.eval_data_path
            else None
        ),
        "data_collator": DataCollatorForSupervisedDataset(tokenizer),
    }
