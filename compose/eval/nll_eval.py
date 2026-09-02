"""Compose answer-NLL calculator over arbitrary per-sample expert sets.

Loads a compose checkpoint (old experts), then for each sample and each
requested expert set computes the teacher-forced answer NLL using the
Stage-04 scorer (answer-masked, token mean). Output is
``{sample_id: {set_key: mean_nll}}``.

This is the workhorse behind the answer-supervised teacher search
(empty/single/pair over the per-sample Top-M) and the residual judgment
on real UCIT data. Testing never uses answers (see the acceptance
module); this module is train-time only.
"""

import argparse
import json
import os
from typing import Dict, List

import torch
from compose.adapters.types import ComposeSelection, pad_selection
from compose.eval.load_compose import load_compose_model
from compose.eval.sharding import partial_path, shard_records
from compose.train.arguments import DataArguments
from compose.train.data import DataCollatorForSupervisedDataset, LazySupervisedDataset
from compose.v7.training import supervised_token_mask, teacher_forcing_token_nll
from compose.v7.provenance import (
    build_runtime_contract,
    load_runtime_contract,
    validate_runtime_contract,
)
from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX


def _records(question_file: str) -> List[Dict]:
    with open(question_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_nll_output(output: str, results: Dict[str, object], num_shards: int, shard_index: int) -> str:
    """Persist per-sample NLL results, honoring multi-shard output naming."""
    target = partial_path(output, shard_index) if num_shards > 1 else output
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--projector-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--selections", required=True,
                        help="json: {sample_id: {set_key: [expert_ids]}}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-aspect-ratio", default="pad")
    parser.add_argument("--runtime-contract")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    with open(args.selections, "r", encoding="utf-8") as handle:
        selections = json.load(handle)
    bundle = load_compose_model(
        model_path=args.model_path,
        checkpoint_dir=args.checkpoint_dir,
        vision_tower=args.vision_tower,
        projector_path=args.projector_path,
        expert_id=None,
        device=args.device,
        dtype=torch.bfloat16,
        model_max_length=2048,
    )
    model = bundle.model
    tokenizer = bundle.tokenizer
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    data_args = DataArguments(
        data_path=args.question_file,
        image_folder=args.image_folder,
        image_aspect_ratio=args.image_aspect_ratio,
    )
    if args.runtime_contract:
        validate_runtime_contract(
            load_runtime_contract(args.runtime_contract),
            build_runtime_contract(
                image_aspect_ratio=args.image_aspect_ratio,
                vision_tower=args.vision_tower,
                mm_vision_select_layer=-2,
                mm_vision_select_feature="patch",
                mm_projector_type="mlp2x_gelu",
                projector_path=args.projector_path,
            ),
            "pruning_nll",
        )
    data_args.image_processor = bundle.image_processor
    data_args.is_multimodal = True
    data_args.mm_use_im_start_end = False
    dataset = LazySupervisedDataset(args.question_file, tokenizer, data_args)
    collator = DataCollatorForSupervisedDataset(tokenizer)
    indices = list(range(len(dataset)))
    if args.max_samples:
        indices = indices[: args.max_samples]
    indices = shard_records(indices, args.num_shards, args.shard_index)

    results = {}
    for record_index in indices:
        record = dataset.records[record_index]
        sample_id = str(record.get("id", record.get("question_id")))
        if sample_id not in selections:
            continue
        batch = collator([dataset[record_index]])
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["labels"].to(args.device)
        images = batch["images"].to(args.device, dtype=torch.bfloat16)
        # Expand labels in lockstep with the multimodal sequence so they
        # align with the forward logits (image patch tokens included).
        expanded = model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=labels,
            images=images,
        )
        prepared_ids = expanded[0]
        prepared_attention_mask = expanded[2]
        prepared_inputs_embeds = expanded[4]
        prepared_labels = expanded[5]
        supervised_token_count = int(supervised_token_mask(prepared_labels).sum().item())
        if supervised_token_count <= 0:
            raise ValueError("record {} has zero supervised answer tokens".format(sample_id))

        per_set = {}
        for set_key, expert_ids in sorted(selections[sample_id].items()):
            ids = [int(value) for value in expert_ids]
            padded_ids, padded_gates = pad_selection(tuple(ids), tuple(1.0 for _ in ids))
            selection = ComposeSelection(
                torch.tensor([padded_ids], dtype=torch.long),
                torch.tensor([padded_gates], dtype=torch.float32),
            )
            with torch.inference_mode(), bundle.expert_pool.manager.selection_context(selection):
                outputs = model(
                    input_ids=prepared_ids,
                    inputs_embeds=prepared_inputs_embeds,
                    attention_mask=prepared_attention_mask,
                    labels=prepared_labels,
                    return_dict=True,
                )
            logits = outputs.logits
            nll = teacher_forcing_token_nll(logits, prepared_labels)
            per_set[set_key] = {
                "mean_answer_nll": float(nll),
                "supervised_token_count": supervised_token_count,
                "supervision_contract": "compose_train_preprocess_v1_shifted",
                "eos_policy": "same_as_training_labels",
            }
        results[sample_id] = per_set
        print(
            "  sample {}: {}".format(
                sample_id,
                ", ".join(
                    "{}={:.4f}".format(key, value["mean_answer_nll"])
                    for key, value in per_set.items()
                ),
            ),
            flush=True,
        )

    target = write_nll_output(args.output, results, args.num_shards, args.shard_index)
    print("NLL results written to {} (shard {}/{})".format(
        target, args.shard_index, args.num_shards
    ))


if __name__ == "__main__":
    main()
