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
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from compose.adapters.types import ComposeSelection, pad_selection
from compose.eval.load_compose import load_compose_model
from compose.teacher.scorer import answer_token_nll
from compose.train.data import DataCollatorForSupervisedDataset
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token


def _records(question_file: str) -> List[Dict]:
    with open(question_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sample_text(record: Dict) -> Tuple[str, Optional[str]]:
    if "conversations" in record:
        for message in record["conversations"]:
            if message["from"] == "human":
                text = message["value"]
            elif message["from"] == "gpt":
                answer = message["value"]
        return text, answer
    return record.get("text", ""), record.get("answer")


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
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.selections, "r", encoding="utf-8") as handle:
        selections = json.load(handle)
    records = _records(args.question_file)
    if args.max_samples:
        records = records[: args.max_samples]

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
    image_processor = bundle.image_processor
    conv_template = conversation_lib.conv_templates["vicuna_v1"].copy()
    collator = DataCollatorForSupervisedDataset(tokenizer)

    results = {}
    for record in records:
        sample_id = str(record.get("id", record.get("question_id")))
        if sample_id not in selections:
            continue
        text, answer = _sample_text(record)
        if not answer:
            raise ValueError("record {} has no answer".format(sample_id))
        image_path = os.path.join(args.image_folder, record["image"])
        if not os.path.isfile(image_path):
            raise ValueError("missing image: {}".format(image_path))

        conv_template.messages = []
        conv_template.append_message(conv_template.roles[0], text)
        conv_template.append_message(conv_template.roles[1], answer)
        prompt = conv_template.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, return_tensors="pt"
        ).to(args.device)
        image = image_processor.preprocess(
            Image.open(image_path).convert("RGB"), return_tensors="pt"
        )["pixel_values"][0].to(args.device, dtype=torch.bfloat16)
        image = image.unsqueeze(0)
        # Expand labels in lockstep with the multimodal sequence so they
        # align with the forward logits (image patch tokens included).
        raw_labels = input_ids.clone().unsqueeze(0)
        expanded = model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids.unsqueeze(0),
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=raw_labels,
            images=image,
        )
        prepared_ids = expanded[0]
        prepared_inputs_embeds = expanded[4]
        prepared_labels = expanded[5]

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
                    labels=prepared_labels,
                    return_dict=True,
                )
            logits = outputs.logits
            nll = answer_token_nll(
                logits, prepared_labels,
                eos_token_id=tokenizer.eos_token_id,
                include_eos=False,
            )
            per_set[set_key] = float(nll["mean_nll"])
        results[sample_id] = per_set
        print(
            "  sample {}: {}".format(
                sample_id,
                ", ".join("{}={:.4f}".format(key, value) for key, value in per_set.items()),
            ),
            flush=True,
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    print("NLL results written to {}".format(args.output))


if __name__ == "__main__":
    main()
