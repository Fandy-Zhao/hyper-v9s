import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys

import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token

from .load_compose import load_compose_model, load_peft_model


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _fixed_logits(args, record, checkpoint_dir):
    loader_kwargs = {
        "model_path": args.model_path,
        "checkpoint_dir": checkpoint_dir,
        "vision_tower": args.vision_tower,
        "projector_path": args.projector_path,
        "device": args.device,
        "dtype": torch.bfloat16,
        "model_max_length": 2048,
    }
    if args.adapter_kind == "compose":
        bundle = load_compose_model(
            expert_id=args.expert_id,
            gate=1.0,
            normalization="none",
            **loader_kwargs,
        )
    else:
        bundle = load_peft_model(**loader_kwargs)
    conversation = conv_templates["vicuna_v1"].copy()
    conversation.append_message(
        conversation.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + str(record["text"])
    )
    conversation.append_message(conversation.roles[1], None)
    input_ids = tokenizer_image_token(
        conversation.get_prompt(),
        bundle.tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(args.device)
    image = Image.open(
        os.path.join(args.image_folder, str(record["image"]))
    ).convert("RGB")
    image_tensor = process_images(
        [image], bundle.image_processor, bundle.model.config
    )[0].unsqueeze(0).to(device=args.device, dtype=torch.bfloat16)
    with torch.inference_mode():
        logits = bundle.model(input_ids=input_ids, images=image_tensor).logits
    fixed = logits[0, -1].float().cpu().contiguous()
    summary = bundle.load_summary
    del logits, image_tensor, input_ids, bundle
    gc.collect()
    torch.cuda.empty_cache()
    return fixed, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--comparison-checkpoint-dir")
    parser.add_argument("--projector-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument(
        "--adapter-kind", choices=("compose", "peft"), default="compose"
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    with open(args.question_file, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    record = records[args.sample_index]
    comparison_checkpoint = args.comparison_checkpoint_dir or args.checkpoint_dir
    first, first_summary = _fixed_logits(args, record, args.checkpoint_dir)
    second, second_summary = _fixed_logits(args, record, comparison_checkpoint)
    max_abs = (first - second).abs().max().item()
    if not torch.equal(first, second):
        raise AssertionError(
            "fixed-sample logits changed across strict reloads; max_abs={}".format(max_abs)
        )
    digest = hashlib.sha256(first.numpy().tobytes()).hexdigest()
    result = {
        "checkpoint": args.checkpoint_dir,
        "comparison_checkpoint": comparison_checkpoint,
        "adapter_kind": args.adapter_kind,
        "command": sys.argv,
        "git_commit": _git_commit(),
        "sample_index": args.sample_index,
        "question_id": str(record["question_id"]),
        "logit_count": first.numel(),
        "logits_sha256": digest,
        "maximum_absolute_difference": max_abs,
        "exactly_equal": True,
        "first_load": first_summary,
        "second_load": second_summary,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
