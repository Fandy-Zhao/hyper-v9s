import argparse
import json
import math
import os
import subprocess
import sys
import time
from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token

from .load_compose import load_compose_model, load_peft_model


def _chunk(records: List[Dict[str, object]], count: int, index: int):
    if count <= 0 or not 0 <= index < count:
        raise ValueError("invalid chunk selection {}/{}".format(index, count))
    size = int(math.ceil(len(records) / count))
    return records[index * size : (index + 1) * size]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _prompt(record, model_config, conv_mode):
    question = str(record["text"])
    image_token = DEFAULT_IMAGE_TOKEN
    if model_config.mm_use_im_start_end:
        image_token = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN
    conversation = conv_templates[conv_mode].copy()
    conversation.append_message(conversation.roles[0], image_token + "\n" + question)
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-kind", choices=("compose", "peft"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--projector-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--run-summary-file", required=True)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--gate", type=float, default=1.0)
    parser.add_argument("--normalization", choices=("none", "l1", "l2"), default="none")
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--model-max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    started = time.time()
    common = dict(
        model_path=args.model_path,
        checkpoint_dir=args.checkpoint_dir,
        vision_tower=args.vision_tower,
        projector_path=args.projector_path,
        device=args.device,
        dtype=torch.bfloat16,
        model_max_length=args.model_max_length,
    )
    if args.adapter_kind == "compose":
        bundle = load_compose_model(
            expert_id=args.expert_id,
            gate=args.gate,
            normalization=args.normalization,
            **common
        )
    else:
        bundle = load_peft_model(**common)

    with open(args.question_file, "r", encoding="utf-8") as handle:
        all_records = json.load(handle)
    records = _chunk(all_records, args.num_chunks, args.chunk_idx)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    os.makedirs(os.path.dirname(os.path.abspath(args.answers_file)), exist_ok=True)
    with open(args.answers_file, "w", encoding="utf-8") as output:
        for record in tqdm(records):
            prompt = _prompt(record, bundle.model.config, args.conv_mode)
            input_ids = tokenizer_image_token(
                prompt, bundle.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(args.device)
            image = Image.open(
                os.path.join(args.image_folder, str(record["image"]))
            ).convert("RGB")
            image_tensor = process_images(
                [image], bundle.image_processor, bundle.model.config
            )[0].unsqueeze(0).to(device=args.device, dtype=torch.bfloat16)
            with torch.inference_mode():
                output_ids = bundle.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            input_length = input_ids.shape[1]
            text = bundle.tokenizer.batch_decode(
                output_ids[:, input_length:], skip_special_tokens=True
            )[0].strip()
            output.write(
                json.dumps(
                    {
                        "question_id": str(record["question_id"]),
                        "prompt": str(record["text"]),
                        "text": text,
                        "model_id": args.adapter_kind,
                        "metadata": {
                            "checkpoint": args.checkpoint_dir,
                            "git_commit": _git_commit(),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    duration = time.time() - started
    peak_memory = (
        torch.cuda.max_memory_allocated(torch.device(args.device))
        if torch.cuda.is_available()
        else 0
    )
    summary = {
        "adapter_kind": args.adapter_kind,
        "checkpoint": args.checkpoint_dir,
        "command": sys.argv,
        "git_commit": _git_commit(),
        "seed": 42,
        "samples": len(records),
        "duration_seconds": duration,
        "samples_per_second": len(records) / duration if duration else 0.0,
        "peak_memory_bytes": peak_memory,
        "load_summary": bundle.load_summary,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.run_summary_file)), exist_ok=True)
    with open(args.run_summary_file, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
