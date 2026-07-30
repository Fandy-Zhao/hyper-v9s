import argparse
import copy
import json
import time

import transformers
from tqdm import tqdm

from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX

from .arguments import DataArguments
from .data import preprocess, preprocess_multimodal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--model-max-length", type=int, default=2048)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    conversation_lib.default_conversation = conversation_lib.conv_templates.get(
        args.version, conversation_lib.conv_templates["vicuna_v1"]
    )
    data_args = DataArguments(data_path=args.data_path, is_multimodal=True)
    data_args.mm_use_im_start_end = False
    with open(args.data_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)

    started = time.time()
    total = 0
    minimum = None
    maximum = 0
    zero_sample_ids = []
    for index, record in enumerate(tqdm(records)):
        sources = [copy.deepcopy(record["conversations"])]
        has_image = "image" in record
        if has_image:
            sources = preprocess_multimodal(sources, data_args)
        labels = preprocess(sources, tokenizer, has_image=has_image)["labels"][0]
        supervised = int(labels[: args.model_max_length].ne(IGNORE_INDEX).sum())
        total += supervised
        minimum = supervised if minimum is None else min(minimum, supervised)
        maximum = max(maximum, supervised)
        if supervised == 0:
            zero_sample_ids.append(str(record.get("id", index)))

    result = {
        "data_path": args.data_path,
        "model_path": args.model_path,
        "model_max_length": args.model_max_length,
        "prompt_version": args.version,
        "samples": len(records),
        "supervised_token_min": minimum or 0,
        "supervised_token_mean": total / len(records) if records else 0.0,
        "supervised_token_max": maximum,
        "supervised_token_total": total,
        "zero_supervision_count": len(zero_sample_ids),
        "zero_supervision_sample_ids": zero_sample_ids,
        "duration_seconds": time.time() - started,
    }
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))
    if zero_sample_ids:
        raise ValueError("zero-supervision samples found: {}".format(zero_sample_ids))


if __name__ == "__main__":
    main()
