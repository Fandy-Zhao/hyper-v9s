import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from typing import Dict, List, Sequence

import torch
from PIL import Image
from tqdm import tqdm

from llava import conversation as conversation_lib
from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images

from compose.eval.load_compose import load_compose_model
from compose.train.data import preprocess, preprocess_multimodal

from .cache import config_hash
from .candidate_sets import CandidateSet, build_candidate_sets
from .losses import compute_per_sample_nll
from .metrics import compute_oracle_record, summarize_oracle_records


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _chunk(records: Sequence[Dict[str, object]], count: int, index: int):
    if count <= 0 or not 0 <= index < count:
        raise ValueError("invalid chunk selection {}/{}".format(index, count))
    size = int(math.ceil(len(records) / count))
    return records[index * size : (index + 1) * size]


def _answer(record: Dict[str, object]) -> str:
    if "answer" in record:
        return str(record["answer"])
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for message in reversed(conversations):
            if message.get("from") == "gpt":
                return str(message["value"])
    raise ValueError("record has no teacher answer")


def _instance(record, tokenizer, data_args):
    source = [[
        {"from": "human", "value": DEFAULT_IMAGE_TOKEN + "\n" + str(record["text"])},
        {"from": "gpt", "value": _answer(record)},
    ]]
    source = preprocess_multimodal(copy.deepcopy(source), data_args)
    encoded = preprocess(source, tokenizer, has_image=True)
    return encoded["input_ids"][0], encoded["labels"][0]


def _collate(records, bundle, image_folder, device):
    data_args = type("OracleDataArguments", (), {
        "is_multimodal": True,
        "mm_use_im_start_end": False,
    })()
    pairs = [_instance(record, bundle.tokenizer, data_args) for record in records]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [pair[0] for pair in pairs],
        batch_first=True,
        padding_value=bundle.tokenizer.pad_token_id,
    )[:, : bundle.context_length]
    labels = torch.nn.utils.rnn.pad_sequence(
        [pair[1] for pair in pairs], batch_first=True, padding_value=IGNORE_INDEX
    )[:, : bundle.context_length]
    images = [
        Image.open(os.path.join(image_folder, str(record["image"]))).convert("RGB")
        for record in records
    ]
    image_tensor = process_images(images, bundle.image_processor, bundle.model.config)
    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "attention_mask": input_ids.ne(bundle.tokenizer.pad_token_id).to(device),
        "images": image_tensor.to(device=device, dtype=torch.bfloat16),
    }


def _prepare_multimodal_batch(bundle, batch):
    with torch.inference_mode():
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = bundle.model.prepare_inputs_labels_for_multimodal(
            batch["input_ids"],
            None,
            batch["attention_mask"],
            None,
            batch["labels"],
            batch["images"],
        )
    if inputs_embeds is None or labels is None:
        raise AssertionError("oracle multimodal preparation did not produce embeddings/labels")
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "inputs_embeds": inputs_embeds,
        "labels": labels,
    }


def _candidate_nll(bundle, candidate: CandidateSet, batch):
    manager = bundle.expert_pool.manager
    if not candidate.expert_ids:
        manager.clear_default_selection()
        with torch.inference_mode():
            logits = bundle.model(**batch).logits
    else:
        selection = manager.make_selection(
            candidate.expert_ids,
            batch_size=batch["labels"].shape[0],
            gates=candidate.gates,
            device=batch["labels"].device,
            normalization=candidate.normalization,
        )
        with manager.selection_context(selection), torch.inference_mode():
            logits = bundle.model(**batch).logits
    details = compute_per_sample_nll(logits, batch["labels"], return_details=True)
    return details["mean_nll"].cpu(), details["valid_token_count"].cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--projector-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-max-length", type=int, default=2048)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    started = time.time()
    bundle = load_compose_model(
        model_path=args.model_path,
        checkpoint_dir=args.checkpoint_dir,
        vision_tower=args.vision_tower,
        projector_path=args.projector_path,
        expert_id=None,
        device=args.device,
        dtype=torch.bfloat16,
        model_max_length=args.model_max_length,
    )
    conversation_lib.default_conversation = conv_templates["vicuna_v1"]
    candidates = build_candidate_sets(bundle.expert_pool.expert_ids())
    candidate_config = [candidate.to_dict() for candidate in candidates]
    run_config = {
        "checkpoint": args.checkpoint_dir,
        "candidates": candidate_config,
        "model_max_length": args.model_max_length,
        "teacher_forcing": True,
        "dtype": "bfloat16",
    }
    run_hash = config_hash(run_config)
    model_commit = _git_commit()
    with open(args.question_file, encoding="utf-8") as handle:
        records = _chunk(json.load(handle), args.num_chunks, args.chunk_idx)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    oracle_rows = []
    with open(args.output_file, "w", encoding="utf-8") as output:
        for offset in tqdm(range(0, len(records), args.batch_size)):
            batch_records = records[offset : offset + args.batch_size]
            raw_batch = _collate(batch_records, bundle, args.image_folder, args.device)
            batch = _prepare_multimodal_batch(bundle, raw_batch)
            all_losses = []
            token_counts = None
            for candidate in candidates:
                losses, counts = _candidate_nll(bundle, candidate, batch)
                all_losses.append(losses)
                if token_counts is None:
                    token_counts = counts
                elif not torch.equal(token_counts, counts):
                    raise AssertionError("target token counts changed across candidates")
            matrix = torch.stack(all_losses, dim=1)
            for row_index, record in enumerate(batch_records):
                losses = [float(value) for value in matrix[row_index].tolist()]
                metrics = compute_oracle_record(losses, candidates)
                row = {
                    "sample_id": str(record.get("question_id", offset + row_index)),
                    "task_id": str(record.get("task_id", args.task_id)),
                    "candidate_expert_ids": [list(value.expert_ids) for value in candidates],
                    "set_ids": [value.index for value in candidates],
                    "set_gates": [list(value.gates) for value in candidates],
                    "set_normalization": [value.normalization for value in candidates],
                    "set_nll": losses,
                    "target_token_count": int(token_counts[row_index]),
                    "checkpoint_ids": bundle.expert_pool.expert_ids(),
                    "model_commit": model_commit,
                    "config_hash": run_hash,
                    **metrics,
                }
                output.write(json.dumps(row, sort_keys=True) + "\n")
                output.flush()
                oracle_rows.append(row)

    duration = time.time() - started
    summary = {
        **summarize_oracle_records(oracle_rows),
        "checkpoint": args.checkpoint_dir,
        "command": sys.argv,
        "config_hash": run_hash,
        "candidate_forward_count_per_sample": len(candidates),
        "candidate_sets": candidate_config,
        "duration_seconds": duration,
        "samples_per_second": len(records) / duration if duration else 0.0,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(torch.device(args.device)),
        "mean_nll_by_set": [
            sum(float(row["set_nll"][index]) for row in oracle_rows) / len(oracle_rows)
            for index in range(len(candidates))
        ],
        "model_commit": model_commit,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.summary_file)), exist_ok=True)
    with open(args.summary_file, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
