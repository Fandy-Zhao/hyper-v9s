import argparse
import json
import math
import os
import subprocess
import sys
import time

import torch
from tqdm import tqdm

from llava import conversation as conversation_lib
from llava.conversation import conv_templates

from compose.eval.load_compose import load_compose_model

from .cache import config_hash
from .candidate_sets import CandidateSet
from .evaluator import _candidate_nll, _chunk, _collate, _prepare_multimodal_batch


def _csv_ints(value: str):
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _csv_floats(value: str):
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


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
    parser.add_argument("--selection-name", required=True)
    parser.add_argument("--expert-ids", default="")
    parser.add_argument("--gates", default="")
    parser.add_argument("--normalization", choices=("none", "l1", "l2"), default="none")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-max-length", type=int, default=2048)
    args = parser.parse_args()
    expert_ids = _csv_ints(args.expert_ids)
    gates = _csv_floats(args.gates) or tuple(1.0 for _ in expert_ids)
    if len(expert_ids) not in (0, 1, 2) or len(gates) != len(expert_ids):
        raise ValueError("selection requires zero, one, or two experts with matching gates")
    candidate = CandidateSet(0, expert_ids, gates, args.normalization)

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
    missing = sorted(set(expert_ids) - set(bundle.expert_pool.expert_ids()))
    if missing:
        raise KeyError("selection references missing experts: {}".format(missing))
    conversation_lib.default_conversation = conv_templates["vicuna_v1"]
    with open(args.question_file, encoding="utf-8") as handle:
        records = _chunk(json.load(handle), args.num_chunks, args.chunk_idx)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    selection_config = candidate.to_dict()
    run_hash = config_hash(
        {"checkpoint": args.checkpoint_dir, "selection": selection_config, "dtype": "bfloat16"}
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    total_nll = 0.0
    sample_count = 0
    with open(args.output_file, "w", encoding="utf-8") as output:
        for offset in tqdm(range(0, len(records), args.batch_size)):
            batch_records = records[offset : offset + args.batch_size]
            raw_batch = _collate(batch_records, bundle, args.image_folder, args.device)
            batch = _prepare_multimodal_batch(bundle, raw_batch)
            losses, counts = _candidate_nll(bundle, candidate, batch)
            for index, record in enumerate(batch_records):
                nll = float(losses[index])
                row = {
                    "sample_id": str(record.get("question_id", offset + index)),
                    "task_id": str(record.get("task_id", "unknown")),
                    "selection_name": args.selection_name,
                    "expert_ids": list(expert_ids),
                    "gates": list(gates),
                    "normalization": args.normalization,
                    "nll": nll,
                    "target_token_count": int(counts[index]),
                    "checkpoint": args.checkpoint_dir,
                    "model_commit": commit,
                    "config_hash": run_hash,
                }
                output.write(json.dumps(row, sort_keys=True) + "\n")
                output.flush()
                total_nll += nll
                sample_count += 1
    duration = time.time() - started
    summary = {
        "samples": sample_count,
        "selection_name": args.selection_name,
        "selection": selection_config,
        "checkpoint": args.checkpoint_dir,
        "mean_nll": total_nll / sample_count,
        "duration_seconds": duration,
        "samples_per_second": sample_count / duration if duration else 0.0,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(torch.device(args.device)),
        "model_commit": commit,
        "config_hash": run_hash,
        "command": sys.argv,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.summary_file)), exist_ok=True)
    with open(args.summary_file, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
