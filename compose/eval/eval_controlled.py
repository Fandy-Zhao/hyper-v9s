"""Evaluate one controlled-benchmark selection with NLL and exact generation."""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images, tokenizer_image_token

from compose.data.records import answer_text, question_text
from compose.oracle.candidate_sets import CandidateSet
from compose.oracle.evaluator import _candidate_nll, _collate, _prepare_multimodal_batch

from .eval_task import _prompt
from .load_compose import load_compose_model


def _csv_ints(value: str):
    return tuple(int(item) for item in value.split(",") if item.strip())


def _csv_floats(value: str):
    return tuple(float(item) for item in value.split(",") if item.strip())


def exact_match(prediction: str, target: str) -> bool:
    return prediction.strip().casefold() == target.strip().casefold()


def summarize(rows, duration_seconds: float):
    latencies = [float(row["generation_seconds"]) for row in rows]
    return {
        "samples": len(rows),
        "mean_nll": statistics.fmean(float(row["nll"]) for row in rows),
        "accuracy": statistics.fmean(float(row["correct"]) for row in rows),
        "accuracy_percent": 100.0 * statistics.fmean(float(row["correct"]) for row in rows),
        "duration_seconds": duration_seconds,
        "samples_per_second": len(rows) / duration_seconds,
        "mean_generation_seconds": statistics.fmean(latencies),
        "median_generation_seconds": statistics.median(latencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--projector-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-name", required=True)
    parser.add_argument("--expert-ids", default="")
    parser.add_argument("--gates", default="")
    parser.add_argument("--normalization", choices=("none", "l1", "l2"), default="none")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experiment-seed", type=int, required=True)
    args = parser.parse_args()
    ids = _csv_ints(args.expert_ids)
    gates = _csv_floats(args.gates) or tuple(1.0 for _ in ids)
    if len(ids) not in (0, 1, 2) or len(ids) != len(gates):
        raise ValueError("selection requires zero, one, or two experts")
    output_dir = Path(args.output_dir)
    if (output_dir / "summary.json").exists() or (output_dir / "per_sample.jsonl").exists():
        raise FileExistsError("refusing completed/partial output directory: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    started = time.time()
    bundle = load_compose_model(
        model_path=args.model_path, checkpoint_dir=args.checkpoint_dir,
        vision_tower=args.vision_tower, projector_path=args.projector_path,
        expert_id=None, device=args.device, dtype=torch.bfloat16,
        model_max_length=2048,
    )
    missing = sorted(set(ids) - set(bundle.expert_pool.expert_ids()))
    if missing:
        raise KeyError("selection references missing experts: {}".format(missing))
    candidate = CandidateSet(0, ids, gates, args.normalization)
    if ids:
        bundle.expert_pool.manager.set_default_selection(
            ids, gates, normalization=args.normalization
        )
    else:
        bundle.expert_pool.manager.clear_default_selection()
    with open(args.question_file, encoding="utf-8") as handle:
        records = json.load(handle)
    if args.max_samples is not None:
        records = records[:args.max_samples]

    nll_by_id = {}
    token_count_by_id = {}
    for offset in tqdm(range(0, len(records), args.batch_size), desc="nll"):
        batch_records = records[offset:offset + args.batch_size]
        raw = _collate(batch_records, bundle, args.image_folder, args.device)
        prepared = _prepare_multimodal_batch(bundle, raw)
        losses, counts = _candidate_nll(bundle, candidate, prepared)
        for index, record in enumerate(batch_records):
            sample_id = str(record["question_id"])
            nll_by_id[sample_id] = float(losses[index])
            token_count_by_id[sample_id] = int(counts[index])

    rows = []
    predictions_path = output_dir / "per_sample.jsonl"
    with predictions_path.open("w", encoding="utf-8") as output:
        for record in tqdm(records, desc="generation"):
            prompt = _prompt(record, bundle.model.config, "vicuna_v1")
            input_ids = tokenizer_image_token(
                prompt, bundle.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(args.device)
            image = Image.open(os.path.join(args.image_folder, str(record["image"]))).convert("RGB")
            image_tensor = process_images([image], bundle.image_processor, bundle.model.config)[0]
            image_tensor = image_tensor.unsqueeze(0).to(device=args.device, dtype=torch.bfloat16)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generation_started = time.time()
            with torch.inference_mode():
                output_ids = bundle.model.generate(
                    input_ids=input_ids, images=image_tensor, do_sample=False,
                    num_beams=1, max_new_tokens=args.max_new_tokens, use_cache=True,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generation_seconds = time.time() - generation_started
            prediction = bundle.tokenizer.batch_decode(
                output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
            )[0].strip()
            sample_id = str(record["question_id"])
            target = answer_text(record)
            row = {
                "sample_id": sample_id,
                "task_id": str(record.get("task_id", "unknown")),
                "selection_name": args.selection_name,
                "expert_ids": list(ids), "gates": list(gates),
                "normalization": args.normalization,
                "nll": nll_by_id[sample_id],
                "target_token_count": token_count_by_id[sample_id],
                "prediction": prediction, "target": target,
                "correct": exact_match(prediction, target),
                "generation_seconds": generation_seconds,
            }
            rows.append(row)
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()

    duration = time.time() - started
    summary = {
        **summarize(rows, duration),
        "selection_name": args.selection_name,
        "selection": candidate.to_dict(),
        "checkpoint": args.checkpoint_dir,
        "question_file": args.question_file,
        "experiment_seed": args.experiment_seed,
        "evaluation_seed": 42,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(torch.device(args.device)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": sys.argv,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
