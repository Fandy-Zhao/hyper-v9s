"""Format-controlled A/B evaluation for the composition validation.

Primary metrics come from the answer-token logits restricted to the legal
answer tokens {"A", "B"}:

  - prediction      = argmax over {logit_A, logit_B}
  - probability_A/B = softmax over {logit_A, logit_B}
  - answer NLL      = full-vocabulary negative log-likelihood at the answer
                      token ONLY (structural EOS excluded)

Free generation is retained as a diagnostic only.

Per-sample output carries the full preregistered field set (section 九):
sample_id, scene_id, task, split, seed, model/config, target, prediction,
correct, logit_A, logit_B, probability_A, probability_B, answer_token_nll,
polarity, negative_type, required_functions, queried_shape, queried_count,
queried_relation, true_count, latency, peak_memory.
"""

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

from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from llava.mm_utils import process_images, tokenizer_image_token

from compose.oracle.candidate_sets import CandidateSet
from compose.oracle.evaluator import _collate, _prepare_multimodal_batch
from compose.oracle.losses import compute_per_sample_nll

from .eval_task import _prompt
from .load_compose import load_compose_model


def _csv_ints(value: str):
    return tuple(int(item) for item in value.split(",") if item.strip())


def _csv_floats(value: str):
    return tuple(float(item) for item in value.split(",") if item.strip())


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _source_meta(record: dict) -> dict:
    source = record.get("source") or {}
    metadata = source.get("metadata") or {}
    return {
        "scene_id": str(source.get("scene_id") or ""),
        "task": str(source.get("task") or ""),
        "polarity": str(source.get("polarity") or ""),
        "negative_type": source.get("negative_type"),
        "required_functions": list(source.get("required_functions") or []),
        "queried_shape": metadata.get("queried_shape"),
        "queried_count": metadata.get("queried_count"),
        "queried_relation": metadata.get("queried_relation"),
        "true_count": metadata.get("true_count"),
    }


def ece(values: list, bins: int = 15) -> float:
    """Expected calibration error on p_A with fixed equal-width bins."""
    edges = [index / bins for index in range(bins + 1)]
    bin_conf = [0.0] * bins
    bin_acc = [0.0] * bins
    bin_count = [0] * bins
    for p_a, y_a in values:
        index = min(bins - 1, int(p_a * bins)) if p_a < 1.0 else bins - 1
        bin_count[index] += 1
        bin_conf[index] += p_a
        bin_acc[index] += float(y_a)
    total = sum(bin_count)
    if not total:
        return 0.0
    error = 0.0
    for index in range(bins):
        if bin_count[index]:
            conf = bin_conf[index] / bin_count[index]
            acc = bin_acc[index] / bin_count[index]
            error += (bin_count[index] / total) * abs(conf - acc)
    return error


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
    parser.add_argument("--split", default="test")
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

    a_id = int(bundle.tokenizer.encode("A", add_special_tokens=False)[0])
    b_id = int(bundle.tokenizer.encode("B", add_special_tokens=False)[0])
    eos_id = int(bundle.tokenizer.convert_tokens_to_ids("</s>"))
    answer_ids = {a_id, b_id}

    with open(args.question_file, encoding="utf-8") as handle:
        records = json.load(handle)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    # ---- batched forward: answer-token logits / NLL -------------------------
    rows = []
    forward_latencies = []
    forward_times = 0
    with torch.inference_mode():
        for offset in tqdm(range(0, len(records), args.batch_size), desc="answer-logits"):
            batch_records = records[offset : offset + args.batch_size]
            raw = _collate(batch_records, bundle, args.image_folder, args.device)
            prepared = _prepare_multimodal_batch(bundle, raw)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_started = time.time()
            logits = bundle.model(**prepared).logits
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_seconds = time.time() - forward_started
            forward_times += forward_seconds
            batch_size = len(batch_records)
            per_sample_latency = forward_seconds / batch_size
            forward_latencies.extend([per_sample_latency] * batch_size)
            peak_after_batch = (
                torch.cuda.max_memory_allocated(torch.device(args.device))
                if torch.cuda.is_available() else 0
            )
            labels = prepared["labels"]
            for index, record in enumerate(batch_records):
                label_row = labels[index].tolist()
                positions = [i for i, value in enumerate(label_row) if value != IGNORE_INDEX]
                if not positions:
                    raise AssertionError("sample {} has no supervised answer token".format(record["question_id"]))
                answer_token_pos = positions[0]
                supervised_ids = [int(label_row[i]) for i in positions]
                if supervised_ids[0] not in answer_ids or len(supervised_ids) > 2:
                    raise AssertionError(
                        "unexpected supervised region {} for {}".format(supervised_ids, record["question_id"])
                    )
                # LM convention: logits[t] predicts token t+1 (same shift as the
                # training loss). The answer token at position answer_token_pos
                # is predicted by logits at answer_token_pos - 1.
                answer_pos = answer_token_pos - 1
                token_logits = logits[index, answer_pos].float()
                logit_a = float(token_logits[a_id])
                logit_b = float(token_logits[b_id])
                max_logit = max(logit_a, logit_b)
                exp_a = torch.exp(torch.tensor(logit_a - max_logit))
                exp_b = torch.exp(torch.tensor(logit_b - max_logit))
                p_a = float(exp_a / (exp_a + exp_b))
                target = str(record["answer"])
                if target not in ("A", "B"):
                    raise AssertionError("unexpected target {!r}".format(target))
                target_id = a_id if target == "A" else b_id
                target_logit = token_logits[target_id]
                log_sum = torch.logsumexp(token_logits, dim=0)
                nll = float(log_sum - target_logit)
                correct = (logit_a > logit_b) == (target == "A")
                y_a = 1.0 if target == "A" else 0.0
                meta = _source_meta(record)
                row = {
                    "sample_id": str(record["question_id"]),
                    "image": str(record.get("image", "")),
                    "scene_id": meta["scene_id"],
                    "task": meta["task"] or str(record.get("task_id", "")),
                    "split": args.split,
                    "seed": args.experiment_seed,
                    "selection_name": args.selection_name,
                    "expert_ids": list(ids),
                    "gates": list(gates),
                    "normalization": args.normalization,
                    "target": target,
                    "prediction": "A" if logit_a > logit_b else "B",
                    "correct": correct,
                    "logit_A": logit_a,
                    "logit_B": logit_b,
                    "probability_A": p_a,
                    "probability_B": 1.0 - p_a,
                    "answer_token_nll": nll,
                    "brier": (p_a - y_a) ** 2,
                    "polarity": meta["polarity"],
                    "negative_type": meta["negative_type"],
                    "required_functions": meta["required_functions"],
                    "queried_shape": meta["queried_shape"],
                    "queried_count": meta["queried_count"],
                    "queried_relation": meta["queried_relation"],
                    "true_count": meta["true_count"],
                    "answer_token_position": answer_token_pos,
                    "answer_logits_position": answer_pos,
                    "supervised_token_ids": supervised_ids,
                    "latency": per_sample_latency,
                    "peak_memory": peak_after_batch,
                    "question": record.get("text", ""),
                    "options": (record.get("source") or {}).get("options"),
                }
                rows.append(row)

    nll_duration = time.time() - started

    # ---- free generation (diagnostic only) ----------------------------------
    peak_before_generation = (
        torch.cuda.max_memory_allocated(torch.device(args.device)) if torch.cuda.is_available() else 0
    )
    with torch.inference_mode():
        for row in tqdm(rows, desc="generation"):
            sample_id = row["sample_id"]
            record = next(r for r in records if str(r["question_id"]) == sample_id)
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
            output_ids = bundle.model.generate(
                input_ids=input_ids, images=image_tensor, do_sample=False,
                num_beams=1, max_new_tokens=args.max_new_tokens, use_cache=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            row["generation_seconds"] = time.time() - generation_started
            free = bundle.tokenizer.batch_decode(
                output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
            )[0].strip()
            row["free_prediction"] = free
            row["free_correct"] = free.strip().casefold() == row["target"].strip().casefold()

    peak_memory = (
        torch.cuda.max_memory_allocated(torch.device(args.device)) if torch.cuda.is_available() else 0
    )
    duration = time.time() - started
    with (output_dir / "per_sample.jsonl").open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")

    # ---- summary --------------------------------------------------------------
    def acc(subset):
        return statistics.fmean(float(r["correct"]) for r in subset) if subset else 0.0

    positives = [r for r in rows if r["polarity"] == "positive"]
    negatives = [r for r in rows if r["polarity"] == "negative"]
    gen_latencies = [float(r["generation_seconds"]) for r in rows]
    forward_sorted = sorted(forward_latencies)
    brier_by_type = {}
    for r in rows:
        key = str(r["negative_type"] or "positive")
        brier_by_type.setdefault(key, []).append(float(r["brier"]))
    manifest = {}
    manifest_path = os.path.join(args.checkpoint_dir, "compose_experts.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    total_parameters = sum(parameter.numel() for parameter in bundle.model.parameters())
    summary = {
        "samples": len(rows),
        "accuracy": acc(rows),
        "accuracy_percent": 100.0 * acc(rows),
        "positive_accuracy": acc(positives),
        "negative_accuracy": acc(negatives),
        "balanced_accuracy": 0.5 * (acc(positives) + acc(negatives)),
        "mean_answer_token_nll": statistics.fmean(float(r["answer_token_nll"]) for r in rows),
        "brier": statistics.fmean(float(r["brier"]) for r in rows),
        "brier_by_negative_type": {
            key: statistics.fmean(values) for key, values in sorted(brier_by_type.items())
        },
        "ece_15_bins": ece([(float(r["probability_A"]), 1.0 if r["target"] == "A" else 0.0) for r in rows], 15),
        "hard_negative": {
            key: {
                "count": len(subset),
                "accuracy": acc(subset),
                "mean_nll": statistics.fmean(float(r["answer_token_nll"]) for r in subset) if subset else 0.0,
            }
            for key, subset in {
                "count_negative": [r for r in rows if r["negative_type"] == "count_negative"],
                "attribute_negative": [r for r in rows if r["negative_type"] == "attribute_negative"],
                "relation_negative": [r for r in rows if r["negative_type"] == "relation_negative"],
                "positive": positives,
            }.items()
        },
        "free_generation_accuracy": acc(rows) and statistics.fmean(float(r["free_correct"]) for r in rows),
        "mean_answer_logit_seconds": statistics.fmean(forward_latencies),
        "median_answer_logit_seconds": statistics.median(forward_latencies),
        "p50_answer_logit_seconds": forward_sorted[len(forward_sorted) // 2] if forward_sorted else 0.0,
        "p95_answer_logit_seconds": forward_sorted[min(len(forward_sorted) - 1, int(0.95 * len(forward_sorted)))] if forward_sorted else 0.0,
        "mean_generation_seconds": statistics.fmean(gen_latencies) if gen_latencies else 0.0,
        "median_generation_seconds": statistics.median(gen_latencies) if gen_latencies else 0.0,
        "nll_throughput_samples_per_second": len(rows) / nll_duration if nll_duration else 0.0,
        "peak_memory_bytes": peak_memory,
        "peak_memory_before_generation_bytes": peak_before_generation,
        "total_parameters": total_parameters,
        "adapter_parameter_count": manifest.get("metrics", {}).get("adapter_parameter_count"),
        "active_expert_count": len(ids),
        "selection": candidate.to_dict(),
        "selection_name": args.selection_name,
        "checkpoint": args.checkpoint_dir,
        "question_file": args.question_file,
        "experiment_seed": args.experiment_seed,
        "evaluation_seed": 42,
        "git_commit": _git_commit(),
        "command": sys.argv,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
