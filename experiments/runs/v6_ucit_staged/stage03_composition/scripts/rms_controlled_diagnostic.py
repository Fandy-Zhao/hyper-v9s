"""Frozen train-calibration RMS collection and one-shot controlled diagnosis."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from compose.eval.load_compose import load_compose_model
from compose.experts import ExpertMetadata, ExpertRegistry
from compose.lora import (AdapterBridge, CompositionRuntime, ExpertComposer,
                          RMSCompositionConfig, RMSStatistics, StatisticKey)
from compose.lora.rms_composition import frozen_coefficients
from compose.oracle.evaluator import _collate, _prepare_multimodal_batch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration-questions", required=True)
    parser.add_argument("--test-questions", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--expert-ids", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    ids = tuple(map(int, args.expert_ids.split(",")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    bundle = load_compose_model(
        model_path="/data/ckpt/zhaozhuofan/models/llava-v1.5-7b", checkpoint_dir=args.checkpoint,
        vision_tower="/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336",
        projector_path="/data/ckpt/zhaozhuofan/models/llava-v1.5-7b/mm_projector.bin",
        expert_id=None, device="cuda:0", dtype=torch.bfloat16, model_max_length=2048)
    bridge = AdapterBridge(bundle.model, verify_ddp=False)
    registry = ExpertRegistry()
    for expert_id in bridge.expert_ids:
        registry.register(ExpertMetadata(expert_id=expert_id, adapter_name=str(expert_id)))
    config = RMSCompositionConfig()
    checkpoint_hash = hashlib.sha256((sha256(Path(args.checkpoint) / "compose_experts.json") + sha256(Path(args.checkpoint) / "compose_experts.bin")).encode()).hexdigest()
    provenance = {"calibration_split": "train_calibration", "checkpoint_hash": checkpoint_hash,
                  "dataset_manifest_hash": sha256(args.calibration_questions),
                  "composition_config_hash": "29eacd7c25718e085a864a742b3c0d8ab343759b2be958202b11cc846923d0f2",
                  "calibration_seed": 42}
    rms_stats = RMSStatistics(provenance)
    with open(args.calibration_questions, encoding="utf-8") as handle:
        calibration = json.load(handle)[:args.calibration_samples]
    calibration_ids = [str(row["question_id"]) for row in calibration]
    hooks = []
    for layer_name, module in bridge.named_layers:
        def collect(current_module, inputs, base_output, name=layer_name):
            for expert_id in ids:
                delta = bridge.compute_expert_delta(current_module, expert_id, inputs[0])
                rms_stats.update(StatisticKey(expert_id, name, name, type(current_module).__name__),
                                 delta, base_output + delta.to(base_output.dtype), base_output)
        hooks.append(module.register_forward_hook(collect))
    with torch.inference_mode():
        for offset in range(0, len(calibration), args.batch_size):
            raw = _collate(calibration[offset:offset + args.batch_size], bundle, args.images, "cuda:0")
            bundle.model(**_prepare_multimodal_batch(bundle, raw))
    for hook in hooks:
        hook.remove()
    stats_path = output_dir / "rms_statistics.json"
    rms_stats.save_json(stats_path)
    composer = ExpertComposer(bridge, rms_stats, config)

    with open(args.test_questions, encoding="utf-8") as handle:
        records = json.load(handle)[:args.test_samples]
    a_id = int(bundle.tokenizer.encode("A", add_special_tokens=False)[0])
    b_id = int(bundle.tokenizer.encode("B", add_special_tokens=False)[0])
    rows = []
    modes = {"base_only": ((), "base_only"), "single_left": ((ids[0],), "single"),
             "single_right": ((ids[1],), "single"), "direct_sum": (ids, "direct_sum"),
             "rms_calibrated": (ids, "rms_calibrated")}
    performance = {name: [] for name in modes}
    peak_memory = {}
    with torch.inference_mode():
        for offset in range(0, len(records), args.batch_size):
            batch = records[offset:offset + args.batch_size]
            prepared = _prepare_multimodal_batch(bundle, _collate(batch, bundle, args.images, "cuda:0"))
            labels = prepared["labels"]
            positions = [int(torch.where(labels[index].ne(-100))[0][0].item()) - 1 for index in range(len(batch))]
            outputs = {}
            for name, (active, mode) in modes.items():
                if offset == 0:
                    torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                with CompositionRuntime(registry, bridge, composer, active, [], mode):
                    logits = bundle.model(**prepared).logits
                torch.cuda.synchronize()
                performance[name].append((time.perf_counter() - started) / len(batch))
                if offset == 0:
                    peak_memory[name] = int(torch.cuda.max_memory_allocated())
                outputs[name] = logits
            for index, (record, position) in enumerate(zip(batch, positions)):
                target = a_id if record["answer"] == "A" else b_id
                row = {"sample_id": str(record["question_id"]), "target": record["answer"]}
                for name, logits in outputs.items():
                    token = logits[index, position].float()
                    pair = token[[a_id, b_id]]
                    row[name] = {"logit_A": float(pair[0]), "logit_B": float(pair[1]),
                                 "prediction": "A" if pair[0] > pair[1] else "B",
                                 "correct": bool((pair[0] > pair[1]) == (record["answer"] == "A")),
                                 "nll": float(torch.logsumexp(token, 0) - token[target])}
                row["direct_rms_mean_absolute_logit_difference"] = float((outputs["direct_sum"][index].float() - outputs["rms_calibrated"][index].float()).abs().mean())
                rows.append(row)
    with (output_dir / "per_sample.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {"status": "COMPLETED", "pair": list(ids), "checkpoint": args.checkpoint,
               "calibration": {"split": "train_calibration", "seed": 42, "sample_count": len(calibration),
                               "sample_ids": calibration_ids, "sample_ids_sha256": hashlib.sha256("\n".join(calibration_ids).encode()).hexdigest(),
                               "provenance": provenance}, "test_sample_count": len(rows), "modes": {},
               "performance": {}, "stats_bytes": stats_path.stat().st_size,
               "direct_rms_mean_absolute_logit_difference": statistics.fmean(row["direct_rms_mean_absolute_logit_difference"] for row in rows)}
    best_single = [min(row["single_left"]["nll"], row["single_right"]["nll"]) for row in rows]
    for name in modes:
        nlls = [row[name]["nll"] for row in rows]
        synergy = [single - pair for single, pair in zip(best_single, nlls)]
        summary["modes"][name] = {"accuracy": statistics.fmean(float(row[name]["correct"]) for row in rows),
                                  "mean_nll": statistics.fmean(nlls), "mean_synergy": statistics.fmean(synergy),
                                  "median_synergy": statistics.median(synergy),
                                  "positive_synergy_rate": statistics.fmean(float(value > 0) for value in synergy),
                                  "worst_10_percent_synergy": statistics.fmean(sorted(synergy)[:max(1, len(synergy)//10)]),
                                  "pair_harmful_rate": statistics.fmean(float(value < 0) for value in synergy)}
        timings = performance[name][1:] or performance[name]
        summary["performance"][name] = {"mean_seconds_per_sample": statistics.fmean(timings),
                                        "p50_seconds_per_sample": statistics.median(timings),
                                        "p95_seconds_per_sample": percentile(timings, .95),
                                        "throughput_samples_per_second": 1.0 / statistics.fmean(timings),
                                        "peak_cuda_memory_bytes": peak_memory[name]}
    coefficient_rows = []
    for layer_name, _ in bridge.named_layers:
        values = [rms_stats.delta_rms(expert_id, layer_name) for expert_id in ids]
        coefficients, audit = frozen_coefficients(values[0], values[1], config)
        coefficient_rows.append({"layer_name": layer_name, "expert_ids": list(ids), "delta_rms": values,
                                 "coefficients": list(coefficients), **audit})
    summary["coefficients"] = coefficient_rows
    summary["coefficient_distribution"] = {"min": min(value for row in coefficient_rows for value in row["coefficients"]),
                                            "max": max(value for row in coefficient_rows for value in row["coefficients"]),
                                            "mean": statistics.fmean(value for row in coefficient_rows for value in row["coefficients"]),
                                            "epsilon_fallback_layers": sum(bool(row["epsilon_fallback"]) for row in coefficient_rows)}
    direct = summary["modes"]["direct_sum"]
    rms = summary["modes"]["rms_calibrated"]
    summary["direction_consistency"] = {"accuracy_delta": rms["accuracy"] - direct["accuracy"],
                                        "nll_delta": rms["mean_nll"] - direct["mean_nll"],
                                        "consistent": (rms["accuracy"] - direct["accuracy"]) * (rms["mean_nll"] - direct["mean_nll"]) <= 0}
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": summary["status"], "pair": list(ids), "direct": direct, "rms": rms,
                      "coefficient_distribution": summary["coefficient_distribution"]}, sort_keys=True))


if __name__ == "__main__":
    main()
