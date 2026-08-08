#!/usr/bin/env python3
"""D0: metric-mismatch audit — multi-answer NLL vs canonical NLL.

For every mode (base, B, C, c0, c1, c2, c3) on the B+C test pool, recompute
per sample:
  - canonical NLL     : -log P(majority raw answer)        (D0.1, old metric)
  - min NLL           : min over unique normalized answers (D0.2, diagnostic)
  - marginal NLL      : -log sum_a w_a P(a)                (D0.3, primary)
  - expected VQA util : sum over candidates w_a*P(a)*utility(a) (D0.4)

Answer sequence probability is the FULL product over answer tokens
(teacher-forced), never first-token only. Answers are normalized with the
official VQA2.0 normalization and merged by normalized form; weights are the
annotator frequencies among the 10 human answers.

Outputs metrics/d0_metric_audit.csv (per sample, per mode) and
metrics/d0_metric_audit_summary.json.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import torch

from llava.constants import IGNORE_INDEX

from compose.data.real_p1.official_metric import normalize, vqa_accuracy
from compose.eval.compose_p1_real import _collate_real, selection
from compose.eval.load_compose import load_compose_model
from compose.experts import ExpertMetadata, ExpertRegistry
from compose.lora import (
    AdapterBridge,
    CompositionRuntime,
    ExpertComposer,
    RMSCompositionConfig,
    RMSStatistics,
    StatisticKey,
)
from compose.oracle.evaluator import _prepare_multimodal_batch


def make_variants(record):
    """Unique normalized answers with annotator weights (10 answers)."""
    answers = [str(a) for a in record["answers"]]
    merged = {}
    for answer in answers:
        key = normalize(answer)
        merged.setdefault(key, {"answer": answer, "weight": 0.0, "count": 0})
        merged[key]["weight"] += 1.0 / len(answers)
        merged[key]["count"] += 1
    variants = []
    for key, info in merged.items():
        variant = dict(record)
        variant["answer"] = info["answer"]
        variant["answers"] = [info["answer"]]
        variant["_n_key"] = key
        variant["_n_weight"] = info["weight"]
        variants.append(variant)
    variants.sort(key=lambda v: -v["_n_weight"])
    return variants


def variant_sequence_logprob(bundle, variant, image_folder, device):
    """Full-sequence teacher-forced log-prob of the answer tokens."""
    prepared = _prepare_multimodal_batch(
        bundle, _collate_real(bundle, [variant], image_folder, device))
    with torch.inference_mode():
        logits = bundle.model(**prepared).logits
    labels = prepared["labels"][0]
    supervised = torch.where(labels.ne(IGNORE_INDEX))[0]
    if not supervised.numel():
        return float("-inf")
    positions = supervised.tolist()
    gold = labels[supervised].tolist()
    predicting = torch.tensor([p - 1 for p in positions], device=logits.device)
    token_logprobs = torch.log_softmax(logits[0, predicting].float(), dim=-1)
    return float(token_logprobs[torch.arange(len(gold)), gold].sum().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-questions", required=True)
    parser.add_argument("--calibration-questions", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--checkpoint-seed", type=int, required=True)
    parser.add_argument("--analysis-seed", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode-filter", default="base,single_left,single_right,c0,c1,c2,c3")
    parser.add_argument("--test-samples", type=int, default=0)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.analysis_seed)
    torch.cuda.manual_seed_all(args.analysis_seed)
    bundle = load_compose_model(
        model_path=args.model_path,
        checkpoint_dir=args.checkpoint,
        vision_tower=args.vision_tower,
        projector_path=str(Path(args.model_path) / "mm_projector.bin"),
        expert_id=None,
        device=args.device,
        dtype=torch.bfloat16,
        model_max_length=2048,
    )
    bridge = AdapterBridge(bundle.model, verify_ddp=False)
    registry = ExpertRegistry()
    for expert_id in bridge.expert_ids:
        registry.register(ExpertMetadata(expert_id=expert_id, adapter_name=str(expert_id)))
    ids = (1, 2)

    # RMS statistics from the calibration set (for c2/c3)
    rms_stats = RMSStatistics({
        "calibration_split": "BC_calib", "checkpoint_hash": "d0",
        "dataset_manifest_hash": "d0", "composition_config_hash": "d0",
    })
    calibration = json.loads(Path(args.calibration_questions).read_text(encoding="utf-8"))
    hooks = []
    for layer_name, module in bridge.named_layers:
        def collect(current_module, inputs, base_output, name=layer_name):
            for expert_id in ids:
                delta = bridge.compute_expert_delta(current_module, expert_id, inputs[0])
                rms_stats.update(StatisticKey(expert_id, name, name, type(current_module).__name__),
                                 delta, base_output + delta.to(base_output.dtype), base_output)
        hooks.append(module.register_forward_hook(collect))
    with torch.inference_mode():
        for offset in range(0, len(calibration), 4):
            records = calibration[offset: offset + 4]
            prepared = _prepare_multimodal_batch(bundle, _collate_real(bundle, records, args.images, args.device))
            bundle.model(**prepared)
    for hook in hooks:
        hook.remove()

    test = json.loads(Path(args.test_questions).read_text(encoding="utf-8"))
    if args.test_samples > 0:
        test = test[: args.test_samples]
    modes = tuple(args.mode_filter.split(","))
    variant_lists = [make_variants(record) for record in test]

    rows = []
    for mode in modes:
        for record, variants in zip(test, variant_lists):
            logprobs = []
            for variant in variants:
                with torch.inference_mode():
                    if mode in ("c2", "c3"):
                        composer = ExpertComposer(bridge, rms_stats, RMSCompositionConfig(expert_scalars=(1.0, 1.0)))
                        with CompositionRuntime(registry, bridge, composer, ids, (), "rms_calibrated"):
                            lp = variant_sequence_logprob(bundle, variant, args.images, args.device)
                    else:
                        with selection(bundle, registry, bridge, rms_stats, ids, mode):
                            lp = variant_sequence_logprob(bundle, variant, args.images, args.device)
                logprobs.append(lp)
            rows.append(compute_record_metrics(record, variants, logprobs, mode,
                                               args.checkpoint_seed, args.analysis_seed))

    fields = ["sample_id", "checkpoint_seed", "analysis_seed", "mode",
              "n_answers", "canonical_nll", "min_nll", "marginal_nll",
              "expected_vqa_utility", "canonical_is_majority_norm"]
    with (output_root / "metrics" / "d0_metric_audit_seed{}.csv".format(args.checkpoint_seed)).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (output_root / "metrics" / "d0_metric_audit_summary_seed{}.json".format(args.checkpoint_seed)).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("wrote metrics/d0_metric_audit.csv")
    print(json.dumps(summary, indent=2, sort_keys=True))


def compute_record_metrics(record, variants, logprobs, mode, cs, aseed):
    keys = [v["_n_key"] for v in variants]
    weights = [v["_n_weight"] for v in variants]
    canonical_key = normalize(record["answer"])
    canonical_idx = keys.index(canonical_key) if canonical_key in keys else 0
    canonical_nll = -logprobs[canonical_idx] if logprobs[canonical_idx] != float("-inf") else float("inf")
    min_nll = -max(logprobs)
    weighted = sum(w * math.exp(lp) for w, lp in zip(weights, logprobs) if lp != float("-inf"))
    marginal_nll = -math.log(weighted + 1e-12)
    probs = [math.exp(lp) if lp != float("-inf") else 0.0 for lp in logprobs]
    utilities = [vqa_accuracy(v["answer"], record["answers"]) for v in variants]
    expected_utility = sum(w * p * u for w, p, u in zip(weights, probs, utilities))
    return {
        "sample_id": str(record["question_id"]),
        "checkpoint_seed": cs,
        "analysis_seed": aseed,
        "mode": mode,
        "n_answers": len(variants),
        "canonical_nll": canonical_nll,
        "min_nll": min_nll,
        "marginal_nll": marginal_nll,
        "expected_vqa_utility": expected_utility,
        "canonical_is_majority_norm": float(canonical_idx == 0),
    }


def summarize(rows):
    summary = {}
    for mode in sorted({r["mode"] for r in rows}):
        subset = [r for r in rows if r["mode"] == mode]
        summary[mode] = {
            "n": len(subset),
            "canonical_nll_mean": statistics.fmean(r["canonical_nll"] for r in subset),
            "min_nll_mean": statistics.fmean(r["min_nll"] for r in subset),
            "marginal_nll_mean": statistics.fmean(r["marginal_nll"] for r in subset),
            "expected_vqa_utility_mean": statistics.fmean(r["expected_vqa_utility"] for r in subset),
            "canonical_is_majority_norm_rate": statistics.fmean(r["canonical_is_majority_norm"] for r in subset),
        }
    return summary


if __name__ == "__main__":
    main()
