#!/usr/bin/env python3
"""D1: causal controls — does C really provide function in B+C?

Distinguishes functional complementarity from ensemble/perturbation effects:
  - B+C (real)          : reference
  - B+0.5C / B+0.25C    : scaled C
  - B-C                 : negative C (gate -1)
  - B+randomC           : randomly initialized C, per-layer RMS-matched
  - B+shuffledC         : element-wise shuffled C params, RMS-matched
  - C+B                 : identical math, reversed argument order

All controls are pure forward-time manipulations of the loaded checkpoint
(no training). Per-sample metrics: official VQA accuracy (via greedy
generation), canonical NLL and annotator-weighted marginal NLL (full answer
sequence probability). Per-layer RMS matching uses the BC_calib forward.

Outputs metrics/d1_causal_controls.csv and metrics/d1_causal_controls_summary.json.
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
from compose.eval.compose_p1_real import _collate_real, generate_rows
from compose.eval.load_compose import load_compose_model
from compose.experts import ExpertMetadata, ExpertRegistry
from compose.lora import AdapterBridge
from compose.oracle.evaluator import _prepare_multimodal_batch


def expert_params(bridge, expert_id):
    """Return the LoRA A/B weight tensors of one expert."""
    params = []
    for name, module in bridge.named_layers:
        adapter = module.experts[str(expert_id)]
        params.append((name, adapter))
    return params


def make_replacement(bridge, expert_id, kind, seed):
    """Return (restore_fn, scale_factors_later). Modifies weights in place."""
    rng = torch.Generator(device="cuda").manual_seed(seed)
    saved = []
    for name, adapter in expert_params(bridge, expert_id):
        saved.append((adapter.lora_A.weight, adapter.lora_A.weight.detach().clone()))
        saved.append((adapter.lora_B.weight, adapter.lora_B.weight.detach().clone()))
    with torch.no_grad():
        for name, adapter in expert_params(bridge, expert_id):
            a, b = adapter.lora_A.weight, adapter.lora_B.weight
            if kind == "shuffled":
                flat_a = a.flatten()
                perm_a = torch.randperm(flat_a.numel(), device=a.device, generator=rng)
                a.copy_(flat_a[perm_a].reshape_as(a))
                flat_b = b.flatten()
                perm_b = torch.randperm(flat_b.numel(), device=b.device, generator=rng)
                b.copy_(flat_b[perm_b].reshape_as(b))
            elif kind == "random":
                a.normal_(mean=0.0, std=float(a.std()), generator=rng)
                b.normal_(mean=0.0, std=float(b.std()), generator=rng)
            elif kind == "negate":
                b.mul_(-1.0)  # only B: delta = A@(-B) = -delta
            else:
                raise ValueError(kind)
    def restore():
        for parameter, tensor in saved:
            parameter.copy_(tensor)
    return restore


def collect_delta_rms(bridge, expert_id, records, bundle, image_folder, device):
    """Per-layer delta RMS over the calibration set (before manipulation)."""
    rms = {}
    hooks = []
    for layer_name, module in bridge.named_layers:
        state = [0.0, 0]
        def hook(current_module, inputs, base_output, name=layer_name, state=state):
            delta = bridge.compute_expert_delta(current_module, expert_id, inputs[0])
            value = float(torch.sqrt(torch.mean(delta.float() ** 2)).item())
            state[0] += value
            state[1] += 1
        hooks.append(module.register_forward_hook(hook))
        rms[layer_name] = state
    with torch.inference_mode():
        for offset in range(0, len(records), 4):
            batch = records[offset: offset + 4]
            prepared = _prepare_multimodal_batch(bundle, _collate_real(bundle, batch, image_folder, device))
            bundle.model(**prepared)
    for hook in hooks:
        hook.remove()
    return {name: state[0] / max(1, state[1]) for name, state in rms.items()}


def rescale_to_match(bridge, expert_id, target_rms, current_rms):
    """Scale B weights so each layer's delta RMS matches the target."""
    with torch.no_grad():
        for name, adapter in expert_params(bridge, expert_id):
            target = target_rms.get(name, 1.0)
            current = current_rms.get(name, 1.0)
            if current > 1e-12:
                scale = math.sqrt(max(target, 1e-12) / max(current, 1e-12))
                adapter.lora_B.weight.mul_(scale)


def answer_logprob(bundle, record, variant_answer, image_folder, device):
    variant = dict(record)
    variant["answer"] = variant_answer
    variant["answers"] = [variant_answer]
    prepared = _prepare_multimodal_batch(bundle, _collate_real(bundle, [variant], image_folder, device))
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
    manager = bundle.expert_pool.manager

    calibration = json.loads(Path(args.calibration_questions).read_text(encoding="utf-8"))
    test = json.loads(Path(args.test_questions).read_text(encoding="utf-8"))
    if args.test_samples > 0:
        test = test[: args.test_samples]

    # per-layer delta RMS of the REAL C expert on calibration data
    real_c_rms = collect_delta_rms(bridge, 2, calibration, bundle, args.images, args.device)

    # merged unique answers per test record (weights from 10 annotators)
    merged_answers = []
    for record in test:
        merged = {}
        for answer in [str(a) for a in record["answers"]]:
            key = normalize(answer)
            merged.setdefault(key, {"answer": answer, "weight": 0.0})
            merged[key]["weight"] += 1.0 / len(record["answers"])
        merged_answers.append(sorted(merged.values(), key=lambda v: -v["weight"]))

    def evaluate_configuration(name, gate_b, gate_c, c_transform=None):
        """Run one configuration; returns per-sample metric rows."""
        restore = None
        if c_transform is not None:
            restore = make_replacement(bridge, 2, c_transform, 12345 + args.checkpoint_seed * 7)
            if c_transform in ("shuffled", "random"):
                current = collect_delta_rms(bridge, 2, calibration, bundle, args.images, args.device)
                rescale_to_match(bridge, 2, real_c_rms, current)
        rows = []
        try:
            with torch.inference_mode():
                for record, variants in zip(test, merged_answers):
                    # generation under the current configuration
                    manager.set_default_selection((1, 2), (gate_b, gate_c), normalization="none")
                    prediction = generate_rows(bundle, [record], args.images, args.device, 32)[0]
                    manager.clear_default_selection()
                    answers = [str(a) for a in record["answers"]]
                    score = vqa_accuracy(prediction, answers)
                    correct = score >= 2.0 / 3.0
                    # canonical + marginal NLL
                    canonical_key = normalize(record["answer"])
                    logprobs = []
                    for variant in variants:
                        manager.set_default_selection((1, 2), (gate_b, gate_c), normalization="none")
                        logprobs.append(answer_logprob(bundle, record, variant["answer"], args.images, args.device))
                        manager.clear_default_selection()
                    keys = [normalize(v["answer"]) for v in variants]
                    weights = [v["weight"] for v in variants]
                    canonical_idx = keys.index(canonical_key) if canonical_key in keys else 0
                    canonical_nll = -logprobs[canonical_idx]
                    marginal_nll = -math.log(
                        sum(w * math.exp(lp) for w, lp in zip(weights, logprobs) if lp != float("-inf")) + 1e-12)
                    rows.append({
                        "sample_id": str(record["question_id"]),
                        "configuration": name,
                        "checkpoint_seed": args.checkpoint_seed,
                        "vqa_score": score,
                        "accuracy": int(correct),
                        "canonical_nll": canonical_nll,
                        "marginal_nll": marginal_nll,
                        "prediction": prediction,
                    })
        finally:
            manager.clear_default_selection()
            if restore is not None:
                restore()
        return rows

    configurations = [
        ("B+C_real", 1.0, 1.0 / math.sqrt(2.0), None),
        ("C+B_real", 1.0, 1.0 / math.sqrt(2.0), None),  # same math as B+C
        ("B+0.5C", 1.0, 0.5, None),
        ("B+0.25C", 1.0, 0.25, None),
        ("B-C", 1.0, 1.0 / math.sqrt(2.0), "negate"),
        ("B+randomC", 1.0, 1.0 / math.sqrt(2.0), "random"),
        ("B+shuffledC", 1.0, 1.0 / math.sqrt(2.0), "shuffled"),
    ]
    all_rows = []
    for name, gate_b, gate_c, transform in configurations:
        print("evaluating {}".format(name), flush=True)
        all_rows.extend(evaluate_configuration(name, gate_b, gate_c, transform))

    with (output_root / "metrics" / "d1_causal_controls_seed{}.csv".format(args.checkpoint_seed)).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {}
    for name in [c[0] for c in configurations]:
        subset = [r for r in all_rows if r["configuration"] == name]
        summary[name] = {
            "accuracy": statistics.fmean(r["accuracy"] for r in subset),
            "mean_vqa_score": statistics.fmean(r["vqa_score"] for r in subset),
            "canonical_nll": statistics.fmean(r["canonical_nll"] for r in subset),
            "marginal_nll": statistics.fmean(r["marginal_nll"] for r in subset),
        }
    (output_root / "metrics" / "d1_causal_controls_summary_seed{}.json".format(args.checkpoint_seed)).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
