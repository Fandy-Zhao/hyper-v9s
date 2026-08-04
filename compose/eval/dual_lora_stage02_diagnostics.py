"""Stage 02: per-layer, per-module conflict diagnostics for one expert pair.

One pair forward (formal gates 1,1) over the test set, capturing for every
ComposeLinear layer and every sample:

  - ||u_A^l||, ||u_B^l||            per-expert LoRA delta norms
  - cos(u_A^l, u_B^l)               delta direction agreement
  - ||u_A^l + u_B^l||               combined delta norm
  - cancellation ratio              ||u_A+u_B|| / (||u_A||+||u_B||)
  - dominance ratio                 max(||u_A||,||u_B||) / min(||u_A||,||u_B||)

plus the per-sample answer-token logits/NLL/accuracy of the pair. Sample sets
for stratification (gained / lost / worst-10% synergy / cross-seed repeated
failures) are resolved from the recorded per-sample evaluations of seeds
42/43/44, so the diagnostics can reuse them without new runs.

Outputs (under --output-root):
  layer_metrics_{pair}.parquet     per-sample per-layer metrics
  failure_samples_{pair}.jsonl     per-sample group assignments + pair metrics

Usage:
  python -m compose.eval.dual_lora_stage02_diagnostics --pair a_independent_b \
      --device cuda:4 --output-root artifacts/dual_lora_stage02
"""

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import torch

from compose.adapters.lora import ComposeLinear
from compose.eval.load_compose import load_compose_model
from compose.oracle.evaluator import _collate, _prepare_multimodal_batch

BATCH_SIZE = 8

PAIRS = {
    "a_independent_b": ("assembled/seed42/a_independent_b", "A_plus_B", ("expert_a", "independent_b"), (0, 1)),
    "a_residual_b": ("seed42/residual_b", "A_plus_B", ("expert_a", "residual_b"), (0, 1)),
    "independent_b_c": ("assembled/seed42/independent_b_c", "B_plus_C", ("independent_b", "expert_c"), (1, 2)),
    "residual_b_c": ("assembled/seed42/residual_b_c", "B_plus_C", ("residual_b", "expert_c"), (1, 2)),
}
CHECKPOINT_ROOT = "/data/ckpt/zhaozhuofan/compose/format_controlled_composition_v1"
DATA_ROOT = "experiments/data/controlled_format_v1_training/instructions"
EVAL_ROOT = Path("experiments/runs/format_controlled_composition_v1/evaluation")


def read_recorded(dataset: str, name: str, seed: int) -> Dict[str, dict]:
    rows = {}
    path = EVAL_ROOT / "seed{}".format(seed) / dataset / name / "per_sample.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["sample_id"])] = row
    return rows


def resolve_sample_groups(pair: str, dataset: str, singles: Sequence[str]) -> Dict[str, Dict[str, dict]]:
    """Assign every test sample to (gained, lost, both_ok, both_wrong, worst10, cross_seed_fail).

    gained      : pair correct in seed 42 but both singles wrong in seed 42
    lost        : pair wrong in seed 42 but at least one single correct in seed 42
    both_ok     : pair correct and a single correct
    both_wrong  : pair wrong and both singles wrong
    worst10     : bottom decile of seed-42 per-sample synergy
    cross_seed_fail : pair wrong in ALL THREE seeds 42/43/44
    """
    pair_42 = read_recorded(dataset, pair, 42)
    singles_42 = {name: read_recorded(dataset, name, 42) for name in singles}
    groups: Dict[str, Dict[str, dict]] = {name: {} for name in
                                           ("gained", "lost", "both_ok", "both_wrong", "worst10", "cross_seed_fail")}
    synergies = []
    for sid in sorted(pair_42):
        pair_correct = bool(pair_42[sid]["correct"])
        single_correct = any(bool(singles_42[name][sid]["correct"]) for name in singles)
        best_single_nll = min(float(singles_42[name][sid]["answer_token_nll"]) for name in singles)
        synergy = best_single_nll - float(pair_42[sid]["answer_token_nll"])
        synergies.append((sid, synergy))
        if pair_correct and not single_correct:
            groups["gained"][sid] = pair_42[sid]
        elif not pair_correct and single_correct:
            groups["lost"][sid] = pair_42[sid]
        elif pair_correct and single_correct:
            groups["both_ok"][sid] = pair_42[sid]
        else:
            groups["both_wrong"][sid] = pair_42[sid]
    worst10_ids = {sid for sid, _ in sorted(synergies, key=lambda item: item[1])[: max(1, len(synergies) // 10)]}
    groups["worst10"] = {sid: pair_42[sid] for sid in worst10_ids}
    fail_sets = []
    for seed in (42, 43, 44):
        rows = read_recorded(dataset, pair, seed)
        fail_sets.append({sid for sid, row in rows.items() if not row["correct"]})
    cross = set.intersection(*fail_sets)
    groups["cross_seed_fail"] = {sid: pair_42[sid] for sid in sorted(cross)}
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True, choices=sorted(PAIRS))
    parser.add_argument("--model-path", default="/data/ckpt/zhaozhuofan/models/llava-v1.5-7b")
    parser.add_argument("--vision-tower", default="/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336")
    parser.add_argument("--projector-path", default="/data/ckpt/zhaozhuofan/models/llava-v1.5-7b/mm_projector.bin")
    parser.add_argument("--checkpoint-root", default=CHECKPOINT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="artifacts/dual_lora_stage02")
    args = parser.parse_args()

    checkpoint_name, dataset, singles, pair_ids = PAIRS[args.pair]
    checkpoint_dir = os.path.join(args.checkpoint_root, checkpoint_name)
    image_folder = "experiments/data/controlled_format_v1"
    with open(os.path.join(DATA_ROOT, dataset, "test_eval.json"), encoding="utf-8") as handle:
        records = json.load(handle)

    bundle = load_compose_model(
        model_path=args.model_path, checkpoint_dir=checkpoint_dir,
        vision_tower=args.vision_tower, projector_path=args.projector_path,
        expert_id=None, device=args.device, dtype=torch.bfloat16, model_max_length=2048,
    )
    a_id = int(bundle.tokenizer.encode("A", add_special_tokens=False)[0])
    b_id = int(bundle.tokenizer.encode("B", add_special_tokens=False)[0])
    bundle.expert_pool.manager.set_default_selection(list(pair_ids), [1.0, 1.0], normalization="none")

    layer_by_name: Dict[str, ComposeLinear] = {
        name: module for name, module in bundle.model.named_modules()
        if isinstance(module, ComposeLinear)
    }
    metric_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    current_batch: Dict[str, List[str]] = {}
    hooks = []

    # Per-layer metrics are computed inside the forward hook so only one
    # layer's hidden states are alive at a time (keeping all 224 captured
    # inputs simultaneously exceeds GPU memory).
    def make_post_hook(name: str, module: ComposeLinear):
        def post_hook(module, inputs, output):
            hidden = inputs[0]
            da = module.experts[str(pair_ids[0])](hidden).to(torch.float32)
            db = module.experts[str(pair_ids[1])](hidden).to(torch.float32)
            # per-sample aggregation over tokens: RMS of per-token norms,
            # mean cosine over tokens with nonzero denominators
            norm_a = da.norm(dim=-1)          # (batch, seq)
            norm_b = db.norm(dim=-1)
            dot = (da * db).sum(dim=-1)
            denom = (norm_a * norm_b).clamp_min(1e-12)
            cosine = dot / denom
            norm_sum = (da + db).norm(dim=-1)
            rms = lambda x: x.square().mean(dim=-1).sqrt()  # noqa: E731
            rms_a, rms_b = rms(norm_a), rms(norm_b)
            rms_ab = rms(norm_sum)
            cosine_mean = cosine.mean(dim=-1)
            ids = current_batch.get("ids", [])
            for index, sid in enumerate(ids):
                a = float(rms_a[index].item())
                b = float(rms_b[index].item())
                metric_rows.append({
                    "sample_id": sid, "layer": name,
                    "norm_A": a, "norm_B": b,
                    "cosine": float(cosine_mean[index].item()),
                    "norm_A_plus_B": float(rms_ab[index].item()),
                    "cancellation_ratio": float(rms_ab[index].item() / (a + b)) if a + b > 0 else 0.0,
                    "dominance_ratio": float(max(a, b) / min(a, b)) if min(a, b) > 0 else 0.0,
                })
            return output
        return post_hook

    for name, module in layer_by_name.items():
        hooks.append(module.register_forward_hook(make_post_hook(name, module)))

    try:
        with torch.inference_mode():
            for offset in range(0, len(records), BATCH_SIZE):
                batch_records = records[offset: offset + BATCH_SIZE]
                current_batch["ids"] = [str(r["question_id"]) for r in batch_records]
                raw = _collate(batch_records, bundle, image_folder, args.device)
                prepared = _prepare_multimodal_batch(bundle, raw)
                logits = bundle.model(**prepared).logits
                labels = prepared["labels"]
                for index, record in enumerate(batch_records):
                    label_row = labels[index].tolist()
                    positions = [i for i, value in enumerate(label_row) if value != -100]
                    answer_pos = positions[0] - 1
                    token_logits = logits[index, answer_pos].float()
                    target = str(record["answer"])
                    target_id = a_id if target == "A" else b_id
                    nll = float(torch.logsumexp(token_logits, dim=0) - token_logits[target_id])
                    logit_a = float(token_logits[a_id])
                    logit_b = float(token_logits[b_id])
                    correct = (logit_a > logit_b) == (target == "A")
                    sample_rows.append({"sample_id": str(record["question_id"]),
                                        "logit_A": logit_a, "logit_B": logit_b,
                                        "nll": nll, "correct": bool(correct),
                                        "target": target, "task": str(record.get("task_id", ""))})
    finally:
        for hook in hooks:
            hook.remove()

    groups = resolve_sample_groups(args.pair, dataset, singles)
    group_by_sample = {}
    for group_name, members in groups.items():
        for sid in members:
            group_by_sample.setdefault(sid, []).append(group_name)
    for row in sample_rows:
        row["groups"] = group_by_sample.get(row["sample_id"], [])

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    layer_frame = pd.DataFrame(metric_rows)
    layer_frame.to_parquet(output_root / "layer_metrics_{}.parquet".format(args.pair), index=False)
    with (output_root / "failure_samples_{}.jsonl".format(args.pair)).open("w", encoding="utf-8") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "pair": args.pair, "dataset": dataset, "samples": len(sample_rows),
        "layers": len(layer_by_name),
        "group_sizes": {name: len(members) for name, members in groups.items()},
        "pair_accuracy": statistics.fmean(float(r["correct"]) for r in sample_rows),
        "mean_nll": statistics.fmean(float(r["nll"]) for r in sample_rows),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
