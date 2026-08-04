"""Stage 03: expert function purity and context-dependency diagnostics.

Part 1 — conditional contributions (from recorded per-sample evaluations):
  per (expert, context) the conditional marginal gain
    G_X|Y = mean_s ( NLL_Y(s) - NLL_{X+Y}(s) )
  computed on every recorded context (base / A / B / C / pair datasets), and
  the per-layer delta-direction agreement across contexts (cosine of the
  delta vector between context pairs) via forward-time hooks.

Part 2 — low-dimensional gradient signatures:
  for one expert (residual_b or independent_b), capture the per-layer LoRA
  deltas on B_only / A_plus_B / B_plus_C test sets, project every layer delta
  onto a fixed random subspace (seeded), and report
    cos(g_B_only, g_B_given_A), cos(g_B_only, g_B_given_C),
    cos(g_B_given_A, g_B_given_C)
  at global and per-layer level. Context-conditioned experts (counterweights)
  show near-identical signatures everywhere; transferable function experts
  show task-consistent signatures on their own function and different ones
  elsewhere.

Part 3 — answer-logit margin analysis of lost samples:
  for the samples where the pair is wrong but a single is right, compare the
  margin (logit_best - logit_other) of the pair vs the best single.

Outputs: artifacts/dual_lora_stage03/conditional_contributions.parquet,
gradient_signatures.pt, margin_analysis.json
"""

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import torch

from compose.adapters.lora import ComposeLinear
from compose.eval.load_compose import load_compose_model
from compose.oracle.evaluator import _collate, _prepare_multimodal_batch

BATCH_SIZE = 8

# context datasets -> recorded models available on that test set
CONTEXT_DATASETS = ("A_only", "B_only", "C_only", "A_plus_B", "B_plus_C")
EXPERT_MODELS = ("expert_a", "independent_b", "expert_c", "residual_b")
# (expert X, context Y) -> dataset where X+Y is measurable
CONTEXT_PAIRS = [
    ("independent_b", "base", "B_only"), ("residual_b", "base", "B_only"),
    ("expert_a", "base", "A_only"), ("expert_c", "base", "C_only"),
    ("expert_a", "independent_b", "A_plus_B"), ("expert_a", "residual_b", "A_plus_B"),
    ("independent_b", "expert_a", "A_plus_B"), ("residual_b", "expert_a", "A_plus_B"),
    ("independent_b", "expert_c", "B_plus_C"), ("residual_b", "expert_c", "B_plus_C"),
    ("expert_c", "independent_b", "B_plus_C"), ("expert_c", "residual_b", "B_plus_C"),
    ("expert_a", "base", "A_plus_B"), ("independent_b", "base", "A_plus_B"),
    ("residual_b", "base", "A_plus_B"), ("expert_c", "base", "B_plus_C"),
    ("independent_b", "base", "B_plus_C"), ("residual_b", "base", "B_plus_C"),
]

EVAL_ROOT = Path("experiments/runs/format_controlled_composition_v1/evaluation/seed42")
CHECKPOINT_ROOT = "/data/ckpt/zhaozhuofan/compose/format_controlled_composition_v1"
DATA_ROOT = "experiments/data/controlled_format_v1_training/instructions"

PAIR_CHECKPOINTS = {
    "a_independent_b": ("a_independent_b", "A_plus_B"),
    "a_residual_b": ("residual_b", "A_plus_B"),
    "independent_b_c": ("independent_b_c", "B_plus_C"),
    "residual_b_c": ("residual_b_c", "B_plus_C"),
}
SINGLE_NAMES = {"expert_a": "expert_a", "independent_b": "independent_b",
                "expert_c": "expert_c", "residual_b": "residual_b"}
# (expert X, context Y, dataset) -> recorded pair model containing X and Y
PAIR_MODEL_FOR = {
    ("expert_a", "independent_b", "A_plus_B"): "a_independent_b",
    ("independent_b", "expert_a", "A_plus_B"): "a_independent_b",
    ("expert_a", "residual_b", "A_plus_B"): "a_residual_b",
    ("residual_b", "expert_a", "A_plus_B"): "a_residual_b",
    ("independent_b", "expert_c", "B_plus_C"): "independent_b_c",
    ("expert_c", "independent_b", "B_plus_C"): "independent_b_c",
    ("residual_b", "expert_c", "B_plus_C"): "residual_b_c",
    ("expert_c", "residual_b", "B_plus_C"): "residual_b_c",
}


def read_rows(dataset: str, model: str) -> Dict[str, dict]:
    rows = {}
    path = EVAL_ROOT / dataset / model / "per_sample.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["sample_id"])] = row
    return rows


def conditional_contributions() -> List[Dict[str, Any]]:
    rows = []
    for expert, context, dataset in CONTEXT_PAIRS:
        expert_name = SINGLE_NAMES[expert]
        context_rows = read_rows(dataset, "base" if context == "base" else SINGLE_NAMES[context])
        expert_rows = read_rows(dataset, expert_name)
        if context == "base":
            pair_rows = expert_rows  # G_X|base = NLL_base - NLL_X
        else:
            pair_name = PAIR_MODEL_FOR[(expert, context, dataset)]
            pair_rows = read_rows(dataset, pair_name)
        if set(context_rows) != set(expert_rows) or set(context_rows) != set(pair_rows):
            raise ValueError("sample sets differ for {}|{} on {}".format(expert, context, dataset))
        gains = [float(context_rows[sid]["answer_token_nll"]) - float(pair_rows[sid]["answer_token_nll"])
                 for sid in sorted(context_rows)]
        rows.append({
            "expert": expert, "context": context, "dataset": dataset,
            "mean_gain": statistics.fmean(gains), "median_gain": statistics.median(gains),
            "positive_rate": statistics.fmean(float(v > 0) for v in gains),
            "worst10_mean_gain": statistics.fmean(sorted(gains)[: max(1, len(gains) // 10)]),
            "samples": len(gains),
        })
    return rows


def capture_deltas(bundle, records: Sequence[dict], image_folder: str,
                   device: str) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    layer_by_name: Dict[str, ComposeLinear] = {
        name: module for name, module in bundle.model.named_modules()
        if isinstance(module, ComposeLinear)
    }
    captured: Dict[str, torch.Tensor] = {}
    hooks = []

    def make_pre_hook(name: str):
        def pre_hook(module, inputs):
            captured[name] = inputs[0].detach().to(torch.float32)
        return pre_hook

    for name, module in layer_by_name.items():
        hooks.append(module.register_forward_pre_hook(make_pre_hook(name)))
    bundle.expert_pool.manager.set_default_selection([1], [1.0], normalization="none")
    sample_rows: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for offset in range(0, len(records), BATCH_SIZE):
            batch_records = records[offset: offset + BATCH_SIZE]
            raw = _collate(batch_records, bundle, image_folder, device)
            prepared = _prepare_multimodal_batch(bundle, raw)
            bundle.model(**prepared)
    deltas: Dict[str, torch.Tensor] = {}
    for name, module in layer_by_name.items():
        hidden = captured[name]
        # sum over batch tokens: one direction per layer per dataset
        delta = module.experts["1"](hidden)
        deltas[name] = delta.detach().to(torch.float32)
    for hook in hooks:
        hook.remove()
    return deltas, sample_rows


def signature_cosines(bundle, datasets: Sequence[str], image_folder: str,
                      device: str, output_root: Path) -> Dict[str, Any]:
    """Project per-layer deltas to a fixed random subspace and compute
    context-pair cosines of expert-1's contribution."""
    rng = torch.Generator().manual_seed(0)
    signatures: Dict[str, Dict[str, torch.Tensor]] = {}
    layer_names = None
    for dataset in datasets:
        with open(os.path.join(DATA_ROOT, dataset, "test_eval.json"), encoding="utf-8") as handle:
            records = json.load(handle)
        deltas, _ = capture_deltas(bundle, records, image_folder, device)
        if layer_names is None:
            layer_names = sorted(deltas)
        flat = torch.cat([deltas[name].reshape(-1) for name in layer_names])
        # random subspace projection (fixed seed)
        subspace = torch.randn(flat.numel(), 256, generator=rng, dtype=torch.float32)
        subspace = subspace / subspace.norm(dim=0, keepdim=True)
        signatures[dataset] = {"flat": flat, "projected": flat @ subspace,
                               "per_layer": {name: deltas[name].reshape(-1) for name in layer_names}}
    torch.save(signatures, str(output_root / "gradient_signatures.pt"))
    result = {"dataset_pairs": {}}
    for pair in (("B_only", "A_plus_B"), ("B_only", "B_plus_C"), ("A_plus_B", "B_plus_C")):
        left, right = pair
        c = torch.nn.functional.cosine_similarity(
            signatures[left]["projected"], signatures[right]["projected"], dim=0)
        result["dataset_pairs"]["{}_{}".format(*pair)] = {
            "global_cosine": float(torch.dot(signatures[left]["flat"], signatures[right]["flat"]) /
                                   (signatures[left]["flat"].norm() * signatures[right]["flat"].norm()).item()),
            "projected_cosine": float(c.mean().item()),
            "per_layer": {name: float(torch.nn.functional.cosine_similarity(
                signatures[left]["per_layer"][name], signatures[right]["per_layer"][name], dim=0).item())
                for name in layer_names},
        }
    return result


def margin_analysis() -> Dict[str, Any]:
    """For lost samples (pair wrong, best single right), pair margin vs best-single margin."""
    result = {}
    for pair_name, (checkpoint, dataset) in PAIR_CHECKPOINTS.items():
        pair_rows = read_rows(dataset, pair_name)
        singles = {"a_independent_b": ("expert_a", "independent_b"),
                   "a_residual_b": ("expert_a", "residual_b"),
                   "independent_b_c": ("independent_b", "expert_c"),
                   "residual_b_c": ("residual_b", "expert_c")}[pair_name]
        s_a, s_b = (read_rows(dataset, name) for name in singles)
        lost_margins_pair = []
        lost_margins_single = []
        for sid in sorted(pair_rows):
            if pair_rows[sid]["correct"]:
                continue
            best = s_a[sid] if float(s_a[sid]["answer_token_nll"]) <= float(s_b[sid]["answer_token_nll"]) else s_b[sid]
            if not best["correct"]:
                continue
            pair_margin = abs(float(pair_rows[sid]["logit_A"]) - float(pair_rows[sid]["logit_B"]))
            best_margin = abs(float(best["logit_A"]) - float(best["logit_B"]))
            lost_margins_pair.append(pair_margin)
            lost_margins_single.append(best_margin)
        result[pair_name] = {
            "lost_sample_count": len(lost_margins_pair),
            "mean_pair_margin": statistics.fmean(lost_margins_pair) if lost_margins_pair else None,
            "mean_best_single_margin": statistics.fmean(lost_margins_single) if lost_margins_single else None,
            "pair_margin_le_single_rate": statistics.fmean(
                float(p <= s) for p, s in zip(lost_margins_pair, lost_margins_single)) if lost_margins_pair else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/data/ckpt/zhaozhuofan/models/llava-v1.5-7b")
    parser.add_argument("--vision-tower", default="/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336")
    parser.add_argument("--projector-path", default="/data/ckpt/zhaozhuofan/models/llava-v1.5-7b/mm_projector.bin")
    parser.add_argument("--checkpoint", required=True, help="expert checkpoint (independent_b or residual_b)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="artifacts/dual_lora_stage03")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    contrib = conditional_contributions()
    pd.DataFrame(contrib).to_parquet(output_root / "conditional_contributions.parquet", index=False)

    bundle = load_compose_model(
        model_path=args.model_path, checkpoint_dir=args.checkpoint,
        vision_tower=args.vision_tower, projector_path=args.projector_path,
        expert_id=None, device=args.device, dtype=torch.bfloat16, model_max_length=2048,
    )
    signatures = signature_cosines(bundle, ("B_only", "A_plus_B", "B_plus_C"),
                                   "experiments/data/controlled_format_v1", args.device, output_root)
    margins = margin_analysis()
    report = {
        "checkpoint": args.checkpoint,
        "conditional_contribution_rows": len(contrib),
        "signatures": signatures,
        "margin_analysis": margins,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    with (output_root / "gradient_signatures.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
