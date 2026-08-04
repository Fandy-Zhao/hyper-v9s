"""Stage 02: module/layer-scope ablations of the dual-expert combination.

Runs the pair forward with per-layer, per-expert gate selections so only a
chosen subset of target modules / layers carries the two LoRA deltas. No
training; all manipulations are forward-time gate masks.

Module scopes (all layers):
  attn_all (q,k,v,o), qv, qk, vo, ffn (gate,up,down), gate, updown,
  split_attn_ffn (expert1 on attention, expert2 on FFN),
  split_ffn_attn (expert2 on attention, expert1 on FFN),
  full (all modules)

Layer scopes (all modules):
  low (first 1/3), mid, high (last 1/3), x1_lowmid_x2_midhigh,
  x2_lowmid_x1_midhigh, pos_cos (layers with mean cosine > 0),
  neg_cos (layers with mean cosine <= 0, negative control)

Outputs: artifacts/dual_lora_stage02/ablation_{pair}.parquet with per-scope
aggregate metrics (accuracy, NLL, Brier, ECE, synergy, marginals) and
per-sample rows.

Usage:
  python -m compose.eval.dual_lora_stage02_ablation --pair a_independent_b \
      --device cuda:4 --output-root artifacts/dual_lora_stage02
"""

import argparse
import json
import math
import os
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch

from compose.adapters.lora import ComposeLinear
from compose.eval.load_compose import load_compose_model
from compose.oracle.evaluator import _collate, _prepare_multimodal_batch

BATCH_SIZE = 8
LAYER_COUNT = 32
PAIR_SCALE = 1.0 / math.sqrt(2.0)

PAIRS = {
    "a_independent_b": ("assembled/seed42/a_independent_b", "A_plus_B", ("expert_a", "independent_b"), (0, 1)),
    "a_residual_b": ("seed42/residual_b", "A_plus_B", ("expert_a", "residual_b"), (0, 1)),
    "independent_b_c": ("assembled/seed42/independent_b_c", "B_plus_C", ("independent_b", "expert_c"), (1, 2)),
    "residual_b_c": ("assembled/seed42/residual_b_c", "B_plus_C", ("residual_b", "expert_c"), (1, 2)),
}
CHECKPOINT_ROOT = "/data/ckpt/zhaozhuofan/compose/format_controlled_composition_v1"
DATA_ROOT = "experiments/data/controlled_format_v1_training/instructions"
EVAL_ROOT = Path("experiments/runs/format_controlled_composition_v1/evaluation/seed42")

MODULE_GROUPS = {
    "attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "ffn": ("gate_proj", "up_proj", "down_proj"),
}

MODULE_SCOPES: Dict[str, Optional[Tuple[Sequence[str], Optional[str]]]] = {
    # scope -> (modules to keep, optional split: which expert takes each group)
    "attn_all": (MODULE_GROUPS["attn"], None),
    "qv": (("q_proj", "v_proj"), None),
    "qk": (("q_proj", "k_proj"), None),
    "vo": (("v_proj", "o_proj"), None),
    "ffn": (MODULE_GROUPS["ffn"], None),
    "gate": (("gate_proj",), None),
    "updown": (("up_proj", "down_proj"), None),
    "split_attn_ffn": (None, ("attn", "ffn")),   # expert1 on attn, expert2 on ffn
    "split_ffn_attn": (None, ("ffn", "attn")),   # expert1 on ffn, expert2 on attn
    "full": (None, None),
}

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
_MODULE_RE = re.compile(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")


def layer_index(name: str) -> int:
    match = _LAYER_RE.match(name)
    if not match:
        raise ValueError("unexpected layer name: {}".format(name))
    return int(match.group(1))


def module_type(name: str) -> str:
    match = _MODULE_RE.search(name)
    if not match:
        raise ValueError("unexpected module name: {}".format(name))
    return match.group(1)


def layer_ranges() -> Dict[str, Tuple[int, int]]:
    third = LAYER_COUNT // 3
    return {"low": (0, third), "mid": (third, 2 * third), "high": (2 * third, LAYER_COUNT)}


def selection_for(name: str, scope: str, cos_sets: Optional[Dict[str, set]] = None) -> Optional[Tuple[Sequence[int], Sequence[float]]]:
    """Return (expert_ids, gates) for one ComposeLinear under one scope, or
    None for excluded layers (cleared selection = base only). Slot ids 0/1
    denote the first/second expert of the pair."""
    layer = layer_index(name)
    module = module_type(name)
    ranges = layer_ranges()

    if scope in MODULE_SCOPES:
        modules, split = MODULE_SCOPES[scope]
        if split is None:
            if modules is None or module in modules:
                return [0, 1], [1.0, 1.0]
            return None
        group_for = {"attn": MODULE_GROUPS["attn"], "ffn": MODULE_GROUPS["ffn"]}
        expert1_groups, expert2_groups = split
        if module in group_for[expert1_groups]:
            return [0], [1.0]
        if module in group_for[expert2_groups]:
            return [1], [1.0]
        raise ValueError("module {} in no group".format(module))

    if scope in ("low", "mid", "high"):
        lo, hi = ranges[scope]
        if lo <= layer < hi:
            return [0, 1], [1.0, 1.0]
        return None

    if scope == "x1_lowmid_x2_midhigh":
        lo, mid, hi = ranges["low"][0], ranges["low"][1], ranges["high"][1]
        if lo <= layer < mid:
            return [0], [1.0]
        if mid <= layer < hi:
            return [1], [1.0]
        return None
    if scope == "x2_lowmid_x1_midhigh":
        lo, mid, hi = ranges["low"][0], ranges["low"][1], ranges["high"][1]
        if lo <= layer < mid:
            return [1], [1.0]
        if mid <= layer < hi:
            return [0], [1.0]
        return None

    if scope in ("pos_cos", "neg_cos"):
        if cos_sets is None:
            raise ValueError("{} requires cosine diagnostics".format(scope))
        layer_means = cos_sets["layer_mean_cosine"]
        if (scope == "pos_cos" and layer_means[name] > 0) or (scope == "neg_cos" and layer_means[name] <= 0):
            return [0, 1], [1.0, 1.0]
        return None

    raise ValueError("unknown scope: {}".format(scope))


def load_cosine_diagnostics(pair: str, input_root: Path) -> Dict[str, float]:
    frame = pd.read_parquet(input_root / "layer_metrics_{}.parquet".format(pair))
    return frame.groupby("layer")["cosine"].mean().to_dict()


def ece(values: Sequence[Tuple[float, float]], bins: int = 15) -> float:
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
    parser.add_argument("--pair", required=True, choices=sorted(PAIRS))
    parser.add_argument("--model-path", default="/data/ckpt/zhaozhuofan/models/llava-v1.5-7b")
    parser.add_argument("--vision-tower", default="/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336")
    parser.add_argument("--projector-path", default="/data/ckpt/zhaozhuofan/models/llava-v1.5-7b/mm_projector.bin")
    parser.add_argument("--checkpoint-root", default=CHECKPOINT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="artifacts/dual_lora_stage02")
    parser.add_argument("--scopes", default="", help="comma-separated subset (default: all)")
    args = parser.parse_args()

    checkpoint_name, dataset, singles, pair_ids = PAIRS[args.pair]
    checkpoint_dir = os.path.join(args.checkpoint_root, checkpoint_name)
    image_folder = "experiments/data/controlled_format_v1"
    with open(os.path.join(DATA_ROOT, dataset, "test_eval.json"), encoding="utf-8") as handle:
        records = json.load(handle)

    input_root = Path(args.output_root)
    cos_sets = None
    if (input_root / "layer_metrics_{}.parquet".format(args.pair)).is_file():
        cos_sets = {"layer_mean_cosine": load_cosine_diagnostics(args.pair, input_root)}

    scopes = [scope for scope in list(MODULE_SCOPES) + ["low", "mid", "high",
              "x1_lowmid_x2_midhigh", "x2_lowmid_x1_midhigh", "pos_cos", "neg_cos"]]
    if args.scopes:
        requested = [s.strip() for s in args.scopes.split(",") if s.strip()]
        missing = sorted(set(requested) - set(scopes))
        if missing:
            raise ValueError("unknown scopes: {}".format(missing))
        scopes = requested

    bundle = load_compose_model(
        model_path=args.model_path, checkpoint_dir=checkpoint_dir,
        vision_tower=args.vision_tower, projector_path=args.projector_path,
        expert_id=None, device=args.device, dtype=torch.bfloat16, model_max_length=2048,
    )
    a_id = int(bundle.tokenizer.encode("A", add_special_tokens=False)[0])
    b_id = int(bundle.tokenizer.encode("B", add_special_tokens=False)[0])
    layer_by_name: Dict[str, ComposeLinear] = {
        name: module for name, module in bundle.model.named_modules()
        if isinstance(module, ComposeLinear)
    }
    singles_rows = {}
    for name in singles:
        rows = {}
        with (EVAL_ROOT / dataset / name / "per_sample.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows[str(row["sample_id"])] = row
        singles_rows[name] = rows

    sample_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []
    with torch.inference_mode():
        slot_to_id = {0: pair_ids[0], 1: pair_ids[1]}
        for scope in scopes:
            for name, module in layer_by_name.items():
                selection = selection_for(name, scope, cos_sets)
                if selection is None:
                    module.clear_default_selection()
                    continue
                slot_ids, gates = selection
                ids = [slot_to_id[s] for s in slot_ids]
                module.set_default_selection(ids, gates, normalization="none")
            per_sample = []
            for offset in range(0, len(records), BATCH_SIZE):
                batch_records = records[offset: offset + BATCH_SIZE]
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
                    max_logit = max(logit_a, logit_b)
                    p_a = float(torch.exp(torch.tensor(logit_a - max_logit)) /
                                (torch.exp(torch.tensor(logit_a - max_logit)) + torch.exp(torch.tensor(logit_b - max_logit))))
                    per_sample.append({"sample_id": str(record["question_id"]), "scope": scope,
                                       "logit_A": logit_a, "logit_B": logit_b, "nll": nll,
                                       "correct": bool(correct), "p_A": p_a,
                                       "target": target, "y_A": 1.0 if target == "A" else 0.0})
            sample_rows.extend(per_sample)
            n = len(per_sample)
            accuracy = statistics.fmean(float(r["correct"]) for r in per_sample)
            synergy = []
            for r in per_sample:
                sid = r["sample_id"]
                best_single_nll = min(float(singles_rows[singles[0]][sid]["answer_token_nll"]),
                                      float(singles_rows[singles[1]][sid]["answer_token_nll"]))
                synergy.append(best_single_nll - float(r["nll"]))
            best_single_acc = max(
                statistics.fmean(float(singles_rows[singles[0]][sid]["correct"]) for sid in singles_rows[singles[0]]),
                statistics.fmean(float(singles_rows[singles[1]][sid]["correct"]) for sid in singles_rows[singles[1]]),
            )
            aggregate_rows.append({
                "pair": args.pair, "scope": scope,
                "accuracy": accuracy, "accuracy_percent": 100.0 * accuracy,
                "accuracy_delta_vs_best_single_pp": 100.0 * (accuracy - best_single_acc),
                "mean_answer_token_nll": statistics.fmean(float(r["nll"]) for r in per_sample),
                "median_synergy": statistics.median(synergy),
                "mean_synergy": statistics.fmean(synergy),
                "positive_synergy_rate": statistics.fmean(float(v > 0) for v in synergy),
                "worst10_mean_synergy": statistics.fmean(sorted(synergy)[: max(1, n // 10)]),
                "brier": statistics.fmean((float(r["p_A"]) - float(r["y_A"])) ** 2 for r in per_sample),
                "ece_15_bins": ece([(float(r["p_A"]), float(r["y_A"])) for r in per_sample]),
            })
            print("scope={} acc={:.2f}% nll={:.4f} delta_best_single={:+.2f}pp".format(
                scope, aggregate_rows[-1]["accuracy_percent"],
                aggregate_rows[-1]["mean_answer_token_nll"],
                aggregate_rows[-1]["accuracy_delta_vs_best_single_pp"]))

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sample_rows).to_parquet(output_root / "ablation_per_sample_{}.parquet".format(args.pair), index=False)
    pd.DataFrame(aggregate_rows).to_parquet(output_root / "ablation_{}.parquet".format(args.pair), index=False)
    summary = {"pair": args.pair, "dataset": dataset, "scopes": scopes,
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    with (output_root / "ablation_{}.json".format(args.pair)).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
