"""Stage 01: two-dimensional oracle weight search for one expert pair.

Scans (alpha, beta) over {0.0, 0.1, ..., 1.5}^2 (coarse, 256 points) plus an
optional fine grid (step 0.025) around the coarse best point, in two modes:

  A  raw deltas:            y = base + alpha*u_A + beta*u_B
  B  RMS-calibrated deltas: y = base + pair_scale * sum_i scalar_i *
                            kappa_i^l * u_i^l  (per-layer kappa from the
                            val-split RMS statistics, formal Compose rule;
                            the fixed point (1,1) is the current
                            RMS-calibrated combination)
  C  first-expert-fixed:    the alpha=1 slice of mode A (extracted, no
                            extra forwards)

Every grid point is a full 400-sample forward (batch 8). Per-sample
answer-token logits are stored so accuracy / NLL / Brier / ECE / synergy /
marginals can be recomputed for every point without re-running the model.

Reference singles and rank-16 per-sample values come from the recorded
format-controlled evaluation (evaluation/seed42/...), never from new runs.

Outputs (under --output-root):
  per_sample/{pair}_{phase}.parquet   per-sample logits at every grid point
  grid_results.parquet                per-point aggregate metrics
  per_sample_oracle.parquet           per-sample best-point diagnostics
  figures/*.png                       heatmaps, scatter, transitions

Usage:
  python -m compose.eval.dual_lora_stage01_grid --pair a_independent_b \
      --phase coarse --device cuda:4 --output-root artifacts/dual_lora_stage01
  python -m compose.eval.dual_lora_stage01_grid --pair a_independent_b \
      --phase fine --best-point 0.9,0.8 --device cuda:4 --output-root ...
"""

import argparse
import json
import math
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from compose.adapters.lora import ComposeLinear
from compose.eval.load_compose import EvaluationBundle, load_compose_model
from compose.lora.rms_composition import RMSCompositionConfig, frozen_coefficients
from compose.oracle.evaluator import _collate, _prepare_multimodal_batch

COARSE_STEP = 0.1
FINE_STEP = 0.025
FINE_RADIUS = 0.2
BATCH_SIZE = 8
PAIR_SCALE = 1.0 / math.sqrt(2.0)

# pair name -> (checkpoint template under /data/ckpt/.../assembled/seed{seed},
#               dataset, single names in recorded evaluation, rank16 name)
PAIRS = {
    "a_independent_b": ("assembled/seed42/a_independent_b", "A_plus_B", ("expert_a", "independent_b"), "rank16_ab", (0, 1)),
    "a_residual_b": ("seed42/residual_b", "A_plus_B", ("expert_a", "residual_b"), "rank16_ab", (0, 1)),
    "independent_b_c": ("assembled/seed42/independent_b_c", "B_plus_C", ("independent_b", "expert_c"), "upper_bc", (1, 2)),
    "residual_b_c": ("assembled/seed42/residual_b_c", "B_plus_C", ("residual_b", "expert_c"), "upper_bc", (1, 2)),
}
SEED = 42
EVAL_ROOT = "experiments/runs/format_controlled_composition_v1/evaluation/seed42"
CHECKPOINT_ROOT = "/data/ckpt/zhaozhuofan/compose/format_controlled_composition_v1"
DATA_ROOT = "experiments/data/controlled_format_v1_training/instructions"


def coarse_points() -> List[Tuple[float, float]]:
    values = [round(step * COARSE_STEP, 2) for step in range(16)]
    return [(a, b) for a in values for b in values]


def fine_points(best: Tuple[float, float], done: set) -> List[Tuple[float, float]]:
    lo_a, hi_a = max(0.0, best[0] - FINE_RADIUS), min(1.5, best[0] + FINE_RADIUS)
    lo_b, hi_b = max(0.0, best[1] - FINE_RADIUS), min(1.5, best[1] + FINE_RADIUS)
    points = []
    value = lo_a
    while value <= hi_a + 1e-9:
        row = value
        value2 = lo_b
        while value2 <= hi_b + 1e-9:
            point = (round(row, 4), round(value2, 4))
            if point not in done:
                points.append(point)
            value2 = round(value2 + FINE_STEP, 4)
        value = round(value + FINE_STEP, 4)
    return points


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


def read_recorded_per_sample(pair: str) -> Dict[str, Dict[str, dict]]:
    """Load recorded per-sample rows for the pair's singles and rank-16."""
    dataset, singles, rank16 = PAIRS[pair][1], PAIRS[pair][2], PAIRS[pair][3]
    result: Dict[str, Dict[str, dict]] = {}
    for name in (*singles, rank16):
        path = Path(EVAL_ROOT) / dataset / name / "per_sample.jsonl"
        rows = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows[str(row["sample_id"])] = row
        result[name] = rows
    return result


class GridEvaluator:
    """One loaded model, many (alpha, beta) forward passes."""

    def __init__(self, bundle: EvaluationBundle, records: Sequence[dict],
                 image_folder: str, device: str, expert_ids: Sequence[int]) -> None:
        self.bundle = bundle
        self.records = records
        self.image_folder = image_folder
        self.device = device
        self.expert_ids = tuple(int(v) for v in expert_ids)
        self.a_id = int(bundle.tokenizer.encode("A", add_special_tokens=False)[0])
        self.b_id = int(bundle.tokenizer.encode("B", add_special_tokens=False)[0])
        self.rms_coefficients: Optional[Dict[str, Tuple[float, float]]] = None
        self._hooks: List[Any] = []
        self._captured: List[Dict[str, torch.Tensor]] = []
        self._layer_by_name: Dict[str, ComposeLinear] = {}

    def set_raw_gates(self, alpha: float, beta: float) -> None:
        self._remove_hooks()
        if alpha == 0.0 and beta == 0.0:
            # (0,0) is not expressible via ComposeSelection (>=1 positive gate
            # required); the equivalent base forward is a cleared selection.
            self.bundle.expert_pool.manager.clear_default_selection()
            return
        self.bundle.expert_pool.manager.set_default_selection(list(self.expert_ids), [alpha, beta], normalization="none")

    def set_rms_gates(self, alpha: float, beta: float, coefficients: Dict[str, Tuple[float, float]]) -> None:
        """Apply the formal RMS-calibrated rule per layer with grid scalars:
        y = base + pair_scale * (alpha * kappa_A^l * u_A^l + beta * kappa_B^l * u_B^l),
        i.e. rms_compose with expert_scalars=(alpha, beta)."""
        self.rms_coefficients = coefficients
        self._remove_hooks()
        self.bundle.expert_pool.manager.clear_default_selection()
        self._layer_by_name = {
            name: module for name, module in self.bundle.model.named_modules()
            if isinstance(module, ComposeLinear)
        }
        config = RMSCompositionConfig()
        pair_scale = config.pair_scale

        for name, module in self._layer_by_name.items():
            def post_hook(module, inputs, base_output, name=name, alpha=alpha, beta=beta):
                hidden = inputs[0]
                kappa_a, kappa_b = coefficients[name]
                delta_a = module.experts[str(self.expert_ids[0])](hidden).to(base_output.dtype)
                delta_b = module.experts[str(self.expert_ids[1])](hidden).to(base_output.dtype)
                composed = base_output
                composed = composed + delta_a * (pair_scale * kappa_a * alpha)
                composed = composed + delta_b * (pair_scale * kappa_b * beta)
                return composed
            self._hooks.append(module.register_forward_hook(post_hook))

    def _remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._captured = []

    def close(self) -> None:
        self._remove_hooks()

    def run(self, alpha: float, beta: float, mode: str,
            coefficients: Optional[Dict[str, Tuple[float, float]]] = None) -> List[Dict[str, Any]]:
        if mode == "raw":
            self.set_raw_gates(alpha, beta)
        elif mode == "rms":
            if coefficients is None:
                raise ValueError("rms mode requires coefficients")
            self.set_rms_gates(alpha, beta, coefficients)
        else:
            raise ValueError("unknown mode: {}".format(mode))
        rows: List[Dict[str, Any]] = []
        with torch.inference_mode():
            for offset in range(0, len(self.records), BATCH_SIZE):
                batch_records = self.records[offset: offset + BATCH_SIZE]
                raw = _collate(batch_records, self.bundle, self.image_folder, self.device)
                prepared = _prepare_multimodal_batch(self.bundle, raw)
                logits = self.bundle.model(**prepared).logits
                labels = prepared["labels"]
                for index, record in enumerate(batch_records):
                    label_row = labels[index].tolist()
                    positions = [i for i, value in enumerate(label_row) if value != -100]
                    answer_pos = positions[0] - 1
                    token_logits = logits[index, answer_pos].float()
                    target = str(record["answer"])
                    target_id = self.a_id if target == "A" else self.b_id
                    target_logit = token_logits[target_id]
                    log_sum = torch.logsumexp(token_logits, dim=0)
                    nll = float(log_sum - target_logit)
                    logit_a = float(token_logits[self.a_id])
                    logit_b = float(token_logits[self.b_id])
                    max_logit = max(logit_a, logit_b)
                    p_a = float(torch.exp(torch.tensor(logit_a - max_logit)) /
                                (torch.exp(torch.tensor(logit_a - max_logit)) + torch.exp(torch.tensor(logit_b - max_logit))))
                    correct = (logit_a > logit_b) == (target == "A")
                    rows.append({"sample_id": str(record["question_id"]),
                                 "logit_A": logit_a, "logit_B": logit_b, "nll": nll,
                                 "correct": bool(correct), "p_A": p_a,
                                 "target": target, "y_A": 1.0 if target == "A" else 0.0})
        if mode == "rms":
            self._remove_hooks()
        return rows


def aggregate(rows: Sequence[Dict[str, Any]], pair: str,
              recorded: Dict[str, Dict[str, dict]]) -> Dict[str, Any]:
    n = len(rows)
    accuracy = statistics.fmean(float(r["correct"]) for r in rows)
    nll_mean = statistics.fmean(float(r["nll"]) for r in rows)
    nll_median = statistics.median(float(r["nll"]) for r in rows)
    brier = statistics.fmean((float(r["p_A"]) - float(r["y_A"])) ** 2 for r in rows)
    ece_val = ece([(float(r["p_A"]), float(r["y_A"])) for r in rows])
    dataset = PAIRS[pair][1]
    singles = PAIRS[pair][2]
    rank16 = PAIRS[pair][3]
    s_a, s_b = recorded[singles[0]], recorded[singles[1]]
    r16 = recorded[rank16]
    synergy = []
    for r in rows:
        sid = r["sample_id"]
        best_single_nll = min(float(s_a[sid]["answer_token_nll"]), float(s_b[sid]["answer_token_nll"]))
        synergy.append(best_single_nll - float(r["nll"]))
    positive_rate = statistics.fmean(float(v > 0) for v in synergy)
    worst10 = sorted(synergy)[: max(1, n // 10)]
    marginals = {
        "g_b_given_a": statistics.fmean(float(s_a[r["sample_id"]]["answer_token_nll"]) - float(r["nll"]) for r in rows),
        "g_a_given_b": statistics.fmean(float(s_b[r["sample_id"]]["answer_token_nll"]) - float(r["nll"]) for r in rows),
    }
    best_single_acc = max(
        statistics.fmean(float(s_a[sid]["correct"]) for sid in s_a),
        statistics.fmean(float(s_b[sid]["correct"]) for sid in s_b),
    )
    rank16_acc = statistics.fmean(float(r16[sid]["correct"]) for sid in r16)
    return {
        "accuracy": accuracy, "accuracy_percent": 100.0 * accuracy,
        "accuracy_delta_vs_best_single_pp": 100.0 * (accuracy - best_single_acc),
        "accuracy_gap_vs_rank16_pp": 100.0 * (rank16_acc - accuracy),
        "mean_answer_token_nll": nll_mean, "median_answer_token_nll": nll_median,
        "brier": brier, "ece_15_bins": ece_val,
        "mean_synergy": statistics.fmean(synergy), "median_synergy": statistics.median(synergy),
        "positive_synergy_rate": positive_rate,
        "worst10_mean_synergy": statistics.fmean(worst10),
        "g_b_given_a": marginals["g_b_given_a"], "g_a_given_b": marginals["g_a_given_b"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True, choices=sorted(PAIRS))
    parser.add_argument("--phase", required=True, choices=("coarse", "fine"))
    parser.add_argument("--best-point", default="", help="'a,b' for the fine phase")
    parser.add_argument("--mode", default="both", choices=("raw", "rms", "both"))
    parser.add_argument("--model-path", default="/data/ckpt/zhaozhuofan/models/llava-v1.5-7b")
    parser.add_argument("--vision-tower", default="/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336")
    parser.add_argument("--projector-path", default="/data/ckpt/zhaozhuofan/models/llava-v1.5-7b/mm_projector.bin")
    parser.add_argument("--checkpoint-root", default=CHECKPOINT_ROOT)
    parser.add_argument("--rms-stats", default="", help="JSON from dual_lora_stage01_rms_stats.py")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="artifacts/dual_lora_stage01")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    checkpoint_name, dataset, _, _, _ = PAIRS[args.pair]
    checkpoint_dir = os.path.join(args.checkpoint_root, checkpoint_name)
    image_folder = "experiments/data/controlled_format_v1"
    question_file = os.path.join(DATA_ROOT, dataset, "test_eval.json")
    with open(question_file, encoding="utf-8") as handle:
        records = json.load(handle)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    bundle = load_compose_model(
        model_path=args.model_path, checkpoint_dir=checkpoint_dir,
        vision_tower=args.vision_tower, projector_path=args.projector_path,
        expert_id=None, device=args.device, dtype=torch.bfloat16, model_max_length=2048,
    )
    evaluator = GridEvaluator(bundle, records, image_folder, args.device, PAIRS[args.pair][4])

    rms_coefficients = None
    if args.mode in ("rms", "both"):
        stats_path = args.rms_stats or os.path.join(args.output_root, "rms_stats", args.pair + ".json")
        if not os.path.isfile(stats_path):
            raise FileNotFoundError("RMS statistics not found; run dual_lora_stage01_rms_stats.py first: {}".format(stats_path))
        with open(stats_path, encoding="utf-8") as handle:
            stats = json.load(handle)
        entries = stats["entries"]
        pair_ids = PAIRS[args.pair][4]
        by_layer: Dict[str, Dict[int, float]] = {}
        for token, entry in entries.items():
            moments = entry["delta"]
            rms = math.sqrt(max(moments["sum_squares"] / moments["count"], 0.0)) if moments["count"] else 0.0
            by_layer.setdefault(entry["key"]["layer_name"], {})[int(entry["key"]["expert_id"])] = rms
        config = RMSCompositionConfig()
        rms_coefficients = {}
        for layer, rms_by_expert in by_layer.items():
            if pair_ids[0] not in rms_by_expert or pair_ids[1] not in rms_by_expert:
                raise KeyError("missing per-layer RMS for {}".format(layer))
            kappa, _ = frozen_coefficients(rms_by_expert[pair_ids[0]], rms_by_expert[pair_ids[1]], config)
            rms_coefficients[layer] = kappa

    if args.phase == "coarse":
        points = coarse_points()
        modes = ("raw", "rms") if args.mode == "both" else (args.mode,)
    else:
        if not args.best_point:
            raise ValueError("fine phase requires --best-point a,b")
        best = tuple(float(v) for v in args.best_point.split(","))
        coarse = set(coarse_points())
        points = fine_points(best, coarse)
        modes = ("raw", "rms") if args.mode == "both" else (args.mode,)

    recorded = read_recorded_per_sample(args.pair)
    out_root = Path(args.output_root)
    per_sample_dir = out_root / "per_sample"
    per_sample_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = per_sample_dir / "{}_{}.parquet".format(args.pair, args.phase)

    # Resume: skip (mode, alpha, beta) points already present in the parquet.
    done_keys = set()
    if parquet_path.is_file():
        existing = pd.read_parquet(parquet_path)
        for _, row in existing.iterrows():
            done_keys.add((str(row["mode"]), float(row["alpha"]), float(row["beta"])))
        print("resume: {} points already completed".format(len(done_keys)))
    pending = [(a, b) for a, b in points if any((m, a, b) not in done_keys for m in modes)]
    print("pending points: {} of {}".format(len(pending), len(points)))

    all_rows: List[Dict[str, Any]] = []
    grid_rows: List[Dict[str, Any]] = []
    checkpoint_interval = 25
    try:
        for index, (alpha, beta) in enumerate(pending):
            for mode in modes:
                if mode == "rms":
                    sample_rows = evaluator.run(alpha, beta, "rms", rms_coefficients)
                else:
                    sample_rows = evaluator.run(alpha, beta, "raw")
                for row in sample_rows:
                    all_rows.append({"mode": mode, "alpha": alpha, "beta": beta, **row})
                metrics = aggregate(sample_rows, args.pair, recorded)
                grid_rows.append({"pair": args.pair, "phase": args.phase, "mode": mode,
                                  "alpha": alpha, "beta": beta, **metrics})
                print("[{}/{}] mode={} alpha={} beta={} acc={:.4f} nll={:.4f} brier={:.4f}".format(
                    index + 1, len(points), mode, alpha, beta,
                    metrics["accuracy_percent"], metrics["mean_answer_token_nll"], metrics["brier"]))
            if (index + 1) % checkpoint_interval == 0 or index == len(pending) - 1:
                combined_rows = pd.DataFrame(all_rows)
                combined_grid = pd.DataFrame(grid_rows)
                if parquet_path.is_file():
                    combined_rows = pd.concat([pd.read_parquet(parquet_path), combined_rows], ignore_index=True)
                if (out_root / "grid_results.parquet").is_file():
                    combined_grid = pd.concat([pd.read_parquet(out_root / "grid_results.parquet"), combined_grid], ignore_index=True)
                combined_rows.to_parquet(parquet_path, index=False)
                combined_grid.to_parquet(out_root / "grid_results.parquet", index=False)
    finally:
        evaluator.close()
        del bundle
        torch.cuda.empty_cache()

    best = max(grid_rows, key=lambda r: (r["accuracy_percent"], -r["mean_answer_token_nll"]))
    per_mode_best = {}
    for mode in {r["mode"] for r in grid_rows}:
        mode_rows = [r for r in grid_rows if r["mode"] == mode]
        best_of_mode = max(mode_rows, key=lambda r: (r["accuracy_percent"], -r["mean_answer_token_nll"]))
        per_mode_best[mode] = {k: best_of_mode[k] for k in ("mode", "alpha", "beta", "accuracy_percent",
                                                            "mean_answer_token_nll", "brier", "ece_15_bins",
                                                            "mean_synergy", "median_synergy", "positive_synergy_rate",
                                                            "worst10_mean_synergy")}
    print("BEST POINT: mode={} alpha={} beta={} acc={:.2f}% nll={:.4f} brier={:.4f}".format(
        best["mode"], best["alpha"], best["beta"], best["accuracy_percent"],
        best["mean_answer_token_nll"], best["brier"]))
    with (out_root / "best_point_{}_{}.json".format(args.pair, args.phase)).open("w", encoding="utf-8") as handle:
        json.dump({"best": {k: best[k] for k in ("pair", "phase", "mode", "alpha", "beta", "accuracy_percent",
                                                 "mean_answer_token_nll", "brier", "ece_15_bins",
                                                 "mean_synergy", "median_synergy", "positive_synergy_rate",
                                                 "worst10_mean_synergy")},
                   "per_mode_best": per_mode_best,
                   "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()},
                  handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
