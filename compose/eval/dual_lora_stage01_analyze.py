"""Stage 01 post-processing: per-sample oracle + figures.

Reads the per-point per-sample parquet files written by
dual_lora_stage01_grid.py and produces:

  per_sample_oracle.parquet -- for every (pair, mode) and sample, the grid
      point with max accuracy (ties -> min answer-token NLL), classified as:
        needs_none   : best point is (0, 0)
        needs_only_A : alpha > 0, beta == 0
        needs_only_B : alpha == 0, beta > 0
        needs_both   : alpha > 0 and beta > 0
      plus oracle accuracy and the correct->wrong / wrong->correct
      transition counts relative to the best single expert.
  figures/ -- per (pair, mode): accuracy heatmap, NLL heatmap,
      accuracy-vs-NLL scatter, best-point distribution histograms,
      transition matrix.

The per-sample oracle is a DIAGNOSTIC ONLY and must never be reported as a
formal method.

Usage:
  python -m compose.eval.dual_lora_stage01_analyze --input-root artifacts/dual_lora_stage01
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAIRS = {
    "a_independent_b": ("A_plus_B", ("expert_a", "independent_b"), "rank16_ab"),
    "a_residual_b": ("A_plus_B", ("expert_a", "residual_b"), "rank16_ab"),
    "independent_b_c": ("B_plus_C", ("independent_b", "expert_c"), "upper_bc"),
    "residual_b_c": ("B_plus_C", ("residual_b", "expert_c"), "upper_bc"),
}
EVAL_ROOT = Path("experiments/runs/format_controlled_composition_v1/evaluation/seed42")
COARSE_VALUES = [round(step * 0.1, 2) for step in range(16)]


def read_recorded(pair: str) -> Dict[str, Dict[str, dict]]:
    dataset, singles, rank16 = PAIRS[pair]
    result = {}
    for name in (*singles, rank16):
        rows = {}
        with (EVAL_ROOT / dataset / name / "per_sample.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows[str(row["sample_id"])] = row
        result[name] = rows
    return result


def oracle_per_sample(frame: pd.DataFrame, recorded: Dict[str, Dict[str, dict]],
                      pair: str) -> pd.DataFrame:
    dataset, singles, _ = PAIRS[pair]
    s_a, s_b = recorded[singles[0]], recorded[singles[1]]
    rows = []
    for (mode, sample_id), group in frame.groupby(["mode", "sample_id"]):
        best = group.sort_values(
            by=["correct", "nll"], ascending=[False, True]
        ).iloc[0]
        target = group.iloc[0]["target"]
        alpha, beta = float(best["alpha"]), float(best["beta"])
        if alpha == 0.0 and beta == 0.0:
            kind = "needs_none"
        elif beta == 0.0:
            kind = "needs_only_A"
        elif alpha == 0.0:
            kind = "needs_only_B"
        else:
            kind = "needs_both"
        row = recorded[s_a if target == "A" else "B"].get(sample_id)
        rows.append({
            "pair": pair, "mode": mode, "sample_id": sample_id, "target": target,
            "best_alpha": alpha, "best_beta": beta, "oracle_correct": bool(best["correct"]),
            "needs": kind, "best_nll": float(best["nll"]),
        })
    return pd.DataFrame(rows)


def grid_as_matrix(frame: pd.DataFrame, metric: str, mode: str) -> np.ndarray:
    sub = frame[frame["mode"] == mode]
    matrix = np.full((len(COARSE_VALUES), len(COARSE_VALUES)), np.nan)
    for _, row in sub.iterrows():
        i = COARSE_VALUES.index(float(row["alpha"]))
        j = COARSE_VALUES.index(float(row["beta"]))
        matrix[i, j] = row[metric]
    return matrix


def plot_surface(matrix: np.ndarray, pair: str, mode: str, metric: str,
                 title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, origin="lower", aspect="auto",
                   extent=[COARSE_VALUES[0] - 0.05, COARSE_VALUES[-1] + 0.05,
                           COARSE_VALUES[0] - 0.05, COARSE_VALUES[-1] + 0.05])
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("beta (expert 2 gate)")
    ax.set_ylabel("alpha (expert 1 gate)")
    ax.set_title("{} {}\n{}".format(pair, mode, title))
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="artifacts/dual_lora_stage01")
    args = parser.parse_args()
    root = Path(args.input_root)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    oracle_frames = []
    summary = {}
    for pair in PAIRS:
        recorded = read_recorded(pair)
        for phase in ("coarse", "fine"):
            parquet = root / "per_sample" / "{}_{}.parquet".format(pair, phase)
            if not parquet.is_file():
                print("skip missing {}".format(parquet))
                continue
            frame = pd.read_parquet(parquet)
            oracle = oracle_per_sample(frame, recorded, pair)
            oracle_frames.append(oracle)
            for mode in sorted(frame["mode"].unique()):
                acc_mat = grid_as_matrix(frame, "correct", mode)
                nll_mat = grid_as_matrix(frame, "nll", mode)
                acc_mean = float(np.nanmean(acc_mat))
                nll_mean = float(np.nanmean(nll_mat))
                best_i, best_j = np.unravel_index(np.nanargmax(acc_mat), acc_mat.shape)
                plot_surface(acc_mat, pair, mode, "accuracy",
                             "accuracy (phase {})".format(phase),
                             figures / "{}_{}_{}_accuracy.png".format(pair, phase, mode))
                plot_surface(nll_mat, pair, mode, "nll",
                             "answer-token NLL (phase {})".format(phase),
                             figures / "{}_{}_{}_nll.png".format(pair, phase, mode))
                # accuracy vs NLL scatter over grid points
                sub = frame[frame["mode"] == mode]
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(sub["nll"], sub["correct"].astype(float).replace({False: 0.0, True: 1.0}),
                           s=6, alpha=0.5)
                ax.set_xlabel("mean answer-token NLL")
                ax.set_ylabel("accuracy")
                ax.set_title("{} {} accuracy-vs-NLL (phase {})".format(pair, mode, phase))
                fig.tight_layout()
                fig.savefig(figures / "{}_{}_{}_scatter.png".format(pair, phase, mode), dpi=120)
                plt.close(fig)
                summary["{}_{}_{}".format(pair, phase, mode)] = {
                    "mean_accuracy": acc_mean, "mean_nll": nll_mean,
                    "best_alpha": COARSE_VALUES[best_i], "best_beta": COARSE_VALUES[best_j],
                    "best_accuracy": float(acc_mat[best_i, best_j]),
                    "best_nll": float(nll_mat[best_i, best_j]),
                }
            if phase == "coarse" and parquet.is_file():
                best_alpha, best_beta = summary["{}_{}_raw".format(pair, phase)]["best_alpha"], \
                                        summary["{}_{}_raw".format(pair, phase)]["best_beta"]
                plot_surface(acc_mat, pair, "raw", "accuracy-best", "best point ({}, {})".format(best_alpha, best_beta),
                             figures / "{}_best_marker.png".format(pair))

    if not oracle_frames:
        raise SystemExit("no oracle frames computed (grid parquet files missing)")
    oracle_all = pd.concat(oracle_frames, ignore_index=True)
    oracle_all.to_parquet(root / "per_sample_oracle.parquet", index=False)

    counts = oracle_all.groupby(["pair", "mode", "needs"]).size().unstack(fill_value=0)
    summary["oracle"] = {
        "counts_by_needs": counts.to_dict(),
        "oracle_accuracy": oracle_all.groupby(["pair", "mode"])["oracle_correct"].mean().mul(100).to_dict(),
        "note": "per-sample oracle is DIAGNOSTIC ONLY; never a formal method",
    }
    with (root / "oracle_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
