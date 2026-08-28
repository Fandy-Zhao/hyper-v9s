"""Offline-only clustering diagnostic for the controlled seed42 study."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from compose.expansion.query_clustering import cosine_silhouette, spherical_kmeans


SEEDS = (0, 1, 2, 3, 4, 42, 43, 44)
TASKS = ("ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR-Math", "Flickr30k")


def load_queries(formal: Path, task_id: int):
    residual = json.loads((formal / f"task{task_id}" / "residual" / "residual.json").read_text())
    ids = [str(row["sample_id"]) for row in residual]
    features = json.loads((formal / f"task{task_id}" / "features" / "train_features.json").read_text())["records"]
    queries = torch.tensor([features[sid]["query"] for sid in ids], dtype=torch.float32)
    queries = torch.nn.functional.normalize(queries, dim=1)
    return ids, queries


def fit_current(queries, seed):
    traces, labels = {}, {}
    for k in (2, 3, 4):
        assignment, _ = spherical_kmeans(queries, k, seed=seed)
        labels[k] = assignment.numpy()
        traces[k] = cosine_silhouette(queries, assignment, k, seed=seed, sample_size=2000)
    best = max(traces, key=lambda key: traces[key])
    selected = best if traces[best] >= 0.15 else 1
    selected_labels = labels[selected] if selected > 1 else np.zeros(len(queries), dtype=np.int64)
    return selected, traces, selected_labels


def fit_standard(queries, seed, n_init):
    traces, labels = {}, {}
    for k in (2, 3, 4):
        best_inertia, best_assignment = None, None
        for init in range(n_init):
            init_seed = int(seed + init * 1_000_003)
            assignment, centers = spherical_kmeans(queries, k, seed=init_seed)
            inertia = float((1.0 - (queries * centers[assignment]).sum(dim=1)).sum())
            if best_inertia is None or inertia < best_inertia:
                best_inertia, best_assignment = inertia, assignment
        array = best_assignment.numpy()
        labels[k] = array
        traces[k] = float(silhouette_score(queries.numpy(), array, metric="cosine"))
    best = max(traces, key=lambda key: traces[key])
    selected = best if traces[best] >= 0.15 else 1
    selected_labels = labels[selected] if selected > 1 else np.zeros(len(queries), dtype=np.int64)
    return selected, traces, selected_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    formal, output = Path(args.formal_root), Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for task_id, task_name in enumerate(TASKS):
        _, queries = load_queries(formal, task_id)
        task_results = {}
        for algorithm, n_init in (("current_nn_ninit1", 1), ("standard_ninit10", 10), ("standard_ninit20", 20)):
            for seed in SEEDS:
                result = fit_current(queries, seed) if algorithm.startswith("current") else fit_standard(queries, seed, n_init)
                task_results[(algorithm, seed)] = result
        for (algorithm, seed), (selected, trace, labels) in task_results.items():
            reference = task_results[(algorithm, 42)][2]
            sizes = sorted(Counter(int(value) for value in labels).values(), reverse=True)
            rows.append({
                "task_id": task_id, "task": task_name, "algorithm": algorithm,
                "n_init": 1 if algorithm.startswith("current") else int(algorithm.rsplit("ninit", 1)[1]),
                "seed": seed, "selected_k": selected,
                "S2": trace[2], "S3": trace[3], "S4": trace[4],
                "cluster_sizes": "|".join(map(str, sizes)),
                "ARI_vs_seed42": adjusted_rand_score(reference, labels),
                "NMI_vs_seed42": normalized_mutual_info_score(reference, labels),
            })
    csv_path = output / "E_clustering_algorithm_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    focus = {name: [row for row in rows if row["task"] == name] for name in ("ArxivQA", "IconQA", "Flickr30k")}
    report = ["# Experiment E — Clustering Algorithm Diagnostic", "", "Offline only; no experts were trained.", ""]
    for name, values in focus.items():
        report += [f"## {name}", "", "| algorithm | selected-K over seeds | mean ARI vs seed42 | mean NMI vs seed42 |", "|---|---|---:|---:|"]
        for algorithm in ("current_nn_ninit1", "standard_ninit10", "standard_ninit20"):
            subset = [row for row in values if row["algorithm"] == algorithm]
            report.append("| {} | {} | {:.4f} | {:.4f} |".format(
                algorithm, ",".join(str(row["selected_k"]) for row in subset),
                sum(row["ARI_vs_seed42"] for row in subset) / len(subset),
                sum(row["NMI_vs_seed42"] for row in subset) / len(subset)))
        report.append("")
    (output / "E_clustering_stability_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"status": "COMPLETE", "rows": len(rows), "csv": str(csv_path)}))


if __name__ == "__main__":
    main()
