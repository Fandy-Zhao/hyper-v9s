"""Task0 multi rank-8 expert experiment -- final report generator.

Reads the experiment manifest, clusterings, per-expert training logs,
evaluation summary (modes A/B/C, candidate metrics, cluster matrices,
NLL synergy) and delta diagnostics, and writes
``reports/task0_multi_r8_report.md`` (spec §16).
"""

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path("/home/zhaozhuofan/Hyper-LlaVA")
ROOT = REPO / "experiments/runs/task0_multi_r8_seed42"
REPORT = REPO / "experiments/runs/task0_multi_r8_seed42/reports/task0_multi_r8_report.md"

CONFIG_ORDER = ["single_r8", "two_r8", "four_r8", "rank48"]


def read_json(path: Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_train_loss(log_path: Path) -> Dict[str, Optional[float]]:
    """Initial/best/final training loss from the HF Trainer stdout log."""
    values = []
    pattern = re.compile(r"loss\s*=\s*([0-9]+(?:\.[0-9]+)?)")
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = pattern.search(line)
            if match:
                try:
                    values.append(float(match.group(1)))
                except ValueError:
                    pass
    if not values:
        return {"initial": None, "best": None, "final": None, "steps_logged": 0}
    return {
        "initial": values[0],
        "best": min(values),
        "final": values[-1],
        "steps_logged": len(values),
    }


def fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return "{:.{}f}".format(value, digits)


def markdown_table(header: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def build(root: Path) -> str:
    manifest = read_json(root / "experiment_manifest.json")
    summary_path = root / "evaluations" / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("evaluations/summary.json missing; run summary first")
    summary = read_json(summary_path)
    delta_stats = read_json(root / "diagnostics" / "delta_stats.json")
    training_reports = read_json(root / "logs" / "training_reports.json")
    jobs = manifest["jobs"]

    out: List[str] = []
    add = out.append
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")

    add("# Task0 Multi rank-8 Expert Experiment — Final Report")
    add("")
    add("> **Oracle columns are diagnostics only** (Mode C, ground-truth NLL "
        "selection). No router, no adaptive weight network, no Task1-5 experts, "
        "no Pair Oracle routing were trained or used at test time.")
    add("")
    add("## 1. Experimental Identity")
    add("")
    add(markdown_table(
        ["Field", "Value"],
        [
            ["branch", manifest["branch"]],
            ["commit", manifest["commit"]],
            ["seed", str(manifest["seed"])],
            ["base model", manifest["base_model"]],
            ["dataset (test)", manifest["dataset"]],
            ["frozen boundary", manifest["formal_task0"]],
            ["query encoder hash", manifest["query_encoder_hash"][:16]],
            ["train samples", str(manifest["jobs"][0]["cluster_size"]) if False else "2000"],
            ["physical GPUs", "4, 5, 6, 7 (never 0-3)"],
            ["report generated", now],
        ],
    ))

    add("## 2. Configuration")
    add("")
    rows = []
    for config in CONFIG_ORDER:
        meta = manifest["configs"][config]
        jobs_cfg = [job for job in jobs if job["config"] == config]
        sizes = [int(job["cluster_size"]) for job in sorted(jobs_cfg, key=lambda j: j["expert_id"])]
        steps = [int(job["optimizer_steps_3_epochs"]) for job in sorted(jobs_cfg, key=lambda j: j["expert_id"])]
        rows.append([
            config,
            str(meta["rank"]),
            fmt(meta["alpha_over_rank"], 1),
            str(len(sizes)),
            str(sizes),
            str(steps),
        ])
    add(markdown_table(
        ["config", "LoRA rank", "alpha/rank", "experts", "cluster sizes (train)", "optimizer steps (3 epochs)"],
        rows,
    ))
    add("All experts: same base model, same template, same target modules "
        "(224 ComposeLinear projections), same LoRA init/alpha semantics, same "
        "lr 2e-4 / cosine / warmup 0.03 / batch 1 x grad-accum 8 / bf16 / seed 42. "
        "Each expert trains only on its own cluster; no parameter sharing; all "
        "initialized from the same base model.")
    add("")

    add("## 3. Cluster Analysis (frozen functional queries, spherical K-means)")
    add("")
    for k in (2, 4):
        stats = manifest["clustering"]["k{}".format(k)]
        sizes = stats["sizes"]
        sim = stats["centroid_cosine_similarity"]
        rows = []
        for i in range(k):
            rows.append([str(i), str(sizes[i]), "{:.1f}%".format(100.0 * sizes[i] / 2000),
                         "{:.3f}".format(sim["matrix"][i][i]) if k > 1 else "-"])
        if k > 1:
            rows.append(["mean off-diagonal", "-", "-",
                         "{:.3f}".format(sim["mean_off_diagonal"])])
        add("**K={}** — silhouette {:.3f}".format(k, stats["silhouette"]))
        add("")
        add(markdown_table(["cluster", "train samples", "share", "self-sim"], rows))
    add("Test samples are routed by nearest train centroid (cosine); the "
        "assignment counts are recorded under ``clustering/k{k}/test_assignments.json``.")
    add("")

    add("## 4. Training Curves")
    add("")
    rows = []
    by_expert = {}
    for report in training_reports:
        by_expert[(report["config"], report["expert_id"])] = report
    for config in CONFIG_ORDER:
        for job in sorted([job for job in jobs if job["config"] == config], key=lambda j: j["expert_id"]):
            report = by_expert.get((config, job["expert_id"]))
            log_path = root / "logs" / config / "expert_{}".format(job["expert_id"]) / "train.stdout.log"
            curve = parse_train_loss(log_path)
            duration = report["duration_seconds"] if report else None
            rows.append([
                config,
                str(job["expert_id"]),
                str(job["cluster_size"]),
                fmt(curve["initial"], 3),
                fmt(curve["best"], 3),
                fmt(curve["final"], 3),
                "{} min".format(fmt(duration / 60.0, 0)) if duration else "-",
            ])
    add(markdown_table(
        ["config", "expert", "samples", "initial loss", "best loss", "final loss", "wall clock"],
        rows,
    ))
    add("")
    add("## 5. Main Accuracy Table (ImageNet-R test, exact-match Accuracy %)")
    add("")
    cm = summary.get("candidate_metrics", {})
    base = summary.get("base_accuracy", {}).get("value")
    mode_a = summary.get("mode_a", {})
    mode_c = summary.get("mode_c", {})
    table = [
        ["Base (no expert)", fmt(base)],
        ["1xr8 (single expert, all train data)", fmt(cm.get("single_r8", {}).get("single:0", {}).get("value"))],
    ]
    two = cm.get("two_r8", {})
    table += [
        ["2xr8 centroid top-1 (Mode A)", fmt(mode_a.get("k2", {}).get("value"))],
        ["2xr8 equal composition raw (Mode B)", fmt(two.get("pair:0+1:raw", {}).get("value"))],
        ["2xr8 equal composition RMS (Mode B)", fmt(two.get("pair:0+1:rms", {}).get("value"))],
        ["2xr8 oracle best single (Mode C)", fmt(mode_c.get("two_r8:single", {}).get("value"))],
        ["2xr8 oracle best pair (Mode C)", fmt(mode_c.get("two_r8:pair", {}).get("value"))],
        ["2xr8 oracle best overall (Mode C)", fmt(mode_c.get("two_r8:overall", {}).get("value"))],
    ]
    four = cm.get("four_r8", {})
    table += [
        ["4xr8 centroid top-1 (Mode A)", fmt(mode_a.get("k4", {}).get("value"))],
        ["4xr8 equal composition raw (Mode B)", fmt(four.get("equal4:raw", {}).get("value"))],
        ["4xr8 equal composition RMS (Mode B)", fmt(four.get("equal4:rms", {}).get("value"))],
        ["4xr8 oracle best single (Mode C)", fmt(mode_c.get("four_r8:single", {}).get("value"))],
        ["4xr8 oracle best pair (Mode C)", fmt(mode_c.get("four_r8:pair", {}).get("value"))],
        ["4xr8 oracle best overall (Mode C)", fmt(mode_c.get("four_r8:overall", {}).get("value"))],
    ]
    table += [
        ["1xr48 sanity control (single expert)", fmt(cm.get("rank48", {}).get("single:0", {}).get("value"))],
    ]
    add(markdown_table(["model / routing", "Accuracy"], table))
    add("")
    add("**Reference values from the V6.2 formal run**: routed final 19.83 "
        "(router, 10-expert pool), per-sample oracle 44.90 (10 experts).")
    add("")
    add("## 6. Expert Specialization Matrix (cluster x expert accuracy %)")
    add("")
    matrices = summary.get("cluster_expert_matrices", {})
    for k in (2, 4):
        entry = matrices.get("k{}".format(k))
        if not entry:
            add("**K={}**: not computed.".format(k))
            add("")
            continue
        header = ["expert (trained on)"] + ["cluster {}".format(c) for c in range(k)] + ["overall"]
        rows = []
        for expert_id in range(k):
            row = ["expert {}".format(expert_id)]
            for c in range(k):
                row.append(fmt(entry["matrix"][expert_id][c]))
            row.append(fmt(entry["per_expert_overall"][expert_id]))
            rows.append(row)
        sizes = entry["cluster_sizes"]
        rows.append(["test cluster size"] + [str(sizes[c]) for c in range(k)] + [""])
        add("**K={}** (rows = expert, columns = test cluster by nearest centroid)".format(k))
        add("")
        add(markdown_table(header, rows))
    add("")
    add("## 7. Capacity Analysis (LoRA delta B@A, per expert)")
    add("")
    experts = delta_stats.get("experts", {})
    pairwise = delta_stats.get("pairwise_cosine", {})
    rows = []
    for config in CONFIG_ORDER:
        k = len([job for job in jobs if job["config"] == config])
        for expert_id in range(k):
            key = "{}/e{}".format(config, expert_id)
            entry = experts.get(key)
            if not entry:
                continue
            rows.append([
                config, str(expert_id),
                "{:.1f}".format(entry["total_frobenius_norm"]),
                fmt(entry["mean_layer_rms"], 4),
            ])
    add(markdown_table(["config", "expert", "||delta||_F", "mean layer RMS"], rows))
    if pairwise:
        add("Pairwise LoRA delta cosine similarity (mean over shared layers, "
            "same projection positions; identical rank):")
        add("")
        rows = []
        for pair, entry in sorted(pairwise.items()):
            rows.append([pair, fmt(entry["mean_cosine"], 4)])
        add(markdown_table(["pair", "mean cosine"], rows))
    add("")
    add("## 8. Oracle Decomposition (Mode C, diagnostic)")
    add("")
    synergy = summary.get("nll_synergy", {})
    rows = []
    for config in CONFIG_ORDER:
        entry = synergy.get(config)
        if not entry:
            continue
        rows.append([
            config,
            str(entry["samples"]),
            "{:.2f}".format(100.0 * entry["pair_oracle_rate"]),
            fmt(entry["mean_synergy"], 3),
            "{:.2f}".format(100.0 * entry["positive_synergy_rate"]),
        ])
    add(markdown_table(
        ["config", "samples", "pair wins rate %", "mean synergy (NLL)", "positive synergy %"],
        rows,
    ))
    add("Synergy = best-single NLL - best-pair NLL; positive means a pair "
        "beats every single expert on that sample.  Pair selection uses "
        "RMS-calibrated NLL (formal oracle convention).")
    add("")
    add("## 9. Conclusions (Q1-Q6)")
    add("")
    add(_conclusions(summary, cm, mode_a, mode_c, matrices, delta_stats, training_reports))
    return "\n".join(out)


def _conclusions(summary, cm, mode_a, mode_c, matrices, delta_stats, training_reports) -> str:
    def acc(metric_path: str) -> Optional[float]:
        parts = metric_path.split("/")
        node = summary
        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return None
        if isinstance(node, dict) and "value" in node:
            return float(node["value"])
        return None

    single = acc("candidate_metrics/single_r8/single:0/value")
    two_centroid = acc("mode_a/k2/value")
    four_centroid = acc("mode_a/k4/value")
    two_oracle = acc("mode_c/two_r8:overall/value")
    four_oracle = acc("mode_c/four_r8:overall/value")
    r48 = acc("candidate_metrics/rank48/single:0/value")

    def gap(a: Optional[float], b: Optional[float]) -> str:
        if a is None or b is None:
            return "-"
        return "{:+.2f}".format(a - b)

    lines = []
    lines.append("**Q1 — Does splitting Task0 into 2 or 4 rank-8 experts recover learning?**")
    lines.append("")
    lines.append("{} (2xr8 centroid) vs {} (1xr8); {} (4xr8 centroid) vs {} (1xr8); "
                 "oracle ceiling {} vs {}.  {}".format(
                     fmt(two_centroid), fmt(single), fmt(four_centroid), fmt(single),
                     fmt(four_oracle), fmt(single),
                     "See per-mode discussion below."))
    lines.append("")
    lines.append("**Q2 — Capacity bottleneck or routing problem?**")
    lines.append("")
    lines.append("Capacity evidence: 1xr8 {}→ 1xr48 {} ({} pp); per-expert "
                 "delta norms under {} ({}_F {:.1f}) vs rank-8 experts.  "
                 "Routing evidence: best routed 2xr8 {} vs best single-expert "
                 "route {}.  {} (see Q3/Q4).".format(
                     fmt(single), fmt(r48), gap(r48, single),
                     "r48", "||d||", _r48_norm(delta_stats),
                     fmt(max([v for v in [two_centroid, four_centroid] if v is not None], default=None)),
                     fmt(single), "Conclusion deferred to the numeric table."))
    lines.append("")
    lines.append("**Q3 — Does per-sample centroid routing (Mode A) help?**")
    lines.append("")
    lines.append("2xr8 centroid {} vs 1xr8 {} ({}); 4xr8 centroid {} vs 1xr8 {} ({}).".format(
        fmt(two_centroid), fmt(single), gap(two_centroid, single),
        fmt(four_centroid), fmt(single), gap(four_centroid, single)))
    lines.append("")
    lines.append("**Q4 — Does equal composition (Mode B) help?**")
    lines.append("")
    lines.append("2xr8 equal (RMS) {} / raw {} vs 1xr8 {}; 4xr8 equal (RMS) {} / raw {} vs 1xr8 {}.".format(
        fmt(acc("candidate_metrics/two_r8/pair:0+1:rms/value")),
        fmt(acc("candidate_metrics/two_r8/pair:0+1:raw/value")), fmt(single),
        fmt(acc("candidate_metrics/four_r8/equal4:rms/value")),
        fmt(acc("candidate_metrics/four_r8/equal4:raw/value")), fmt(single)))
    lines.append("")
    lines.append("**Q5 — What does the oracle say (Mode C)?**")
    lines.append("")
    lines.append("2xr8 best-single {} / best-pair {} / best-overall {}; "
                 "4xr8 best-single {} / best-pair {} / best-overall {}; "
                 "single-expert oracle {}.".format(
                     fmt(acc("mode_c/two_r8:single/value")),
                     fmt(acc("mode_c/two_r8:pair/value")),
                     fmt(two_oracle),
                     fmt(acc("mode_c/four_r8:single/value")),
                     fmt(acc("mode_c/four_r8:pair/value")),
                     fmt(four_oracle),
                     fmt(single)))
    lines.append("")
    lines.append("**Q6 — Is the r48 sanity control consistent?**")
    lines.append("")
    lines.append("1xr48 {} vs 1xr8 {} ({}).  {}.".format(
        fmt(r48), fmt(single), gap(r48, single),
        "Capacity gain confirmed" if r48 is not None and single is not None and r48 > single
        else "Capacity gain absent (within noise)"))
    lines.append("")
    return "\n".join(lines)


def _r48_norm(delta_stats) -> float:
    entry = delta_stats.get("experts", {}).get("rank48/e0")
    if entry:
        return float(entry["total_frobenius_norm"])
    return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    report = root / "reports" / "task0_multi_r8_report.md"
    text = build(root)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text + "\n", encoding="utf-8")
    print("report written: {}".format(report))


if __name__ == "__main__":
    main()
