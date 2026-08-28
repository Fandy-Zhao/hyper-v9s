"""Offline specialization analysis over Experiment-A checkpoints only."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from compose.experiments.rank_study_eval import IMAGES, _feature_queries, _sample_id, _snapshot
from compose.eval.load_compose import load_compose_model
from compose.lora.rms import apply_kappa_calibration
from compose.oracle.candidate_sets import CandidateSet
from compose.oracle.evaluator import _candidate_nll, _collate, _prepare_multimodal_batch
from compose.router.router import ComposeRouter, load_compose_router_checkpoint


BASE_MODEL = "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
VISION_TOWER = "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
PROJECTOR = BASE_MODEL + "/mm_projector.bin"
TASK_NAMES = {0: "ImageNet-R", 1: "ArxivQA", 4: "CLEVR-Math"}


def evaluate_candidate(bundle, records, ids, device, batch_size):
    candidate = CandidateSet(0, tuple(ids), tuple(1.0 for _ in ids), "none")
    output = {}
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        raw = _collate(batch, bundle, IMAGES, device)
        prepared = _prepare_multimodal_batch(bundle, raw)
        losses, _ = _candidate_nll(bundle, candidate, prepared)
        for index, record in enumerate(batch):
            output[_sample_id(record)] = float(losses[index])
    return output


def mean(values, ids):
    return statistics.fmean(values[sid] for sid in ids)


def jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def run(run_root: Path, task_id: int, rank: int, output: Path, device: str, batch_size: int):
    task_root = run_root / f"task{task_id}"
    formation = json.loads((task_root / "cluster" / "formation.json").read_text())
    formed = formation["formed_experts"]
    if len(formed) != 2:
        raise ValueError(f"F requires exactly two formed experts, got {len(formed)}")
    expert_a, expert_b = [int(item["expert_id"]) for item in formed]
    cluster_a, cluster_b = [set(str(value) for value in item["sample_ids"]) for item in formed]
    all_records = json.loads((task_root / "data" / "teacher_train.json").read_text())
    cluster_boundary = cluster_a | cluster_b
    records = [record for record in all_records if _sample_id(record) in cluster_boundary]
    record_ids = {_sample_id(record) for record in records}
    if cluster_boundary != record_ids:
        raise ValueError("cluster members are missing from the frozen training set")
    snapshot, pool = _snapshot(run_root, task_id)
    bundle = load_compose_model(model_path=BASE_MODEL, checkpoint_dir=str(pool), vision_tower=VISION_TOWER, projector_path=PROJECTOR, expert_id=None, device=device, dtype=torch.bfloat16, model_max_length=2048)
    calibration = bundle.load_summary.get("rms_calibration")
    if calibration:
        apply_kappa_calibration(bundle.model, calibration)
    base = evaluate_candidate(bundle, records, (), device, batch_size)
    a = evaluate_candidate(bundle, records, (expert_a,), device, batch_size)
    b = evaluate_candidate(bundle, records, (expert_b,), device, batch_size)
    pair = evaluate_candidate(bundle, records, (expert_a, expert_b), device, batch_size)
    gain_a = {sid: base[sid] - a[sid] for sid in record_ids}
    gain_b = {sid: base[sid] - b[sid] for sid in record_ids}
    marginal_a = {sid: b[sid] - pair[sid] for sid in record_ids}
    marginal_b = {sid: a[sid] - pair[sid] for sid in record_ids}
    router = ComposeRouter(); load_compose_router_checkpoint(str(snapshot / "router_checkpoint.pt"), router); router.eval()
    queries = _feature_queries(task_root / "features" / "train_features.json")
    routed_a, routed_b = set(), set()
    with torch.inference_mode():
        for record in records:
            sample = _sample_id(record)
            selection = router.select(torch.tensor(queries[sample]).float().unsqueeze(0), router.expert_ids).sets[0]
            if expert_a in selection: routed_a.add(sample)
            if expert_b in selection: routed_b.add(sample)
    key_a = router.key_store.keys[str(expert_a)].detach().float()
    key_b = router.key_store.keys[str(expert_b)].detach().float()
    key_cosine = float(torch.nn.functional.cosine_similarity(key_a, key_b, dim=0))
    ordered = sorted(record_ids)
    correlation = float(np.corrcoef([gain_a[sid] for sid in ordered], [gain_b[sid] for sid in ordered])[0, 1])
    a_own_advantage = mean(a, cluster_b) - mean(a, cluster_a)
    b_own_advantage = mean(b, cluster_a) - mean(b, cluster_b)
    if a_own_advantage > 0 and b_own_advantage > 0:
        label = "SPECIALIZATION_PRESERVED"
    elif correlation >= 0.8:
        label = "SPECIALIZATION_COLLAPSE_RISK"
    else:
        label = "SPECIALIZATION_MIXED"
    row = {
        "task_id": task_id, "task": TASK_NAMES[task_id], "rank": rank,
        "expert_a": expert_a, "expert_b": expert_b,
        "cluster_a_size": len(cluster_a), "cluster_b_size": len(cluster_b),
        "expert_a_own_nll": mean(a, cluster_a), "expert_a_cross_nll": mean(a, cluster_b),
        "expert_b_own_nll": mean(b, cluster_b), "expert_b_cross_nll": mean(b, cluster_a),
        "expert_a_own_advantage": a_own_advantage, "expert_b_own_advantage": b_own_advantage,
        "marginal_a_own": statistics.fmean(marginal_a[sid] for sid in cluster_a),
        "marginal_a_cross": statistics.fmean(marginal_a[sid] for sid in cluster_b),
        "marginal_b_own": statistics.fmean(marginal_b[sid] for sid in cluster_b),
        "marginal_b_cross": statistics.fmean(marginal_b[sid] for sid in cluster_a),
        "routed_sample_jaccard": jaccard(routed_a, routed_b),
        "contribution_overlap_jaccard": jaccard({sid for sid in record_ids if gain_a[sid] > 0}, {sid for sid in record_ids if gain_b[sid] > 0}),
        "key_cosine": key_cosine, "expert_contribution_correlation": correlation,
        "specialization_label": label,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": row, "per_sample": {"base": base, "expert_a": a, "expert_b": b, "pair": pair, "gain_a": gain_a, "gain_b": gain_b}}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True))


def summarize(study: Path, output: Path):
    rows = []
    for task_id in (0, 1, 4):
        for rank in (8, 16, 32):
            payload = json.loads((output / "details" / f"task{task_id}_rank{rank}.json").read_text())
            rows.append(payload["summary"])
    with (output / "F_rank_specialization.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = ["# Experiment F — Rank and Expert Specialization", "", "Own/cross performance is teacher-forced answer NLL on the frozen Experiment-A training boundary.", "", "| task | rank | A own adv. | B own adv. | contribution corr. | key cosine | label |", "|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        report.append("| {task} | {rank} | {expert_a_own_advantage:+.4f} | {expert_b_own_advantage:+.4f} | {expert_contribution_correlation:+.4f} | {key_cosine:+.4f} | {specialization_label} |".format(**row))
    (output / "F_rank_specialization_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"status": "COMPLETE", "rows": len(rows)}))


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run"); one.add_argument("--run-root", required=True); one.add_argument("--task-id", type=int, required=True); one.add_argument("--rank", type=int, required=True); one.add_argument("--output", required=True); one.add_argument("--device", default="cuda:0"); one.add_argument("--batch-size", type=int, default=8)
    summary = sub.add_parser("summarize"); summary.add_argument("--study-root", required=True); summary.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if args.command == "run": run(Path(args.run_root), args.task_id, args.rank, Path(args.output), args.device, args.batch_size)
    else: summarize(Path(args.study_root), Path(args.output_root))


if __name__ == "__main__":
    main()
