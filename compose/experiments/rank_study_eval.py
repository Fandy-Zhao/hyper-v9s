"""Evaluation helpers for the controlled seed42 rank study.

This module is additive: it does not alter the formal evaluator.  It computes
teacher-forced NLL for the exact frozen Compose route (or one fixed selection)
over a saved train/validation boundary.
"""

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

from compose.eval.load_compose import load_compose_model
from compose.lora.rms import apply_kappa_calibration
from compose.oracle.candidate_sets import CandidateSet
from compose.oracle.evaluator import _candidate_nll, _collate, _prepare_multimodal_batch
from compose.router.router import ComposeRouter, load_compose_router_checkpoint


BASE_MODEL = "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
VISION_TOWER = "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
PROJECTOR = BASE_MODEL + "/mm_projector.bin"
IMAGES = "/data/dataset/zhaozhuofan/UCIT/datasets"


def _sample_id(record):
    return str(record.get("id", record.get("question_id")))


def _snapshot(run_root: Path, task_id: int):
    directory = run_root / f"task{task_id}" / "snapshots" / f"task{task_id}"
    manifest = json.loads((directory / "manifest.json").read_text())
    pool = Path(manifest["pool_checkpoint_dir"])
    if not pool.is_dir():
        raise FileNotFoundError(pool)
    return directory, pool


def _feature_queries(path: Path):
    payload = json.loads(path.read_text())
    return {str(key): value["query"] for key, value in payload["records"].items()}


def _route_map(router_path: Path, queries, records):
    router = ComposeRouter()
    load_compose_router_checkpoint(str(router_path), router)
    router.eval()
    result = {}
    with torch.inference_mode():
        for record in records:
            sid = _sample_id(record)
            query = torch.tensor(queries[sid], dtype=torch.float32).unsqueeze(0)
            result[sid] = tuple(router.select(query, router.expert_ids).sets[0])
    return result, list(router.expert_ids)


def routed_nll(args):
    run_root = Path(args.run_root)
    task_root = run_root / f"task{args.task_id}"
    snapshot, pool = _snapshot(run_root, args.task_id)
    records_path = task_root / "data" / f"teacher_{args.split}.json"
    features_path = task_root / "features" / f"{args.split}_features.json"
    records = json.loads(records_path.read_text())
    if args.max_samples:
        records = records[: args.max_samples]
    queries = _feature_queries(features_path)
    if args.expert_ids is None:
        routes, visible = _route_map(snapshot / "router_checkpoint.pt", queries, records)
        mode = "actual_route"
    else:
        fixed = tuple(int(value) for value in args.expert_ids.split(",") if value)
        routes = {_sample_id(record): fixed for record in records}
        visible = sorted(set(fixed))
        mode = "fixed:" + ",".join(map(str, fixed))

    bundle = load_compose_model(
        model_path=BASE_MODEL, checkpoint_dir=str(pool), vision_tower=VISION_TOWER,
        projector_path=PROJECTOR, expert_id=None, device=args.device,
        dtype=torch.bfloat16, model_max_length=2048,
    )
    calibration = bundle.load_summary.get("rms_calibration")
    if calibration:
        apply_kappa_calibration(bundle.model, calibration)
    groups = defaultdict(list)
    for record in records:
        groups[routes[_sample_id(record)]].append(record)
    per_sample = {}
    token_counts = {}
    started = time.perf_counter()
    for ids, group in sorted(groups.items()):
        candidate = CandidateSet(0, ids, tuple(1.0 for _ in ids), "none")
        for offset in range(0, len(group), args.batch_size):
            batch = group[offset : offset + args.batch_size]
            raw = _collate(batch, bundle, IMAGES, args.device)
            prepared = _prepare_multimodal_batch(bundle, raw)
            losses, counts = _candidate_nll(bundle, candidate, prepared)
            for index, record in enumerate(batch):
                sid = _sample_id(record)
                per_sample[sid] = float(losses[index])
                token_counts[sid] = int(counts[index])
    elapsed = time.perf_counter() - started
    values = list(per_sample.values())
    result = {
        "schema_version": 1,
        "run_root": str(run_root),
        "task_id": args.task_id,
        "split": args.split,
        "selection_mode": mode,
        "visible_expert_ids": visible,
        "route_histogram": {"|".join(map(str, key)): len(value) for key, value in groups.items()},
        "samples": len(values),
        "mean_nll": statistics.fmean(values),
        "median_nll": statistics.median(values),
        "nll_samples_per_second": len(values) / elapsed,
        "duration_seconds": elapsed,
        "per_sample_nll": per_sample,
        "target_token_counts": token_counts,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("per_sample_nll", "target_token_counts")}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("nll")
    command.add_argument("--run-root", required=True)
    command.add_argument("--task-id", type=int, required=True)
    command.add_argument("--split", choices=("train", "val"), required=True)
    command.add_argument("--expert-ids")
    command.add_argument("--output", required=True)
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--batch-size", type=int, default=8)
    command.add_argument("--max-samples", type=int)
    command.set_defaults(func=routed_nll)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
