"""Full-split preparation and machine-readable V7 workflow helpers."""

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import torch

from .pool import V7ExpertKeyPool, initialize_candidate_keys


def write_full_split_with_unique_ids(
    source: str, destination: str, task_index: int, split: str
) -> int:
    records = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("{} split is empty or invalid".format(split))
    for index, record in enumerate(records):
        # A deterministic internal id prevents repeated UCIT question_ids from
        # overwriting feature-cache rows. It does not alter benchmark data.
        record["id"] = "v7_t{}_{}_{}".format(int(task_index), split, index)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return len(records)


def queries_from_cache(path: str, expected_count: int) -> Tuple[torch.Tensor, Tuple[str, ...]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("query_mode") != "v7_fixed":
        raise ValueError("feature cache is not V7 fixed-query data")
    records = payload["records"]
    if len(records) != int(expected_count):
        raise ValueError("full-split query count mismatch")
    sample_ids = tuple(sorted(records))
    queries = torch.tensor([records[value]["query"] for value in sample_ids])
    if queries.shape != (expected_count, 1536):
        raise ValueError("V7 query cache must have shape [N,1536]")
    return queries, sample_ids


def prepare_candidate_pool(
    train_query_cache: str,
    num_train_samples: int,
    task_index: int,
    seed: int,
    perturbation: float,
    previous_key_state: str = None,
) -> Tuple[V7ExpertKeyPool, torch.Tensor, Dict[str, object]]:
    queries, _ = queries_from_cache(train_query_cache, num_train_samples)
    if previous_key_state:
        state = torch.load(previous_key_state, map_location="cpu", weights_only=False)
        pool = V7ExpertKeyPool.from_state(state)
        if pool.current_ids:
            raise ValueError("previous task checkpoint still contains current candidates")
    else:
        pool = V7ExpertKeyPool()
    first_id = max(pool.expert_ids, default=-1) + 1
    keys, center, audit = initialize_candidate_keys(
        queries, num_train_samples, count=4, perturbation=perturbation,
        seed=seed + int(task_index),
    )
    candidate_ids = tuple(range(first_id, first_id + 4))
    for expert_id, key in zip(candidate_ids, keys):
        pool.add(expert_id, key, task_index, "current", True, rms_state={})
    audit.update(
        {
            "task_index": int(task_index),
            "candidate_ids": list(candidate_ids),
            "task_center": center.tolist(),
            "pool_size_before_task": len(pool.historical_ids),
            "initialized_candidate_count": 4,
        }
    )
    return pool, center, audit


def route_manifest(sample_ids: Sequence[str], selected_ids: torch.Tensor) -> Dict[str, object]:
    if len(sample_ids) != selected_ids.shape[0]:
        raise ValueError("route manifest samples and routes do not align")
    return {
        str(sample_id): {"global_top2": [int(value) for value in row]}
        for sample_id, row in zip(sample_ids, selected_ids.detach().cpu().tolist())
    }


def mean_nll(path: str) -> float:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = [float(row["global_top2"]) for row in payload.values()]
    if not values:
        raise ValueError("NLL output is empty")
    return sum(values) / len(values)

