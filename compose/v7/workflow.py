"""Full-split preparation and machine-readable V7 workflow helpers."""

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

from compose.data.records import question_text

from .pool import (
    V7ExpertKeyPool,
    initialize_candidate_keys,
    initialize_current_task_key,
    reuse_key_seed,
)


def validate_query_cache_contract(paths, expected_backbone, expected_path):
    contracts = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        provenance = payload.get("query_backbone_provenance") or {}
        if payload.get("query_mode") != "v7_fixed":
            raise ValueError("query cache is not V7 fixed-query data")
        if Path(str(provenance.get("resolved_path", ""))).resolve() != Path(expected_path).resolve():
            raise ValueError("V7 query backbone path mismatch")
        if payload.get("feature_source") != "frozen_clip_l14_336" or expected_backbone != "clip-vit-large-patch14-336":
            raise ValueError("V7 query backbone kind mismatch")
        contracts.append(provenance)
    if any(value != contracts[0] for value in contracts[1:]):
        raise ValueError("V7 query backbone provenance differs across splits")
    return contracts[0]


def write_full_split_with_unique_ids(
    source: str, destination: str, task_index: int, split: str
) -> int:
    records = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("{} split is empty or invalid".format(split))
    for index, record in enumerate(records):
        # A deterministic internal id prevents repeated UCIT question_ids from
        # overwriting feature-cache rows. It does not alter benchmark data.
        source_id = record.get("id", record.get("question_id"))
        canonical_source_record = json.dumps(
            record, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        source_identity = {
            "source_index": index,
            "source_id": None if source_id is None else str(source_id),
            "image": str(record.get("image", "")),
            "question_sha256": hashlib.sha256(
                question_text(record).strip().encode("utf-8")
            ).hexdigest(),
            "source_record_sha256": hashlib.sha256(
                canonical_source_record.encode("utf-8")
            ).hexdigest(),
        }
        record["v7_source_identity"] = source_identity
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
    reusable_historical_ids: Optional[Sequence[int]] = None,
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
    candidate_seed = seed + int(task_index)
    keys, center, audit = initialize_candidate_keys(
        queries, num_train_samples, count=4, perturbation=perturbation,
        seed=candidate_seed,
    )
    candidate_ids = tuple(range(first_id, first_id + 4))
    for expert_id, key in zip(candidate_ids, keys):
        pool.add(expert_id, key, task_index, "current", True, rms_state={})
    # Reusable historical experts get one ADDITIONAL, learnable current-task
    # routing key.  It is produced by the same initializer kernel as the
    # candidates, from the same task center and the same perturbation scale;
    # only the generator stream differs, which is what breaks the symmetry.
    reusable = tuple(sorted(int(value) for value in (reusable_historical_ids or ())))
    historical = set(pool.historical_ids)
    if not set(reusable).issubset(historical):
        raise ValueError("reusable historical experts must belong to the frozen pool")
    if int(task_index) == 0 and reusable:
        raise ValueError("Task0 reusable historical pool must be empty")
    reuse_keys = {}
    for expert_id in reusable:
        key = initialize_current_task_key(
            center, perturbation=perturbation,
            seed=reuse_key_seed(candidate_seed, task_index, expert_id),
        )
        reuse_keys[str(expert_id)] = pool.add_reuse_key(
            expert_id, task_index, key, trainable=True
        )
    audit.update(
        {
            "task_index": int(task_index),
            "candidate_ids": list(candidate_ids),
            "candidate_seed": int(candidate_seed),
            "task_center": center.tolist(),
            "pool_size_before_task": len(pool.historical_ids),
            "initialized_candidate_count": 4,
            "reusable_historical_ids": list(reusable),
            "reuse_key_ids": reuse_keys,
            "reuse_key_initializer": "compose.v7.pool.initialize_current_task_key",
            "reuse_key_perturbation": float(perturbation),
            "historical_lora_trainable": False,
            "historical_canonical_keys_trainable": False,
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
    values = []
    for row in payload.values():
        value = row["global_top2"]
        if isinstance(value, dict):
            value = value["mean_answer_nll"]
        values.append(float(value))
    if not values:
        raise ValueError("NLL output is empty")
    return sum(values) / len(values)

