"""Few-shot answer teacher screening for a task-level reusable old-expert set.

The functions in this module are deliberately separate from the full-data
trainer.  Ground-truth correctness and answer NLL may enter here; the returned
artifact contains only a frozen set of expert IDs and aggregate diagnostics.

Three things happen here, in order:

1. :func:`sample_teacher_records` draws the seeded, answer-stratified teacher
   subset from the **train** split, recording the sample id hash so the subset is
   reproducible and provably disjoint from test.
2. :func:`aggregate_reusable_experts` collapses the teacher's per-sample
   ``selected_experts`` into the task-level reusable set ``R_t``.  The criterion
   is unchanged from the validated campaign -- selected-solver support **and**
   usage rate -- but the artifact now also reports, per expert,
   ``solved_single_count``: the samples where that expert solved the task *as a
   single* without the teacher having selected it.  Those are alternative
   solvers, and an artifact that only counted selections could not tell a reader
   how many of them were discarded.
3. :func:`initialize_reuse_keys` turns the teacher's own evidence into the
   current-task reuse-key initialisation: the centroid of the queries it selected
   each reusable expert for, falling back to the task center when support is too
   thin.  This is the only place teacher decisions may influence training, and it
   influences **initialisation**, never a loss and never a route.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F

from compose.v8.pool import alias_key_init


SCHEMA_VERSION = 1


def _sample_id(record: Mapping[str, Any], index: int) -> str:
    value = record.get("id", record.get("question_id", index))
    return str(value)


def _answer_stratum(record: Mapping[str, Any]) -> str:
    for key in ("category", "class", "answer_type", "question_type"):
        if record.get(key) is not None:
            return "{}:{}".format(key, record[key])
    for message in reversed(record.get("conversations", ())):
        if str(message.get("from", "")).lower() in ("gpt", "assistant"):
            return "answer:{}".format(str(message.get("value", "")).strip())
    return "all"


def _allocation(bucket_sizes: Mapping[str, int], count: int) -> Dict[str, int]:
    total = sum(bucket_sizes.values())
    exact = {key: count * size / total for key, size in bucket_sizes.items()}
    result = {key: min(size, int(math.floor(exact[key])))
              for key, size in bucket_sizes.items()}
    remaining = count - sum(result.values())
    order = sorted(
        bucket_sizes,
        key=lambda key: (-(exact[key] - result[key]), key),
    )
    for key in order:
        if remaining == 0:
            break
        if result[key] < bucket_sizes[key]:
            result[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("failed to allocate the requested teacher sample count")
    return result


def sample_teacher_records(
    records: Sequence[Mapping[str, Any]],
    *,
    num_samples: Optional[int] = None,
    sample_ratio: Optional[float] = None,
    seed: int = 42,
) -> Tuple[list, Dict[str, Any]]:
    """Return a reproducible, answer-stratified subset of the train split."""
    if (num_samples is None) == (sample_ratio is None):
        raise ValueError("set exactly one of teacher_num_samples or teacher_sample_ratio")
    total = len(records)
    if not total:
        raise ValueError("cannot sample an empty training split")
    if num_samples is None:
        if not 0.0 < float(sample_ratio) <= 1.0:
            raise ValueError("teacher_sample_ratio must be in (0, 1]")
        num_samples = max(1, int(round(total * float(sample_ratio))))
    count = int(num_samples)
    if count < 1 or count > total:
        raise ValueError("teacher_num_samples must be in [1, train_size]")

    buckets: Dict[str, list] = defaultdict(list)
    for index, record in enumerate(records):
        buckets[_answer_stratum(record)].append((_sample_id(record, index), record))
    sizes = {key: len(value) for key, value in buckets.items()}
    allocation = _allocation(sizes, count)
    selected = []
    for key in sorted(buckets):
        values = sorted(buckets[key], key=lambda item: item[0])
        bucket_seed = int(hashlib.sha256(
            "{}:{}".format(seed, key).encode("utf-8")
        ).hexdigest()[:16], 16)
        random.Random(bucket_seed).shuffle(values)
        selected.extend(values[: allocation[key]])
    random.Random(int(seed)).shuffle(selected)
    ids = [sample_id for sample_id, _record in selected]
    if len(ids) != count or len(set(ids)) != count:
        raise AssertionError("teacher sampler produced duplicate or missing IDs")
    report = {
        "schema_version": SCHEMA_VERSION,
        "strategy": "answer_stratified_seeded_random",
        "seed": int(seed),
        "train_samples": total,
        "teacher_samples": count,
        "teacher_sample_ratio": count / total,
        "strata": {key: {"available": sizes[key], "selected": allocation[key]}
                    for key in sorted(sizes)},
        "sample_ids": ids,
        "sample_ids_sha256": hashlib.sha256(
            ("\n".join(sorted(ids)) + "\n").encode("utf-8")
        ).hexdigest(),
    }
    return [record for _sample_id_value, record in selected], report


def _solved_as_single(record: Mapping[str, Any], expert_id: int) -> bool:
    """Did ``expert_id`` meet the task metric on this sample as a *single*?

    Read from ``single_values``, the teacher's full-history single oracle, and
    compared against the same ``solved_threshold`` the teacher's own decision
    used.  This is the alternative-solver signal: an expert can solve a sample
    without having been selected to, and the artifact must be able to say how
    often that happened.
    """
    values = record.get("single_values") or {}
    value = values.get(str(expert_id), values.get(expert_id))
    if value is None:
        return False
    return float(value) >= float(record.get("solved_threshold", 0.0))


def aggregate_reusable_experts(
    teacher_payload: Mapping[str, Any],
    *,
    min_teacher_support: int,
    min_teacher_usage_rate: float,
) -> Dict[str, Any]:
    """Aggregate sample-level valid teacher selections into a task-level set.

    The criterion below is the one the validated campaign used and is unchanged:
    an expert is reusable when the teacher **selected** it for at least
    ``min_teacher_support`` samples and that is at least ``min_teacher_usage_rate``
    of the teacher subset.  What is new is the evidence kept alongside it --
    ``solved_single_count`` in particular, which counts the samples this expert
    solves as a single even when the teacher chose someone else.  It does not
    affect the decision; it makes the cost of the decision measurable.
    """
    if min_teacher_support < 1:
        raise ValueError("min_teacher_support must be positive")
    if not 0.0 <= min_teacher_usage_rate <= 1.0:
        raise ValueError("min_teacher_usage_rate must be in [0, 1]")
    records = list(teacher_payload.get("records", ()))
    if not records:
        raise ValueError("teacher result has no records")
    visible = sorted(int(value) for value in teacher_payload["historical_experts_visible"])
    evidence = {expert_id: {
        "selected_count": 0, "single_usage_count": 0, "pair_usage_count": 0,
        "nll_gains": [], "correct_gains": [],
    } for expert_id in visible}
    for record in records:
        selected = [int(value) for value in record.get("selected_experts", ())]
        if len(selected) not in (0, 1, 2):
            raise ValueError("teacher selection cardinality must be 0, 1, or 2")
        for expert_id in selected:
            if expert_id not in evidence:
                raise ValueError("teacher selected an expert outside the historical pool")
            row = evidence[expert_id]
            row["selected_count"] += 1
            row["single_usage_count" if len(selected) == 1 else "pair_usage_count"] += 1
            if record.get("delta_nll") is not None:
                row["nll_gains"].append(float(record["delta_nll"]))
            if record.get("teacher_gain") is not None:
                row["correct_gains"].append(float(record["teacher_gain"]))

    stats = {}
    reusable = []
    denominator = len(records)
    for expert_id in visible:
        raw = evidence[expert_id]
        count = int(raw["selected_count"])
        rate = count / denominator
        keep = count >= int(min_teacher_support) and rate >= float(min_teacher_usage_rate)
        if keep:
            reusable.append(expert_id)
        solved_singles = [
            record for record in records if _solved_as_single(record, expert_id)
        ]
        alternative_singles = [
            record for record in solved_singles
            if expert_id not in [int(value) for value in record.get("selected_experts", ())]
        ]
        context_count = sum(
            1 for record in records
            if expert_id in [int(value) for value in record.get("residual_context", ())]
        )
        tested_count = sum(
            1 for record in records
            if expert_id in [int(value) for value in record.get("tested_singles", ())]
        )
        stats[str(expert_id)] = {
            "teacher_selected_count": count,
            "teacher_selected_rate": rate,
            "selected_solver_count": count,
            "selected_single_solver_count": int(raw["single_usage_count"]),
            "selected_pair_count": int(raw["pair_usage_count"]),
            "solved_single_count": len(solved_singles),
            "solved_single_rate": len(solved_singles) / denominator,
            "alternative_solved_single_count": len(alternative_singles),
            "context_count": context_count,
            "tested_single_count": tested_count,
            "mean_nll_gain": (statistics.fmean(raw["nll_gains"])
                              if raw["nll_gains"] else None),
            "median_nll_gain": (statistics.median(raw["nll_gains"])
                                if raw["nll_gains"] else None),
            "nll_evidence_count": len(raw["nll_gains"]),
            "correct_gain": sum(raw["correct_gains"]),
            "success_count": count,
            "single_usage_count": int(raw["single_usage_count"]),
            "pair_usage_count": int(raw["pair_usage_count"]),
            "reusable": keep,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": int(teacher_payload["task_id"]),
        "teacher_search_mode": teacher_payload.get("teacher_search_mode"),
        "teacher_sample_count": denominator,
        "historical_expert_ids": visible,
        "reusable_historical_expert_ids": reusable,
        "criterion": {
            "operator": "and",
            "min_teacher_support": int(min_teacher_support),
            "min_teacher_usage_rate": float(min_teacher_usage_rate),
            "evidence": "correctness-valid selected_experts; NLL ranks ties only",
            "diagnostic_only": [
                "solved_single_count",
                "alternative_solved_single_count",
                "context_count",
                "mean_nll_gain",
                "median_nll_gain",
            ],
        },
        "expert_statistics": stats,
        "full_training_oracle_eval_sample_count": 0,
        "historical_keys_trainable": False,
        "historical_lora_trainable": False,
        "task_specific_historical_keys": False,
    }


def teacher_router_recall(
    teacher_payload: Mapping[str, Any],
    screening: Mapping[str, Any],
    query_payload: Mapping[str, Any],
    key_state: Mapping[str, Any],
) -> Dict[str, Any]:
    """Measure teacher-selected expert recall using frozen keys inside R_t."""
    reusable = [int(value) for value in screening["reusable_historical_expert_ids"]]
    records = [record for record in teacher_payload["records"]
               if record.get("selected_experts")]
    if not records or not reusable:
        return {"denominator": len(records), "TeacherRouterRecall@1": 0.0,
                "TeacherRouterRecall@2": 0.0}
    query_ids = [str(value) for value in query_payload["sample_ids"]]
    rows = {sample_id: index for index, sample_id in enumerate(query_ids)}
    keys = key_state["keys"]
    key_matrix = torch.stack([F.normalize(keys[str(value)].float(), dim=0)
                              for value in reusable])
    hits = {1: 0, 2: 0}
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id not in rows:
            raise ValueError("teacher sample missing from query cache: {}".format(sample_id))
        query = F.normalize(query_payload["queries"][rows[sample_id]].float(), dim=0)
        order = torch.argsort(query @ key_matrix.T, descending=True, stable=True).tolist()
        ranked = [reusable[index] for index in order]
        selected = {int(value) for value in record["selected_experts"]}
        for k in (1, 2):
            hits[k] += int(bool(selected & set(ranked[:k])))
    return {
        "denominator": len(records),
        "TeacherRouterRecall@1": hits[1] / len(records),
        "TeacherRouterRecall@2": hits[2] / len(records),
    }


def initialize_reuse_keys(
    teacher_payload: Mapping[str, Any],
    reusable_ids: Sequence[int],
    query_payload: Mapping[str, Any],
    *,
    center: torch.Tensor,
    perturbation: float,
    min_support: int,
    seed: int,
    task_index: int,
) -> Tuple[Dict[int, torch.Tensor], Dict[str, Any]]:
    """Current-task reuse keys for ``R_t``, initialised from teacher evidence.

    For each reusable historical expert the support queries are the samples the
    teacher **selected it as a solver** for -- the most conservative definition
    available, and the same evidence the reusable set itself was built from.  The
    key starts at their normalised centroid and then takes the same tiny
    deterministic tangential perturbation the candidate initialiser uses, so two
    experts with identical support do not start from an identical key.

    When support is below ``min_support`` the centroid is not evidence, so the
    key falls back to the task center with the same deterministic perturbation
    the previous implementation always used.  Both branches are recorded.

    Returns ``(keys_by_expert, audit)``; the audit is what lands in the screening
    artifact, so a reader can tell evidence-driven from fallback keys without
    re-deriving anything.
    """
    from compose.v7.pool import initialize_current_task_key, reuse_key_seed

    if min_support < 1:
        raise ValueError("min_support must be positive")
    if perturbation <= 0:
        raise ValueError("perturbation must be positive")
    records = list(teacher_payload.get("records", ()))
    query_ids = [str(value) for value in query_payload["sample_ids"]]
    rows = {sample_id: index for index, sample_id in enumerate(query_ids)}
    queries = query_payload["queries"]
    if int(queries.shape[0]) != len(query_ids):
        raise ValueError("query payload and sample id list disagree on length")

    keys: Dict[int, torch.Tensor] = {}
    audit: Dict[str, Any] = {}
    for expert_id in sorted(int(value) for value in reusable_ids):
        support_ids = sorted(
            str(record["sample_id"]) for record in records
            if expert_id in [int(value) for value in record.get("selected_experts", ())]
        )
        missing = [sample_id for sample_id in support_ids if sample_id not in rows]
        if missing:
            raise ValueError(
                "teacher support sample missing from the query cache: {}".format(missing[:4])
            )
        if len(support_ids) >= int(min_support):
            rows_index = torch.tensor(
                [rows[sample_id] for sample_id in support_ids], dtype=torch.long
            )
            centroid = alias_key_init(queries[rows_index].detach().float())
            key = initialize_current_task_key(
                centroid, perturbation=perturbation,
                seed=reuse_key_seed(seed, task_index, expert_id),
            )
            source = "teacher_selected_query_centroid"
        else:
            key = initialize_current_task_key(
                center, perturbation=perturbation,
                seed=reuse_key_seed(seed, task_index, expert_id),
            )
            source = "task_center_fallback"
        keys[expert_id] = key
        audit[str(expert_id)] = {
            "expert_id": expert_id,
            "source": source,
            "support": len(support_ids),
            "min_support": int(min_support),
            "query_ids_sha256": hashlib.sha256(
                ("\n".join(support_ids) + "\n").encode("utf-8")
            ).hexdigest(),
            "perturbation": float(perturbation),
            "seed": reuse_key_seed(seed, task_index, expert_id),
        }
    return keys, {
        "schema_version": SCHEMA_VERSION,
        "task_id": int(teacher_payload.get("task_id", task_index)),
        "support_definition": "teacher_selected_solver",
        "perturbation": float(perturbation),
        "min_support": int(min_support),
        "seed": int(seed),
        "experts": audit,
    }


def load_reusable_screening(
    path: str | Path,
    *,
    expected_task: int,
    historical_ids: Sequence[int],
) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported reusable-screening schema")
    if int(payload.get("task_id", -1)) != int(expected_task):
        raise ValueError("reusable-screening artifact belongs to a different task")
    known = {int(value) for value in historical_ids}
    declared = {int(value) for value in payload.get("historical_expert_ids", ())}
    reusable = {int(value) for value in payload.get("reusable_historical_expert_ids", ())}
    if declared != known:
        raise ValueError("screening historical pool does not match the training checkpoint")
    if not reusable.issubset(known):
        raise ValueError("reusable experts must be a subset of historical experts")
    if payload.get("full_training_oracle_eval_sample_count") != 0:
        raise ValueError("full training artifact must declare zero oracle evaluations")
    if payload.get("historical_keys_trainable") or payload.get("historical_lora_trainable"):
        raise ValueError("screening artifact attempts to train historical parameters")
    if payload.get("task_specific_historical_keys"):
        raise ValueError("task-specific historical keys are forbidden in this experiment")
    return payload
