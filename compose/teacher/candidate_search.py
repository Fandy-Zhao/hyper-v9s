"""Deterministic empty/single/pair candidate generation."""

import itertools
from typing import Dict, Iterable, List, Sequence, Tuple

from compose.experts import ExpertStatus


def canonical_expert_set(values: Iterable[int]) -> Tuple[int, ...]:
    ordered = tuple(sorted(int(value) for value in values))
    if len(ordered) != len(set(ordered)):
        raise ValueError("candidate expert IDs must be unique")
    if len(ordered) > 2:
        raise ValueError("Stage 04 candidates support at most two experts")
    return ordered


def temporal_expert_ids(registry, task_id: int, temporal_scope: str, physically_available: Iterable[int]) -> Tuple[int, ...]:
    if temporal_scope not in ("historical_only", "post_task_diagnostic"):
        raise ValueError("unknown temporal scope: {}".format(temporal_scope))
    cutoff = int(task_id) - 1 if temporal_scope == "historical_only" else int(task_id)
    physical = set(int(value) for value in physically_available)
    result = []
    for metadata in registry.list_all():
        if metadata.expert_id not in physical:
            continue
        if metadata.status is ExpertStatus.ARCHIVED:
            continue
        if metadata.creation_task is None or int(metadata.creation_task) > cutoff:
            continue
        if not metadata.checkpoint_path or not metadata.checkpoint_sha256:
            raise ValueError("expert {} has no verified checkpoint".format(metadata.expert_id))
        result.append(metadata.expert_id)
    return tuple(sorted(result))


def empty_and_singles(expert_ids: Iterable[int]) -> Tuple[Tuple[int, ...], ...]:
    values = tuple(sorted(set(int(value) for value in expert_ids)))
    return ((),) + tuple((value,) for value in values)


def pair_candidates(
    expert_ids: Iterable[int],
    single_nll: Dict[int, float],
    *,
    top_k_for_pair: int = 4,
    max_pairs: int = 6,
) -> Tuple[Tuple[int, int], ...]:
    values = tuple(sorted(set(int(value) for value in expert_ids)))
    missing = sorted(set(values) - set(single_nll))
    if missing:
        raise KeyError("single NLL is missing experts {}".format(missing))
    ranked = sorted(values, key=lambda value: (float(single_nll[value]), value))
    pool = values if len(values) <= 4 else tuple(sorted(ranked[:top_k_for_pair]))
    pairs = tuple(itertools.combinations(pool, 2))
    return pairs[:max_pairs]
