import statistics
from typing import Dict, Iterable, List, Sequence

from .candidate_sets import CandidateSet


def _best(indices: Iterable[int], losses: Sequence[float]) -> int:
    values = list(indices)
    if not values:
        raise ValueError("oracle category has no candidates")
    return min(values, key=lambda index: (float(losses[index]), index))


def compute_oracle_record(
    losses: Sequence[float], candidates: Sequence[CandidateSet]
) -> Dict[str, object]:
    if len(losses) != len(candidates):
        raise ValueError("loss count must match candidate count")
    empty = [candidate.index for candidate in candidates if not candidate.expert_ids]
    singles = [candidate.index for candidate in candidates if len(candidate.expert_ids) == 1]
    pairs = [candidate.index for candidate in candidates if len(candidate.expert_ids) == 2]
    best_empty = _best(empty, losses)
    best_single = _best(singles, losses)
    best_pair = _best(pairs, losses)
    best_overall = _best(range(len(candidates)), losses)
    best_single_loss = float(losses[best_single])
    best_pair_loss = float(losses[best_pair])
    return {
        "best_empty_index": best_empty,
        "best_single_index": best_single,
        "best_pair_index": best_pair,
        "best_overall_index": best_overall,
        "best_single_loss": best_single_loss,
        "best_pair_loss": best_pair_loss,
        "synergy": best_single_loss - best_pair_loss,
        "pair_oracle": len(candidates[best_overall].expert_ids) == 2,
    }


def select_with_size_penalty(
    losses: Sequence[float], candidates: Sequence[CandidateSet], lambda_size: float
) -> int:
    return min(
        range(len(candidates)),
        key=lambda index: (
            float(losses[index]) + float(lambda_size) * len(candidates[index].expert_ids),
            index,
        ),
    )


def summarize_oracle_records(records: Iterable[Dict[str, object]]) -> Dict[str, object]:
    rows = list(records)
    if not rows:
        raise ValueError("oracle summary requires at least one record")
    synergies = [float(row["synergy"]) for row in rows]
    return {
        "samples": len(rows),
        "pair_oracle_rate": sum(bool(row["pair_oracle"]) for row in rows) / len(rows),
        "mean_synergy": statistics.fmean(synergies),
        "median_synergy": statistics.median(synergies),
        "positive_synergy_rate": sum(value > 0.0 for value in synergies) / len(rows),
    }
