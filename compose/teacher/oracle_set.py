"""Complexity-penalized Oracle expert-set selection."""

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .candidate_search import canonical_expert_set
from .types import AnswerNLL, CandidateScore, OracleConfig, OracleRecord


def _key(value: CandidateScore):
    return (float(value.score), len(value.expert_ids), value.expert_ids)


def _nll_key(value: CandidateScore):
    return (float(value.nll.mean_nll), len(value.expert_ids), value.expert_ids)


def score_candidate(expert_ids: Iterable[int], nll: AnswerNLL, config: OracleConfig) -> CandidateScore:
    ids = canonical_expert_set(expert_ids)
    return CandidateScore(ids, nll, nll.mean_nll + config.lambda_expert * len(ids))


def select_oracle_set(
    *,
    sample_id: str,
    task_id: int,
    task_name: str,
    split: str,
    candidate_expert_ids: Sequence[int],
    nll_by_set: Mapping[Tuple[int, ...], AnswerNLL],
    config: OracleConfig,
    config_hash: str,
    expert_pool_hash: str,
    dataset_manifest_hash: str,
    temporal_scope: str,
    cache_key: Optional[str] = None,
    diagnostics: Optional[Dict[str, object]] = None,
) -> OracleRecord:
    normalized = {canonical_expert_set(ids): nll for ids, nll in nll_by_set.items()}
    if () not in normalized:
        raise ValueError("Oracle scoring requires the empty candidate")
    empty = score_candidate((), normalized[()], config)
    singles = tuple(sorted(
        (score_candidate(ids, nll, config) for ids, nll in normalized.items() if len(ids) == 1),
        key=_key,
    ))
    pairs = tuple(sorted(
        (score_candidate(ids, nll, config) for ids, nll in normalized.items() if len(ids) == 2),
        key=_key,
    ))
    best_single = min(singles, key=_nll_key) if singles else None
    audited_pairs = []
    for pair in pairs:
        if best_single is None:
            pair.raw_gain_over_best_single = None
            pair.penalized_gain_over_best_single = None
            pair.valid_pair = False
        else:
            pair.raw_gain_over_best_single = best_single.nll.mean_nll - pair.nll.mean_nll
            pair.penalized_gain_over_best_single = best_single.score - pair.score
            pair.valid_pair = (
                pair.raw_gain_over_best_single >= config.delta_pair_raw
                and pair.penalized_gain_over_best_single > 0.0
            )
        audited_pairs.append(pair)
    pairs = tuple(audited_pairs)
    best_pair = min(pairs, key=_nll_key) if pairs else None
    eligible = [empty, *singles, *(pair for pair in pairs if pair.valid_pair)]
    selected = min(eligible, key=_key)
    counts = {value.token_count for value in normalized.values()}
    if len(counts) != 1:
        raise ValueError("answer token count changed across candidate sets")
    return OracleRecord(
        sample_id=str(sample_id), task_id=int(task_id), task_name=str(task_name), split=str(split),
        answer_token_count=next(iter(counts)), candidate_expert_ids=tuple(sorted(map(int, candidate_expert_ids))),
        composition_mode=config.composition_mode, empty=empty, singles=singles, pairs=pairs,
        best_single=best_single, best_pair=best_pair, selected=selected,
        config_hash=str(config_hash), expert_pool_hash=str(expert_pool_hash),
        dataset_manifest_hash=str(dataset_manifest_hash), temporal_scope=str(temporal_scope),
        cache_key=cache_key, diagnostics=dict(diagnostics or {}),
    )
