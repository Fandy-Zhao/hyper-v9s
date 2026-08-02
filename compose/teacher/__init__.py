"""Answer-supervised Oracle expert-set teacher (training/validation only)."""

from .cache import CACHE_SCHEMA_VERSION, cache_key, load_shard, merge_shards, resume_sample_ids, write_shard
from .candidate_search import canonical_expert_set, empty_and_singles, pair_candidates, temporal_expert_ids
from .metrics import summarize_oracles
from .oracle_set import score_candidate, select_oracle_set
from .scorer import answer_token_nll
from .types import AnswerNLL, CandidateScore, OracleConfig, OracleRecord, stable_hash
from .validity import assert_temporal_boundary, validate_cache_provenance, validate_oracle_split

__all__ = [
    "AnswerNLL", "CACHE_SCHEMA_VERSION", "CandidateScore", "OracleConfig", "OracleRecord",
    "answer_token_nll", "assert_temporal_boundary", "cache_key", "canonical_expert_set",
    "empty_and_singles", "load_shard", "merge_shards", "pair_candidates", "resume_sample_ids",
    "score_candidate", "select_oracle_set", "stable_hash", "summarize_oracles",
    "temporal_expert_ids", "validate_cache_provenance", "validate_oracle_split", "write_shard",
]
