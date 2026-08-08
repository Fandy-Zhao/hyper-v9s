"""Answer-supervised old-expert teacher search over per-sample Top-M.

Per training sample i (task t > 0):

  1. the functional query q_i retrieves the sample's OWN historical
     Top-M experts (batch x M; every sample has its own candidate set);
  2. empty loss (backbone only);
  3. answer NLL for every single expert in the Top-M;
  4. keep the best ``top_k_for_pair`` singles;
  5. search pairs among those best singles;
  6. expert-count penalty J(S) = answer_nll(S) + lambda_expert * |S|;
  7. the final teacher set is {}, {Ea} or {Ea, Eb}; a pair must show a
     real conditional gain over its best single (raw gain >=
     delta_pair_raw AND penalized gain > 0) — an extra LoRA never
     auto-joins the teacher set.

Teacher search uses answers during training only. Test-time inference
never consults teacher records, residual labels or cluster labels.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .candidate_search import canonical_expert_set
from .oracle_set import OracleConfig
from .types import AnswerNLL, CandidateScore, stable_hash

MIN_RECALL_AUDIT_RATIO = 0.05


@dataclass(frozen=True)
class ComposeTeacherRecord:
    """Per-sample teacher record (empty/single/pair)."""

    sample_id: str
    task_id: int
    pool_version: int
    router_version: str
    candidate_experts: Tuple[int, ...]  # the sample's retrieved Top-M
    empty_loss: float
    single_losses: Mapping[int, float]
    pair_losses: Mapping[Tuple[int, int], float]
    best_single: Tuple[int, ...]
    best_pair: Tuple[int, ...]
    pair_gain: Optional[float]
    teacher_set: Tuple[int, ...]
    teacher_loss: float
    teacher_multi_hot: Mapping[int, int]
    cache_key: str
    retrieved_top_m: Tuple[int, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "task_id": self.task_id,
            "pool_version": self.pool_version,
            "router_version": self.router_version,
            "candidate_experts": list(self.candidate_experts),
            "retrieved_top_m": list(self.retrieved_top_m),
            "empty_loss": self.empty_loss,
            "single_losses": {
                str(key): value for key, value in self.single_losses.items()
            },
            "pair_losses": {
                "{},{}".format(*key): value
                for key, value in self.pair_losses.items()
            },
            "best_single": list(self.best_single),
            "best_pair": list(self.best_pair),
            "pair_gain": self.pair_gain,
            "teacher_set": list(self.teacher_set),
            "teacher_loss": self.teacher_loss,
            "teacher_multi_hot": {
                str(key): value for key, value in self.teacher_multi_hot.items()
            },
            "cache_key": self.cache_key,
        }


class ComposeTeacherSearcher:
    """Per-sample Top-M teacher search (answer-supervised)."""

    def __init__(
        self,
        config: OracleConfig,
        router_version: str,
        pool_version: int,
        answer_nll_fn: Optional[Any] = None,
        top_m: int = 8,
    ) -> None:
        """``answer_nll_fn(expert_ids)`` returns the answer NLL under the
        given expert set (empty set = backbone only). When None, the caller
        supplies per-set NLL values directly via ``search_from_nll``."""
        if top_m <= 0:
            raise ValueError("top_m must be positive")
        self.config = config
        self.router_version = str(router_version)
        self.pool_version = int(pool_version)
        self.answer_nll_fn = answer_nll_fn
        self.top_m = int(top_m)

    # ------------------------------------------------------------------

    def _score(self, expert_ids: Sequence[int], nll: Optional[float] = None) -> CandidateScore:
        if nll is None:
            if self.answer_nll_fn is None:
                raise ValueError("no answer_nll_fn available")
            nll = self.answer_nll_fn(expert_ids)
            mean_nll = nll.mean_nll
        else:
            mean_nll = float(nll)
        if isinstance(nll, AnswerNLL):
            nll_value = nll
        else:
            # Aggregate NLL (no token decomposition): a degenerate
            # single-token answer keeps sum == mean consistent.
            nll_value = AnswerNLL(
                sum_nll=mean_nll, mean_nll=mean_nll, token_count=1
            )
        return CandidateScore(
            expert_ids=tuple(sorted(int(value) for value in expert_ids)),
            nll=nll_value,
            score=mean_nll + self.config.lambda_expert * len(expert_ids),
        )

    def search_from_nll(
        self,
        sample_id: str,
        task_id: int,
        retrieved_top_m: Sequence[int],
        nll_by_set: Mapping[Tuple[int, ...], float],
        pool_size: int,
        provenance: Mapping[str, Any],
    ) -> ComposeTeacherRecord:
        """Run the teacher search from precomputed per-set NLL values.

        ``nll_by_set`` must contain the empty set and every single/pair
        over the sample's own Top-M that the caller chose to evaluate.
        """
        retrieved = tuple(
            sorted({int(value) for value in retrieved_top_m})
        )
        if len(retrieved) > self.top_m:
            raise ValueError("retrieved set exceeds top_m={}".format(self.top_m))
        # Fall back to the full visible pool when no Top-M was retrieved.
        if not retrieved:
            singles_available = sorted(
                {
                    tuple(sorted(ids))
                    for ids in nll_by_set
                    if len(ids) == 1
                }
            )
            retrieved = tuple(
                int(ids[0]) for ids in singles_available[: self.top_m]
            )

        empty_key = ()
        if empty_key not in nll_by_set:
            raise ValueError("nll_by_set must include the empty set")
        empty = self._score((), nll=nll_by_set[empty_key])
        singles = []
        for expert_id in retrieved:
            key = (expert_id,)
            if key not in nll_by_set:
                continue
            singles.append(self._score(key, nll=nll_by_set[key]))
        singles.sort(key=lambda item: (item.score, item.expert_ids))
        best_singles = singles[: self.config.top_k_for_pair]
        best_single = best_singles[0] if best_singles else None

        pairs = []
        if len(best_singles) >= 2:
            import itertools

            for left, right in itertools.combinations(best_singles, 2):
                pair_key = tuple(
                    sorted((left.expert_ids[0], right.expert_ids[0]))
                )
                if pair_key not in nll_by_set:
                    continue
                pair = self._score(pair_key, nll=nll_by_set[pair_key])
                pair.raw_gain_over_best_single = (
                    best_single.nll.mean_nll - pair.nll.mean_nll
                )
                pair.penalized_gain_over_best_single = (
                    best_single.score - pair.score
                )
                pair.valid_pair = (
                    pair.raw_gain_over_best_single >= self.config.delta_pair_raw
                    and pair.penalized_gain_over_best_single > 0
                )
                pairs.append(pair)
                if len(pairs) >= self.config.max_pairs:
                    break
        pairs.sort(key=lambda item: (item.score, item.expert_ids))
        best_pair = pairs[0] if pairs else None

        if best_single is None or best_single.score >= empty.score:
            teacher = empty
            reason = "no_single_gain_over_empty"
        elif best_pair is None or not best_pair.valid_pair:
            teacher = best_single
            reason = "no_pair_conditional_gain"
        else:
            teacher = best_pair
            reason = "pair_conditional_gain"

        pair_gain = (
            best_pair.raw_gain_over_best_single if best_pair is not None else None
        )
        multi_hot = {int(expert_id): 1 for expert_id in teacher.expert_ids}
        return ComposeTeacherRecord(
            sample_id=str(sample_id),
            task_id=int(task_id),
            pool_version=self.pool_version,
            router_version=self.router_version,
            candidate_experts=retrieved,
            retrieved_top_m=retrieved,
            empty_loss=empty.nll.mean_nll,
            single_losses={
                int(expert_id): score.nll.mean_nll
                for score in singles
                for expert_id in score.expert_ids
            },
            pair_losses={
                (int(left), int(right)): score.nll.mean_nll
                for score in pairs
                for left, right in [score.expert_ids]
            },
            best_single=best_single.expert_ids if best_single else (),
            best_pair=best_pair.expert_ids if best_pair else (),
            pair_gain=float(pair_gain) if pair_gain is not None else None,
            teacher_set=teacher.expert_ids,
            teacher_loss=teacher.nll.mean_nll,
            teacher_multi_hot=multi_hot,
            cache_key=stable_hash(
                dict(provenance) | {"sample_id": str(sample_id), "reason": reason}
            ),
        )

    def search(
        self,
        sample_id: str,
        task_id: int,
        retrieved_top_m: Sequence[int],
        nll_by_set: Mapping[Tuple[int, ...], float],
        pool_size: int,
        provenance: Mapping[str, Any],
    ) -> ComposeTeacherRecord:
        """Alias for ``search_from_nll`` (backward-friendly entry)."""
        return self.search_from_nll(
            sample_id, task_id, retrieved_top_m, nll_by_set, pool_size, provenance
        )


def build_teacher_multi_hot(
    teacher_set: Sequence[int], all_expert_ids: Sequence[int]
) -> Dict[int, int]:
    """Multi-hot label over every pool expert: 1 if in the teacher set."""
    selected = set(int(value) for value in teacher_set)
    return {
        int(expert_id): (1 if int(expert_id) in selected else 0)
        for expert_id in all_expert_ids
    }


def run_recall_audit(
    samples: Sequence[Mapping[str, Any]],
    full_pool_eval: Any,
    retrieved_sets: Mapping[str, Tuple[int, ...]],
    top_m: int,
    pool_expert_ids: Sequence[int],
) -> Dict[str, Any]:
    """OracleRecall@M audit: whether each sample's per-sample Top-M covered
    the answer-best single expert (diagnostics only; never gates anything)."""
    from compose.router.metrics import retrieval_metrics

    predictions = []
    targets = []
    audited_ids = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        best = full_pool_eval(sample)
        best_ids = tuple(sorted(int(value) for value in best))
        retrieved = tuple(
            int(value) for value in retrieved_sets.get(sample_id, ())
        )
        audited_ids.append(sample_id)
        predictions.append(retrieved)
        targets.append(best_ids)
    metrics = retrieval_metrics(predictions, targets, ks=(1, min(top_m, 8)))
    metrics["audited_samples"] = len(samples)
    metrics["pool_size"] = len(set(int(value) for value in pool_expert_ids))
    metrics["top_m"] = int(top_m)
    metrics["sample_ids_audited"] = audited_ids
    return metrics
