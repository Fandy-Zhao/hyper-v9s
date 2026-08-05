"""V6 answer-supervised old-expert teacher with Router Top-M retrieval
(Stage E4).

Per train/calibration sample (task t > 1):

  1. Router Top-M retrieval over historical experts;
  2. empty loss;
  3. single losses over the whole retrieved Top-M;
  4. keep the four best singles;
  5. evaluate six pairs;
  6. expert-count penalty  J(S) = answer_loss(S) + lambda_expert * |S|;
  7. emit an empty/single/pair teacher:
       - best expert below the empty gain floor  -> empty
       - best pair below the conditional gain over the best single -> single
       - otherwise -> pair

The retrieved Top-M never decides the teacher by itself; the answer loss
over each candidate set is the authority. Cache keys bind the data hash,
sample id, expert pool version, expert checkpoint hashes, RMS version,
router version, combination rule and loss-mask version.

A separate full-pool recall audit (OracleRecall@M) checks on at least 5%
of train/validation samples whether the Top-M retrieval missed a truly
contributing expert; a low recall is a Router bug to fix, not an excuse to
lower thresholds.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .candidate_search import canonical_expert_set
from .oracle_set import OracleConfig
from .types import AnswerNLL, CandidateScore, stable_hash

MIN_RECALL_AUDIT_RATIO = 0.05


@dataclass(frozen=True)
class V6TeacherRecord:
    """Per-sample teacher record (task book field set)."""

    sample_id: str
    task_id: int
    pool_version: int
    router_version: str
    candidate_experts: Tuple[int, ...]  # the retrieved Top-M for this sample
    empty_loss: float
    single_losses: Mapping[int, float]  # expert_id -> mean answer NLL
    pair_losses: Mapping[Tuple[int, int], float]
    best_single: Tuple[int, ...]
    best_pair: Tuple[int, ...]
    pair_gain: Optional[float]  # raw gain of best pair over best single
    teacher_set: Tuple[int, ...]
    teacher_loss: float
    teacher_multi_hot: Mapping[int, int]  # expert_id -> 1 if in teacher_set
    cache_key: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "task_id": self.task_id,
            "pool_version": self.pool_version,
            "router_version": self.router_version,
            "candidate_experts": list(self.candidate_experts),
            "empty_loss": self.empty_loss,
            "single_losses": {str(key): value for key, value in self.single_losses.items()},
            "pair_losses": {
                "{},{}".format(*key): value for key, value in self.pair_losses.items()
            },
            "best_single": list(self.best_single),
            "best_pair": list(self.best_pair),
            "pair_gain": self.pair_gain,
            "teacher_set": list(self.teacher_set),
            "teacher_loss": self.teacher_loss,
            "teacher_multi_hot": {str(key): value for key, value in self.teacher_multi_hot.items()},
            "cache_key": self.cache_key,
        }


@dataclass
class RecallAuditResult:
    """Full-pool recall audit (OracleRecall@M) on a sample subset."""

    audited_samples: int
    pool_size: int
    top_m: int
    oracles_with_support: int  # samples whose best expert is not the empty set
    recalled: int  # among them, how many had that expert inside Top-M
    oracle_recall_at_m: float
    missed_contributing_expert_ids: List[int]
    sample_ids_audited: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audited_samples": self.audited_samples,
            "pool_size": self.pool_size,
            "top_m": self.top_m,
            "oracles_with_support": self.oracles_with_support,
            "recalled": self.recalled,
            "oracle_recall_at_m": self.oracle_recall_at_m,
            "missed_contributing_expert_ids": self.missed_contributing_expert_ids,
            "sample_ids_audited": self.sample_ids_audited,
        }


class V6TeacherSearcher:
    """Combines Router Top-M retrieval with answer-supervised teacher search."""

    def __init__(
        self,
        config: OracleConfig,
        router_version: str,
        pool_version: int,
        answer_nll_fn: Callable[[Sequence[int]], AnswerNLL],
        top_m: int = 8,
        retrieve_fn: Optional[Callable[[], Tuple[int, ...]]] = None,
    ) -> None:
        """``answer_nll_fn(expert_ids)`` returns the answer NLL under the
        given expert set (empty set = backbone only). ``retrieve_fn``
        returns the Router Top-M for the current sample; when None, the
        caller supplies candidate sets explicitly."""
        if top_m <= 0:
            raise ValueError("top_m must be positive")
        self.config = config
        self.router_version = str(router_version)
        self.pool_version = int(pool_version)
        self.answer_nll_fn = answer_nll_fn
        self.top_m = int(top_m)
        self.retrieve_fn = retrieve_fn

    # ------------------------------------------------------------------

    def _score(self, expert_ids: Sequence[int]) -> CandidateScore:
        nll = self.answer_nll_fn(expert_ids)
        return CandidateScore(
            expert_ids=tuple(sorted(int(value) for value in expert_ids)),
            nll=nll,
            score=nll.mean_nll + self.config.lambda_expert * len(expert_ids),
        )

    def search(
        self,
        sample_id: str,
        task_id: int,
        visible_expert_ids: Sequence[int],
        pool_size: int,
        provenance: Mapping[str, Any],
    ) -> V6TeacherRecord:
        """Run the full teacher search for one sample.

        ``visible_expert_ids`` are the historical experts the router may
        retrieve; ``pool_size`` is the current pool_version size used only
        for the recall audit denominator (pass 0 to skip audits here).
        """
        # 1. Router Top-M retrieval (a recall set, not a teacher set: it may
        # hold more than two experts).
        if self.retrieve_fn is not None:
            retrieved = tuple(
                sorted({int(value) for value in self.retrieve_fn()})
            )
            if not set(retrieved).issubset(set(visible_expert_ids)):
                raise ValueError(
                    "retrieved experts not visible: {}".format(retrieved)
                )
        else:
            retrieved = tuple(sorted({int(value) for value in visible_expert_ids}))
        if len(retrieved) > self.top_m:
            raise ValueError("retrieved set exceeds top_m={}".format(self.top_m))

        # 2-3. empty + all singles over the retrieved set.
        empty = self._score(())
        singles = [self._score((expert_id,)) for expert_id in retrieved]
        singles.sort(key=lambda item: (item.score, item.expert_ids))

        # 4. keep the four best singles.
        best_singles = singles[: self.config.top_k_for_pair]
        best_single = best_singles[0] if best_singles else None

        # 5. evaluate up to six pairs over the best singles.
        pairs = []
        if len(best_singles) >= 2:
            import itertools

            for left, right in itertools.combinations(best_singles, 2):
                pair = self._score(
                    (left.expert_ids[0], right.expert_ids[0])
                )
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

        # 6-7. teacher rule with expert-count penalty:
        #  - best expert below the empty gain floor -> empty
        #  - best pair below the conditional gain over best single -> single
        #  - otherwise -> pair
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
            best_pair.raw_gain_over_best_single
            if best_pair is not None
            else None
        )
        multi_hot = {
            int(expert_id): 1 for expert_id in teacher.expert_ids
        }
        return V6TeacherRecord(
            sample_id=str(sample_id),
            task_id=int(task_id),
            pool_version=self.pool_version,
            router_version=self.router_version,
            candidate_experts=tuple(retrieved),
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


def build_teacher_multi_hot(teacher_set: Sequence[int], all_expert_ids: Sequence[int]) -> Dict[int, int]:
    """Multi-hot label over every pool expert: 1 if in the teacher set."""
    selected = set(int(value) for value in teacher_set)
    return {int(expert_id): (1 if int(expert_id) in selected else 0)
            for expert_id in all_expert_ids}


def run_recall_audit(
    samples: Sequence[Mapping[str, Any]],
    full_pool_eval: Callable[[Mapping[str, Any]], Tuple[int, ...]],
    retrieved_sets: Mapping[str, Tuple[int, ...]],
    top_m: int,
    pool_expert_ids: Sequence[int],
) -> RecallAuditResult:
    """OracleRecall@M audit on a sample subset (>= MIN_RECALL_AUDIT_RATIO).

    ``full_pool_eval(sample)`` returns the best single expert under the
    full pool (or ``()`` when empty is best). ``retrieved_sets`` maps
    sample ids to the Top-M the router returned.
    """
    if len(samples) < 1:
        raise ValueError("recall audit requires at least one sample")
    audited = list(samples)
    pool_ids = set(int(value) for value in pool_expert_ids)
    support_count = 0
    recalled = 0
    missed = []
    audited_ids = []
    for sample in audited:
        sample_id = str(sample["sample_id"])
        audited_ids.append(sample_id)
        best = full_pool_eval(sample)
        best_ids = tuple(sorted(int(value) for value in best))
        if not best_ids:
            continue
        support_count += 1
        retrieved = set(int(value) for value in retrieved_sets.get(sample_id, ()))
        if set(best_ids).issubset(retrieved):
            recalled += 1
        else:
            missed.extend(best_ids)
    return RecallAuditResult(
        audited_samples=len(audited),
        pool_size=len(pool_ids),
        top_m=int(top_m),
        oracles_with_support=support_count,
        recalled=recalled,
        oracle_recall_at_m=(recalled / support_count) if support_count else 0.0,
        missed_contributing_expert_ids=sorted(set(missed)),
        sample_ids_audited=audited_ids,
    )
