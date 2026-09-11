"""Answer-Supervised Expert Teacher: a Capability Discovery Oracle.

The teacher answers one question per current-task sample: **is there already an
expert in the pool that solves this sample?**  Only the task metric may answer
it.  Answer NLL is computed and recorded, but strictly in a supporting role --
ranking among already-solved experts, soft confidence, pair marginal evidence,
residual context and tie-breaks.  There is no code path in this module where a
loss value can flip ``solved`` from False to True; ``V8TeacherConfig`` rejects
the configuration that would allow it.

**The oracle is decoupled from the router.**  Training and inference ask
different questions and must not share an answer: inference asks *which expert
does the router recall*, training asks *which expert has the capability*.  If
key recall bounded the capability search, a capable expert that the origin keys
rank badly would never be scored, would never become a solver, and would never
earn the alias key that fixes its ranking -- the deficit would be invisible and
self-confirming.  So the single search is **total over the visible historical
pool** and key recall is retained only as a diagnostic.

Per-sample flow:

  STEP A  base alone -> if it solves the sample, ``BaseOnly``, selection empty,
          stop.  No expert is scored, so no expert is marked negative: absence
          of evidence is recorded as ``ignore``.
  STEP B  compute the router's Top-M by key similarity -- **diagnostic only**,
          recorded as ``router_top_m`` and read by no decision below.
  STEP C  score **every visible historical expert** as a single with the
          **metric**.  ``historical_experts_tested`` must equal
          ``historical_experts_visible`` on every base-unsolved sample; the run
          hard-fails otherwise.
            * any solved single  -> ``Reuse1``, take the lexicographic best
              ``(-metric, nll, expert_id)``, and STOP.  A lower-NLL pair is
              never allowed to override a solved single.
            * none solved        -> pairs over a shortlist cut from those
              *scored* singles, ranked ``(-metric, nll, expert_id)``; bounded
              (K_s = 4 -> <= 6 pairs) by default, exhaustive on request.  A pair
              counts only if it is solved *and* clears the task-metric marginal
              criterion.
  RESIDUAL  base failed, every single failed, every legal pair failed -> keep
          the single best historical context by ``(-metric, nll, expert_id)``.
          Cardinality is at most 1: the composition budget is Top-2 and a
          Residual sample spends its second slot on the candidate expert, so a
          two-expert historical context would not fit the inference cardinality.

Marginal contribution is not a gate on identity: ``delta_nll`` is recorded as
soft evidence and can never by itself legitimise or reject a pair.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from compose.v8.config import (
    PAIR_SEARCH_EXHAUSTIVE,
    STATE_BASE_ONLY,
    STATE_RESIDUAL,
    STATE_REUSE1,
    STATE_REUSE2,
    TARGET_CONTEXT_POSITIVE,
    TARGET_IGNORE,
    TARGET_NEGATIVE,
    TARGET_POSITIVE,
    V8TeacherConfig,
)
from compose.v8.metric_adapter import TaskMetricAdapter


#: ``scorer(selection) -> {sample_id: task-metric value}``.
#: ``selection`` maps a sample id to the expert ids composed for that sample
#: (empty list == base only).  The scorer must use the *official* per-sample
#: task metric for decomposable tasks.
RouteScorer = Callable[[Mapping[str, Sequence[int]]], Mapping[str, float]]

#: ``nll_scorer(selection) -> {sample_id: mean answer NLL}``.  Soft signal only.
NLLScorer = Callable[[Mapping[str, Sequence[int]]], Mapping[str, float]]


class TeacherError(RuntimeError):
    """Raised when the teacher cannot make a metric-faithful decision."""


def _pair_key(left: int, right: int) -> str:
    return "pair_{:02d}_{:02d}".format(*sorted((int(left), int(right))))


@dataclass
class TeacherSampleRecord:
    """Everything the teacher concluded about one current-task sample."""

    sample_id: str
    state: str
    selected_experts: List[int]
    base_solved: bool
    base_value: float
    #: The router's Top-M by key similarity.  **Diagnostic only**: no state, no
    #: key target and no training mask is derived from this field.  It is what
    #: the keys are *judged* by, not what the capability search is bounded by.
    recall: List[int]
    single_values: Dict[str, float]
    pair_values: Dict[str, float]
    tested_singles: List[int]
    tested_pairs: List[List[int]]
    achieved_value: float
    achieved_nll: Optional[float]
    delta_nll: Optional[float]
    teacher_gain: float
    key_targets: Dict[str, str]
    decision_reason: str
    #: PART 17 keeps this separate from ``selected_experts``: on a Residual sample
    #: nothing solved it, so ``selected_experts`` is empty and this holds the
    #: single best available historical context by ``(-metric, nll, expert_id)``.
    #: Cardinality is at most 1 -- a two-expert context plus the candidate would
    #: be three experts against an inference budget of two.
    residual_context: List[int] = field(default_factory=list)
    #: The task metric value that counted as "solved", so downstream diagnostics
    #: re-read the raw values with the same criterion the decision used.
    solved_threshold: float = 0.0
    #: The capability-search universe for this sample and what was actually
    #: scored from it.  Equal on every base-unsolved sample by construction, and
    #: checked by ``TeacherError`` if that ever stops being true.
    historical_experts_visible: List[int] = field(default_factory=list)
    historical_experts_tested: List[int] = field(default_factory=list)

    @property
    def solved(self) -> bool:
        """True when the task metric was actually met by this sample's selection."""
        return self.state != STATE_RESIDUAL

    @property
    def cardinality(self) -> int:
        return len(self.selected_experts)

    @property
    def confidence(self) -> Optional[float]:
        """Soft confidence: the achieved NLL when a selection was made."""
        return self.achieved_nll

    @property
    def router_top_m(self) -> List[int]:
        """Alias of :attr:`recall`, named for what it actually is."""
        return list(self.recall)

    @property
    def context_positive_experts(self) -> List[int]:
        """Experts labelled CONTEXT_POSITIVE on this sample (Residual only)."""
        return sorted(
            int(expert_id) for expert_id, target in self.key_targets.items()
            if target == TARGET_CONTEXT_POSITIVE
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["solved"] = self.solved
        payload["cardinality"] = self.cardinality
        payload["router_top_m_diagnostic"] = list(self.recall)
        payload["context_positive_experts"] = self.context_positive_experts
        return payload


@dataclass
class TeacherResult:
    task_id: int
    records: List[TeacherSampleRecord]
    config: Dict[str, Any]
    scored_routes: List[Dict[str, Any]] = field(default_factory=list)

    # -- views ---------------------------------------------------------
    def by_sample(self) -> Dict[str, TeacherSampleRecord]:
        return {record.sample_id: record for record in self.records}

    def state_counts(self) -> Dict[str, int]:
        counts = {STATE_BASE_ONLY: 0, STATE_REUSE1: 0, STATE_REUSE2: 0, STATE_RESIDUAL: 0}
        for record in self.records:
            counts[record.state] += 1
        return counts

    def state_rate(self) -> Dict[str, float]:
        total = max(len(self.records), 1)
        return {state: count / total for state, count in self.state_counts().items()}

    def positive_counts(self) -> Dict[int, int]:
        """``{expert_id: solver positives}`` -- the ``SolverPositiveQueries`` half.

        This counts only samples the expert was *selected* to solve.  Since the
        capability search is a full-history oracle, an expert can be a solver
        without ever being recalled by the router, so this is capability support
        rather than routing support.
        """
        counts: Dict[int, int] = {}
        for record in self.records:
            for expert_id, target in record.key_targets.items():
                if target == TARGET_POSITIVE:
                    counts[int(expert_id)] = counts.get(int(expert_id), 0) + 1
        return counts

    def context_positive_counts(self) -> Dict[int, int]:
        """``{expert_id: context positives}`` -- the ``ContextPositiveQueries`` half.

        A Residual sample's best historical context was not able to solve the
        sample on its own, but it is the expert the composition would lean on.
        That is a reason to *call* it, so its alias key is attracted to those
        queries rather than pushed away from them.
        """
        counts: Dict[int, int] = {}
        for record in self.records:
            for expert_id, target in record.key_targets.items():
                if target == TARGET_CONTEXT_POSITIVE:
                    counts[int(expert_id)] = counts.get(int(expert_id), 0) + 1
        return counts

    def support_counts(self) -> Dict[int, int]:
        """``AliasSupport(E_k, t)`` = solver positives union context positives."""
        solver = self.positive_counts()
        context = self.context_positive_counts()
        return {
            expert_id: solver.get(expert_id, 0) + context.get(expert_id, 0)
            for expert_id in set(solver) | set(context)
        }

    def support_source(self, expert_id: int) -> str:
        """``solver_only`` / ``context_only`` / ``mixed`` / ``none`` for one expert."""
        solver = self.positive_counts().get(int(expert_id), 0)
        context = self.context_positive_counts().get(int(expert_id), 0)
        if solver and context:
            return "mixed"
        if solver:
            return "solver_only"
        if context:
            return "context_only"
        return "none"

    @property
    def search_mode(self) -> str:
        return str(self.config.get("search_mode", ""))

    @property
    def visible_experts(self) -> List[int]:
        """The capability-search universe, unioned over records."""
        experts: set = set()
        for record in self.records:
            experts.update(int(value) for value in record.historical_experts_visible)
        return sorted(experts)

    def context_positives_by_sample(self) -> Dict[str, List[int]]:
        return {
            record.sample_id: record.context_positive_experts
            for record in self.records
        }

    def coverage_report(self) -> Dict[str, Any]:
        """Evidence that the search really was an oracle, per sample.

        The assertion in :func:`assert_full_history_coverage` is the guarantee;
        this is the auditable trace of it, so a reader of ``teacher_result.json``
        can check coverage without re-running the teacher.  ``fully_covered`` is
        the number of base-unsolved samples scored on the entire visible pool.
        """
        searched = [record for record in self.records if not record.base_solved]
        fully = [
            record for record in searched
            if set(int(v) for v in record.historical_experts_tested)
            >= set(int(v) for v in record.historical_experts_visible)
        ]
        sizes = sorted({len(record.historical_experts_tested) for record in searched})
        visible = set(int(value) for value in self.visible_experts)
        tested: set = set()
        for record in searched:
            tested.update(int(value) for value in record.historical_experts_tested)
        return {
            "teacher_search_mode": self.search_mode,
            "historical_experts_visible": list(self.visible_experts),
            "num_visible_experts": len(self.visible_experts),
            #: Union over unsolved samples.  Recorded alongside the visible set so
            #: a reader can name the experts a bounded search would have missed,
            #: rather than only observing that a set size differed.
            "historical_experts_tested": sorted(tested),
            "never_tested": sorted(visible - tested),
            "base_only_samples": len(self.records) - len(searched),
            "searched_samples": len(searched),
            "fully_covered_samples": len(fully),
            "tested_set_sizes": sizes,
            "full_coverage": len(fully) == len(searched),
        }

    def negative_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for record in self.records:
            for expert_id, target in record.key_targets.items():
                if target == TARGET_NEGATIVE:
                    counts[int(expert_id)] = counts.get(int(expert_id), 0) + 1
        return counts

    def ignore_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for record in self.records:
            for expert_id, target in record.key_targets.items():
                if target == TARGET_IGNORE:
                    counts[int(expert_id)] = counts.get(int(expert_id), 0) + 1
        return counts

    def positives_by_sample(self) -> Dict[str, List[int]]:
        return {
            record.sample_id: sorted(
                int(expert_id) for expert_id, target in record.key_targets.items()
                if target == TARGET_POSITIVE
            )
            for record in self.records
        }

    def residual_context_by_sample(self) -> Dict[str, List[int]]:
        """PART 17's ``residual_context.selected_old_set``, per sample."""
        return {
            record.sample_id: list(record.residual_context)
            for record in self.records
            if record.state == STATE_RESIDUAL
        }

    def solved_rate(self) -> float:
        return sum(1 for record in self.records if record.solved) / max(len(self.records), 1)

    def solved_expert_histogram(self) -> Dict[int, int]:
        """How often each expert solved a sample as a *single* (capability view)."""
        counts: Dict[int, int] = {}
        for record in self.records:
            for expert_id in record.tested_singles:
                if record.single_values.get(str(expert_id), 0.0) >= record.solved_threshold:
                    counts[int(expert_id)] = counts.get(int(expert_id), 0) + 1
        return counts

    def marginal_summary(self) -> Dict[str, float]:
        residuals = [record for record in self.records if record.state == STATE_RESIDUAL]
        gains = [record.teacher_gain for record in self.records]
        return {
            "samples": len(self.records),
            "mean_teacher_gain": sum(gains) / max(len(gains), 1),
            "residual_samples": len(residuals),
            "mean_residual_gain": (
                sum(record.teacher_gain for record in residuals) / len(residuals)
                if residuals else 0.0
            ),
        }


class AnswerSupervisedTeacher:
    """Base-first, metric-gated expert teacher."""

    def __init__(
        self,
        metric_adapter: TaskMetricAdapter,
        config: Optional[V8TeacherConfig] = None,
    ) -> None:
        self.metric_adapter = metric_adapter
        self.config = config or V8TeacherConfig()

    # ------------------------------------------------------------------
    def run(
        self,
        task_id: int,
        sample_ids: Sequence[str],
        visible_experts: Sequence[int],
        scorer: RouteScorer,
        nll_scorer: Optional[NLLScorer] = None,
        recall_map: Optional[Mapping[str, Sequence[int]]] = None,
        pair_min_metric_gain: Optional[float] = None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> TeacherResult:
        """Score every visible historical expert as a single, then decide.

        ``visible_experts`` is the capability-search universe: the historical
        experts the continual scope makes available at this task.  It is a
        *required* argument so that no caller can fall back to a recall-limited
        search by omission -- the whole point of the oracle is that the router
        does not get to choose who receives answer supervision.

        ``recall_map`` is the router's Top-M per sample, now purely diagnostic.
        """
        ids = [str(value) for value in sample_ids]
        if not ids:
            raise TeacherError("the teacher needs at least one sample")
        spec = self.metric_adapter.require_decomposable(task_id)
        gain_floor = (
            float(self.config.pair_min_metric_gain)
            if pair_min_metric_gain is None else float(pair_min_metric_gain)
        )
        log = progress or (lambda message: None)
        scored_routes: List[Dict[str, Any]] = []

        # ---------------- STEP A: base ---------------------------------
        base_selection = {sample_id: [] for sample_id in ids}
        base_values = self._score(scorer, base_selection, ids, "base")
        scored_routes.append({"route": "base", "samples": len(ids)})
        base_solved = {
            sample_id: bool(base_values[sample_id] >= spec.solved_value)
            for sample_id in ids
        }
        log(f"teacher STEP A: base solved {sum(base_solved.values())}/{len(ids)}")

        # ---------------- STEP B: recall (diagnostic) ------------------
        # The router's ordering is *measured* here and consulted nowhere below.
        # It is what the keys are judged by -- the recall curve, the alias-key
        # counterfactual, the origin-key routing deficit -- and those measurements
        # only mean something if the router was never allowed to steer the search
        # that produces the solvers they are scored against.
        recall: Dict[str, List[int]] = {}
        diagnostic = recall_map or {}
        for sample_id in ids:
            seen: List[int] = []
            for value in diagnostic.get(sample_id, []):
                if int(value) not in seen:
                    seen.append(int(value))
            recall[sample_id] = seen[: int(self.config.historical_top_m)]

        # The capability universe: every visible historical expert, sorted for a
        # deterministic pass order.  Not the union of recalls, not Top-M.
        candidates = sorted({int(value) for value in visible_experts})

        pending = [sample_id for sample_id in ids if not base_solved[sample_id]]
        log(f"teacher STEP B: {len(candidates)} visible historical experts; "
            f"router Top-{self.config.historical_top_m} recorded for "
            f"{len(ids)} samples as a diagnostic only")

        # ---------------- STEP C: singles ------------------------------
        # Only the samples STEP A could not solve are worth a forward pass.  A
        # BaseOnly sample's single scores are never consulted, so scoring them
        # would burn generation budget for a number nobody reads.  For every
        # *pending* sample, though, the pass is total: each candidate is scored
        # on each pending sample, so ``tested_singles == visible_experts`` holds
        # by construction and is asserted below rather than assumed.
        single_values: Dict[int, Dict[str, float]] = {}
        for expert_id in candidates:
            selection = {sample_id: [expert_id] for sample_id in pending}
            values = self._score(scorer, selection, pending, f"single_{expert_id:02d}")
            single_values[expert_id] = values
            scored_routes.append({
                "route": f"single_{expert_id:02d}",
                "samples": len(pending),
                "solved": int(sum(
                    1 for sample_id in pending if values[sample_id] >= spec.solved_value
                )),
            })
        log(f"teacher STEP C: scored {len(candidates)} singles over "
            f"{len(pending)} unsolved samples")

        # A single already solved these, so the specification's STOP applies
        # before pair search: no pair forward pass is issued for them.
        pair_pending = [
            sample_id for sample_id in pending
            if not any(
                single_values[expert_id][sample_id] >= spec.solved_value
                for expert_id in candidates
            )
        ]
        log(f"teacher: {len(pair_pending)} samples reach pair search "
            f"({len(pending) - len(pair_pending)} stopped at a solved single)")

        # ---------------- soft NLL, singles ---------------------------
        # The shortlist ranking below is (metric, NLL, expert_id), so the single
        # NLLs have to exist before a pair can be chosen.  They are scored over
        # exactly the same samples and selections as the metric pass, so every
        # recorded NLL belongs to a route that was actually evaluated.
        nll_cache: Dict[str, Dict[str, float]] = {}
        if nll_scorer is not None:
            nll_cache["base"] = self._score_nll(nll_scorer, base_selection, ids)
            for expert_id in candidates:
                nll_cache[f"single_{expert_id:02d}"] = self._score_nll(
                    nll_scorer, {sample_id: [expert_id] for sample_id in pending}, pending
                )

        # ---------------- pairs ---------------------------------------
        # The shortlist is cut from the *scored* singles -- ranked by the task
        # metric, then NLL, then expert id -- and never from the router's order.
        # NLL here orders which bounded set of pairs is worth spending generation
        # on; it still cannot make a pair count as solved, which the marginal
        # rule below decides on the task metric alone.
        pair_values: Dict[str, Dict[str, float]] = {}
        pair_owner: Dict[str, List[str]] = {}
        shortlists: Dict[str, List[int]] = {}
        for sample_id in pair_pending:
            shortlist = self._pair_shortlist(
                sample_id, candidates, single_values, nll_cache
            )
            shortlists[sample_id] = shortlist
            if len(shortlist) < 2:
                continue
            for left, right in combinations(shortlist, 2):
                pair_values.setdefault(_pair_key(left, right), {})[sample_id] = 0.0
                pair_owner.setdefault(_pair_key(left, right), []).append(sample_id)
        for key in sorted(pair_values):
            members = pair_owner[key]
            left, right = (int(value) for value in key.split("_")[1:])
            selection = {sample_id: [left, right] for sample_id in members}
            values = self._score(scorer, selection, members, key)
            pair_values[key] = values
            scored_routes.append({
                "route": key,
                "samples": len(members),
                "solved": int(sum(
                    1 for sample_id in members if values[sample_id] >= spec.solved_value
                )),
            })
        log(f"teacher: scored {len(pair_values)} distinct pairs "
            f"({self.config.pair_search_mode}, K_s="
            f"{self.config.pair_top_k_single}) over {len(pair_pending)} samples")

        if nll_scorer is not None:
            for key, members in pair_owner.items():
                left, right = (int(value) for value in key.split("_")[1:])
                nll_cache[key] = self._score_nll(
                    nll_scorer,
                    {sample_id: [left, right] for sample_id in members},
                    members,
                )

        # ---------------- decisions -----------------------------------
        records: List[TeacherSampleRecord] = []
        for sample_id in ids:
            records.append(self._decide(
                sample_id=sample_id,
                task_id=task_id,
                spec_solved_value=spec.solved_value,
                base_value=base_values[sample_id],
                base_is_solved=base_solved[sample_id],
                recall=recall[sample_id],
                visible_experts=candidates,
                shortlist=shortlists.get(sample_id, []),
                single_values=single_values,
                pair_values=pair_values,
                nll_cache=nll_cache,
                gain_floor=gain_floor,
            ))
        assert_full_history_coverage(records, candidates)
        result = TeacherResult(
            task_id=int(task_id),
            records=records,
            config={**asdict(self.config), "pair_min_metric_gain": gain_floor},
            scored_routes=scored_routes,
        )
        log(f"teacher states: {result.state_counts()}")
        return result

    # ------------------------------------------------------------------
    def _pair_shortlist(
        self,
        sample_id: str,
        visible_experts: Sequence[int],
        single_values: Mapping[int, Mapping[str, float]],
        nll_cache: Mapping[str, Mapping[str, float]],
    ) -> List[int]:
        """Rank the *scored* singles and cut the pair shortlist.

        Ranking is ``(-task_metric, answer_nll, expert_id)``: the metric decides
        because it is the solved signal, NLL breaks its ties because it is the
        only other evidence available, and the id makes the order total and
        reproducible.  ``exhaustive`` keeps every visible single; ``bounded``
        keeps the best ``pair_top_k_single``.
        """
        ranked = rank_singles(sample_id, visible_experts, single_values, nll_cache)
        if self.config.pair_search_mode == PAIR_SEARCH_EXHAUSTIVE:
            return ranked
        return ranked[: int(self.config.pair_top_k_single)]

    # ------------------------------------------------------------------
    def _decide(
        self,
        sample_id: str,
        task_id: int,
        spec_solved_value: float,
        base_value: float,
        base_is_solved: bool,
        recall: Sequence[int],
        visible_experts: Sequence[int],
        shortlist: Sequence[int],
        single_values: Mapping[int, Mapping[str, float]],
        pair_values: Mapping[str, Mapping[str, float]],
        nll_cache: Mapping[str, Mapping[str, float]],
        gain_floor: float,
    ) -> TeacherSampleRecord:
        tested_singles: List[int] = []
        tested_pairs: List[List[int]] = []
        visible = [int(value) for value in visible_experts]

        # ---- STEP A: minimal capacity comes first --------------------
        if base_is_solved:
            return TeacherSampleRecord(
                sample_id=sample_id,
                state=STATE_BASE_ONLY,
                selected_experts=[],
                base_solved=True,
                base_value=float(base_value),
                recall=list(recall),
                single_values={},
                pair_values={},
                tested_singles=[],
                tested_pairs=[],
                achieved_value=float(base_value),
                achieved_nll=_pick(nll_cache, "base", sample_id),
                delta_nll=None,
                teacher_gain=0.0,
                # STEP A stops before any expert is scored, so nothing is known
                # to be wrong and no expert may be pushed away: the router's own
                # Top-M -- the only expert list this sample has -- is IGNORE.
                key_targets={int(expert_id): TARGET_IGNORE for expert_id in recall},
                decision_reason="base_solved_minimal_capacity",
                solved_threshold=float(spec_solved_value),
                historical_experts_visible=visible,
                historical_experts_tested=[],
            )

        # ---- STEP C: singles, over the whole visible historical pool --
        solved_singles: List[int] = []
        for expert_id in visible:
            value = float(single_values[expert_id][sample_id])
            tested_singles.append(int(expert_id))
            if value >= spec_solved_value:
                solved_singles.append(int(expert_id))

        if solved_singles:
            best_single = min(
                solved_singles,
                key=lambda expert_id: (
                    -float(single_values[expert_id][sample_id]),
                    _nll_or_inf(nll_cache, f"single_{expert_id:02d}", sample_id),
                    int(expert_id),
                ),
            )
            achieved = float(single_values[best_single][sample_id])
            achieved_nll = _pick(nll_cache, f"single_{best_single:02d}", sample_id)
            targets: Dict[int, str] = {}
            # The rule runs over every expert that was *scored* on this sample,
            # which is ``tested_singles`` -- STEP C scores the pool-wide
            # candidate list on each unsolved sample, so it is a superset of
            # this sample's ``recall``.  Iterating only over ``recall`` and
            # defaulting the rest to NEGATIVE got both edge cases wrong: an
            # expert outside the recall that solved this sample is an
            # alternative solver and must be IGNORE (PART 9 forbids
            # "E5 = negative" -- that label pushes its key away from a query it
            # demonstrably solves), and when the *best* single comes from
            # outside the recall it would lose the POSITIVE label of the
            # expert the teacher actually selected.
            for expert_id in sorted(
                {int(value) for value in tested_singles} | {int(value) for value in recall}
            ):
                if expert_id == best_single:
                    targets[expert_id] = TARGET_POSITIVE
                elif expert_id in solved_singles:
                    # Another expert also solves it.  Pushing its key away from a
                    # query it genuinely solves would destroy real capability.
                    targets[expert_id] = TARGET_IGNORE
                else:
                    targets[expert_id] = TARGET_NEGATIVE
            return TeacherSampleRecord(
                sample_id=sample_id,
                state=STATE_REUSE1,
                selected_experts=[best_single],
                base_solved=False,
                base_value=float(base_value),
                recall=list(recall),
                single_values={str(k): float(v[sample_id]) for k, v in single_values.items()
                               if sample_id in v},
                pair_values={},
                tested_singles=tested_singles,
                tested_pairs=[],
                achieved_value=achieved,
                achieved_nll=achieved_nll,
                delta_nll=_delta(_pick(nll_cache, "base", sample_id), achieved_nll),
                teacher_gain=achieved - float(base_value),
                key_targets=targets,
                decision_reason="single_solved_stop_before_pairs",
                solved_threshold=float(spec_solved_value),
                historical_experts_visible=visible,
                historical_experts_tested=tested_singles,
            )

        # ---- pairs ----------------------------------------------------
        best_value_for_marginal = max(
            [float(base_value)] + [
                float(single_values[expert_id][sample_id]) for expert_id in tested_singles
            ]
        )
        solved_pairs: List[Tuple[int, int, float, Optional[float]]] = []
        for left, right in combinations([int(v) for v in shortlist], 2):
            key = _pair_key(left, right)
            if key not in pair_values or sample_id not in pair_values[key]:
                continue
            value = float(pair_values[key][sample_id])
            tested_pairs.append([int(left), int(right)])
            if value < spec_solved_value:
                continue
            if value < best_value_for_marginal + gain_floor:
                # Solved, but it does not beat what a weaker selection already
                # achieved on the task metric: not a real marginal contribution.
                continue
            solved_pairs.append((int(left), int(right), value,
                                 _pick(nll_cache, key, sample_id)))

        if solved_pairs:
            left, right, achieved, achieved_nll = min(
                solved_pairs,
                key=lambda item: (-item[2], item[3] if item[3] is not None else float("inf"),
                                  item[0], item[1]),
            )
            chosen = {left, right}
            achieved_value = achieved
            achieved_nll = _pick(nll_cache, _pair_key(left, right), sample_id)
            delta = None
            if achieved_nll is not None:
                base_nll = _pick(nll_cache, "base", sample_id)
                if base_nll is not None:
                    delta = float(base_nll) - float(achieved_nll)
            # Total over everything that was scored, for the same reason the
            # Reuse1 rule is: ``tested_singles`` is the scored set and the
            # router's Top-M is a subset of it, so no scored expert can escape
            # a label.  No *single* solved this sample -- that is why the pair
            # branch was reached at all -- so every unselected expert is a
            # legitimate NEGATIVE, and the two that compose the solution are the
            # positives.
            targets = {}
            for expert_id in sorted(
                {int(value) for value in tested_singles} | {int(value) for value in recall}
            ):
                targets[expert_id] = (
                    TARGET_POSITIVE if expert_id in chosen else TARGET_NEGATIVE
                )
            return TeacherSampleRecord(
                sample_id=sample_id,
                state=STATE_REUSE2,
                selected_experts=[left, right],
                base_solved=False,
                base_value=float(base_value),
                recall=list(recall),
                single_values={str(k): float(v[sample_id]) for k, v in single_values.items()
                               if sample_id in v},
                pair_values={_pair_key(int(a), int(b)): float(v)
                             for (a, b, v, _n) in solved_pairs},
                tested_singles=tested_singles,
                tested_pairs=tested_pairs,
                achieved_value=achieved_value,
                achieved_nll=achieved_nll,
                delta_nll=delta,
                teacher_gain=float(achieved_value) - float(base_value),
                key_targets=targets,
                decision_reason="pair_solved_and_cleared_marginal_criterion",
                solved_threshold=float(spec_solved_value),
                historical_experts_visible=visible,
                historical_experts_tested=tested_singles,
            )

        # ---- RESIDUAL -------------------------------------------------
        # Keep the single best historical context by (-metric, NLL, expert_id).
        # Singles only, and at most one of them: the composition budget is Top-2
        # and a Residual sample's second slot belongs to the candidate expert, so
        # a two-expert historical context would ask the inference path for a
        # cardinality it does not have.  A best pair is therefore deliberately
        # *not* a context candidate, even when it outscores every single.
        context: List[int] = []
        if tested_singles:
            best_context = min(
                (int(expert_id) for expert_id in tested_singles),
                key=lambda expert_id: (
                    -float(single_values[expert_id][sample_id]),
                    _nll_or_inf(nll_cache, f"single_{expert_id:02d}", sample_id),
                    int(expert_id),
                ),
            )
            context = [best_context]

        achieved_value = float(base_value)
        if context:
            achieved_value = float(single_values[context[0]][sample_id])
            achieved_nll = _pick(nll_cache, f"single_{context[0]:02d}", sample_id)
        else:
            achieved_nll = _pick(nll_cache, "base", sample_id)

        # Nothing solved this sample, so ``selected_set`` stays empty and the
        # context is reported separately.  The verdict is deliberately *not*
        # "every tested expert is a negative": not being able to solve a sample
        # alone is a different statement from being useless in a composition.
        # The context expert is exactly the one the residual composition would
        # lean on, so it is labelled CONTEXT_POSITIVE and attracts its alias key.
        # The rest are IGNORE -- false negatives here would train the router away
        # from experts that are merely weak on this sample, and V8-v1 has no
        # validated criterion for calling them harmful.
        targets = {int(expert_id): TARGET_IGNORE for expert_id in tested_singles}
        for expert_id in recall:
            targets.setdefault(int(expert_id), TARGET_IGNORE)
        for expert_id in context:
            targets[int(expert_id)] = TARGET_CONTEXT_POSITIVE
        return TeacherSampleRecord(
            sample_id=sample_id,
            state=STATE_RESIDUAL,
            selected_experts=[],
            residual_context=list(context),
            base_solved=False,
            base_value=float(base_value),
            recall=list(recall),
            single_values={str(k): float(v[sample_id]) for k, v in single_values.items()
                           if sample_id in v},
            pair_values={},
            tested_singles=tested_singles,
            tested_pairs=tested_pairs,
            achieved_value=achieved_value,
            achieved_nll=achieved_nll,
            delta_nll=_delta(_pick(nll_cache, "base", sample_id), achieved_nll),
            teacher_gain=float(achieved_value) - float(base_value),
            key_targets=targets,
            decision_reason="no_single_or_pair_solved_residual_context_kept",
            solved_threshold=float(spec_solved_value),
            historical_experts_visible=visible,
            historical_experts_tested=tested_singles,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _score(
        scorer: RouteScorer,
        selection: Mapping[str, Sequence[int]],
        expected_ids: Sequence[str],
        label: str,
    ) -> Dict[str, float]:
        values = scorer(selection)
        out: Dict[str, float] = {}
        for sample_id in expected_ids:
            sample_id = str(sample_id)
            if sample_id not in values:
                raise TeacherError(f"scorer returned no value for {sample_id} ({label})")
            out[sample_id] = float(values[sample_id])
        return out

    @staticmethod
    def _score_nll(
        nll_scorer: NLLScorer,
        selection: Mapping[str, Sequence[int]],
        expected_ids: Sequence[str],
    ) -> Dict[str, float]:
        values = nll_scorer(selection)
        return {str(sample_id): float(values[str(sample_id)]) for sample_id in expected_ids
                if str(sample_id) in values}


def rank_singles(
    sample_id: str,
    visible_experts: Sequence[int],
    single_values: Mapping[int, Mapping[str, float]],
    nll_cache: Mapping[str, Mapping[str, float]],
) -> List[int]:
    """Order scored singles by ``(-task_metric, answer_nll, expert_id)``.

    The metric leads because it is the solved signal; NLL is the only other
    evidence available and only breaks the metric's own ties; the id makes the
    order total, so a run is reproducible without a tie-break lottery.
    """
    return sorted(
        (int(expert_id) for expert_id in visible_experts),
        key=lambda expert_id: (
            -float(single_values[expert_id][sample_id]),
            _nll_or_inf(nll_cache, f"single_{expert_id:02d}", sample_id),
            int(expert_id),
        ),
    )


def assert_full_history_coverage(
    records: Sequence[TeacherSampleRecord],
    visible_experts: Sequence[int],
) -> None:
    """Hard-fail unless every base-unsolved sample was scored on the whole pool.

    This is the invariant that makes the teacher a capability *oracle* rather
    than a recall-bounded search, and it is the one that silently degrades: if a
    future change reintroduces a shortlist, every state, key target and training
    mask downstream inherits the router's blind spot, and the resulting numbers
    would look like evidence about capability.  It is checked per run instead of
    being argued from the code.
    """
    visible = {int(value) for value in visible_experts}
    for record in records:
        tested = {int(value) for value in record.historical_experts_tested}
        if record.base_solved:
            if tested:
                raise TeacherError(
                    f"sample {record.sample_id} is BaseOnly but was scored on "
                    f"{sorted(tested)}; STEP A must stop before the expert search"
                )
            continue
        missing = sorted(visible - tested)
        extra = sorted(tested - visible)
        if missing or extra:
            raise TeacherError(
                f"sample {record.sample_id} was not scored on the full visible "
                f"historical pool: {len(missing)} visible expert(s) untested "
                f"{missing[:8]}, {len(extra)} scored outside the pool "
                f"{extra[:8]}.  The teacher must not let key recall decide which "
                "experts receive answer supervision (DECISION-1)"
            )


def _pick(cache: Mapping[str, Mapping[str, float]], route: str, sample_id: str) -> Optional[float]:
    route_values = cache.get(route)
    if not route_values:
        return None
    value = route_values.get(sample_id)
    return None if value is None else float(value)


def _nll_or_inf(cache: Mapping[str, Mapping[str, float]], route: str, sample_id: str) -> float:
    value = _pick(cache, route, sample_id)
    return float("inf") if value is None else value


def _delta(base_nll: Optional[float], achieved_nll: Optional[float]) -> Optional[float]:
    if base_nll is None or achieved_nll is None:
        return None
    return float(base_nll) - float(achieved_nll)


__all__ = [
    "AnswerSupervisedTeacher",
    "NLLScorer",
    "RouteScorer",
    "TeacherError",
    "TeacherResult",
    "TeacherSampleRecord",
    "assert_full_history_coverage",
    "rank_singles",
]
