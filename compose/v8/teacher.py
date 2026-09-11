"""Answer-Supervised Expert Teacher.

The teacher answers one question per current-task sample: **is there already an
expert in the pool that solves this sample?**  Only the task metric may answer
it.  Answer NLL is computed and recorded, but strictly in a supporting role --
ranking among already-solved experts, soft confidence, pair marginal evidence,
residual context and tie-breaks.  There is no code path in this module where a
loss value can flip ``solved`` from False to True; ``V8TeacherConfig`` rejects
the configuration that would allow it.

Per-sample flow (PART 9):

  STEP A  base alone -> if it solves the sample, ``BaseOnly``, selection empty,
          stop.  No expert search is performed, so no expert is marked negative:
          absence of evidence is recorded as ``ignore``.
  STEP B  recall the Top-M historical experts by key similarity (candidates).
  STEP C  score each recalled single with the **metric**.
            * any solved single  -> ``Reuse1``, take the lexicographic best
              ``(-metric, nll, expert_id)``, and STOP.  A lower-NLL pair is
              never allowed to override a solved single.
            * none solved        -> pairs over the shortlist (<= C(4,2) = 6),
              a pair counts only if it is solved *and* clears the task-metric
              marginal criterion.
  RESIDUAL  base failed, every single failed, every legal pair failed -> keep
          the best available historical context by (metric, nll, cardinality).

Marginal contribution is not a gate on identity: ``delta_nll`` is recorded as
soft evidence and can never by itself legitimise or reject a pair.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from compose.v8.config import (
    STATE_BASE_ONLY,
    STATE_RESIDUAL,
    STATE_REUSE1,
    STATE_REUSE2,
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
    #: best available historical context (metric, then NLL, then cardinality).
    residual_context: List[int] = field(default_factory=list)
    #: The task metric value that counted as "solved", so downstream diagnostics
    #: re-read the raw values with the same criterion the decision used.
    solved_threshold: float = 0.0

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

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["solved"] = self.solved
        payload["cardinality"] = self.cardinality
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
        """``{expert_id: number of current-task positives}``.

        This is ``|P(k, t)|`` in PART 11 -- the support that justifies creating an
        alias key for ``k`` on this task.
        """
        counts: Dict[int, int] = {}
        for record in self.records:
            for expert_id, target in record.key_targets.items():
                if target == TARGET_POSITIVE:
                    counts[int(expert_id)] = counts.get(int(expert_id), 0) + 1
        return counts

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
        recall_map: Mapping[str, Sequence[int]],
        scorer: RouteScorer,
        nll_scorer: Optional[NLLScorer] = None,
        pair_min_metric_gain: Optional[float] = None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> TeacherResult:
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

        # ---------------- STEP B: recall -------------------------------
        recall: Dict[str, List[int]] = {}
        for sample_id in ids:
            candidates = [int(value) for value in recall_map.get(sample_id, [])]
            seen: List[int] = []
            for value in candidates:
                if value not in seen:
                    seen.append(value)
            recall[sample_id] = seen[: int(self.config.historical_top_m)]

        pending = [sample_id for sample_id in ids if not base_solved[sample_id]]
        candidates: List[int] = []
        for sample_id in pending:
            for expert_id in recall[sample_id]:
                if expert_id not in candidates:
                    candidates.append(expert_id)
        candidates.sort()
        log(f"teacher STEP B: {len(candidates)} distinct recalled experts over "
            f"{len(pending)} unsolved samples")

        # ---------------- STEP C: singles ------------------------------
        # Only the samples STEP A could not solve are worth a forward pass.  A
        # BaseOnly sample's single scores are never consulted, so scoring them
        # would burn generation budget for a number nobody reads.
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

        # ---------------- pairs ---------------------------------------
        pair_values: Dict[str, Dict[str, float]] = {}
        pair_owner: Dict[str, List[str]] = {}
        for sample_id in pair_pending:
            shortlist = recall[sample_id][: int(self.config.pair_top_k_single)]
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
        log(f"teacher: scored {len(pair_values)} distinct pairs")

        # ---------------- optional soft NLL ---------------------------
        # Mirrors the metric passes exactly: same samples, same selections, so
        # every recorded NLL belongs to a route that was actually evaluated.
        nll_cache: Dict[str, Dict[str, float]] = {}
        if nll_scorer is not None:
            nll_cache["base"] = self._score_nll(nll_scorer, base_selection, ids)
            for expert_id in candidates:
                nll_cache[f"single_{expert_id:02d}"] = self._score_nll(
                    nll_scorer, {sample_id: [expert_id] for sample_id in pending}, pending
                )
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
                candidates=candidates,
                single_values=single_values,
                pair_values=pair_values,
                nll_cache=nll_cache,
                gain_floor=gain_floor,
            ))
        result = TeacherResult(
            task_id=int(task_id),
            records=records,
            config={**asdict(self.config), "pair_min_metric_gain": gain_floor},
            scored_routes=scored_routes,
        )
        log(f"teacher states: {result.state_counts()}")
        return result

    # ------------------------------------------------------------------
    def _decide(
        self,
        sample_id: str,
        task_id: int,
        spec_solved_value: float,
        base_value: float,
        base_is_solved: bool,
        recall: Sequence[int],
        candidates: Sequence[int],
        single_values: Mapping[int, Mapping[str, float]],
        pair_values: Mapping[str, Mapping[str, float]],
        nll_cache: Mapping[str, Mapping[str, float]],
        gain_floor: float,
    ) -> TeacherSampleRecord:
        tested_singles: List[int] = []
        tested_pairs: List[List[int]] = []

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
                # No expert was evaluated on this sample, so nothing is known
                # to be wrong: every recalled expert is IGNORE, never negative.
                key_targets={int(expert_id): TARGET_IGNORE for expert_id in recall},
                decision_reason="base_solved_minimal_capacity",
                solved_threshold=float(spec_solved_value),
            )

        # ---- STEP C: singles -----------------------------------------
        solved_singles: List[int] = []
        for expert_id in candidates:
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
            for expert_id in recall:
                expert_id = int(expert_id)
                if expert_id == best_single:
                    targets[expert_id] = TARGET_POSITIVE
                elif expert_id in solved_singles:
                    # Another expert also solves it.  Pushing its key away from a
                    # query it genuinely solves would destroy real capability.
                    targets[expert_id] = TARGET_IGNORE
                else:
                    targets[expert_id] = TARGET_NEGATIVE
            # Recall candidates are recorded in `recall`; experts outside this
            # sample's recall were never considered, so they stay untouched.
            for expert_id in tested_singles:
                targets.setdefault(int(expert_id), TARGET_NEGATIVE)
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
            )

        # ---- pairs ----------------------------------------------------
        shortlist = [int(value) for value in recall[: int(self.config.pair_top_k_single)]]
        best_value_for_marginal = max(
            [float(base_value)] + [
                float(single_values[expert_id][sample_id]) for expert_id in tested_singles
            ]
        )
        solved_pairs: List[Tuple[int, int, float, Optional[float]]] = []
        for left, right in combinations(shortlist, 2):
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
            targets = {}
            for expert_id in recall:
                expert_id = int(expert_id)
                if expert_id in chosen:
                    targets[expert_id] = TARGET_POSITIVE
                else:
                    targets[expert_id] = TARGET_NEGATIVE
            for expert_id in tested_singles:
                targets.setdefault(int(expert_id), TARGET_NEGATIVE)
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
            )

        # ---- RESIDUAL -------------------------------------------------
        # Keep the best historical context by (metric, NLL, smaller cardinality).
        context_candidates: List[Tuple[float, float, int, List[int]]] = []
        for expert_id in tested_singles:
            value = float(single_values[expert_id][sample_id])
            context_candidates.append((
                value,
                _nll_or_inf(nll_cache, f"single_{expert_id:02d}", sample_id),
                1,
                [int(expert_id)],
            ))
        for left, right in tested_pairs:
            key = _pair_key(left, right)
            value = float(pair_values[key][sample_id])
            context_candidates.append((
                value,
                _nll_or_inf(nll_cache, key, sample_id),
                2,
                [int(left), int(right)],
            ))
        if context_candidates:
            _value, _nll, _cardinality, context = min(
                context_candidates,
                key=lambda item: (-item[0], item[1], item[2], item[3]),
            )
        else:
            context = []

        achieved_value = float(base_value)
        if context:
            if len(context) == 1:
                achieved_value = float(single_values[context[0]][sample_id])
                achieved_nll = _pick(nll_cache, f"single_{context[0]:02d}", sample_id)
            else:
                achieved_value = float(pair_values[_pair_key(*context)][sample_id])
                achieved_nll = _pick(nll_cache, _pair_key(*context), sample_id)
        else:
            achieved_nll = _pick(nll_cache, "base", sample_id)

        # Nothing solved this sample, so ``selected_set`` stays empty and the
        # context is reported separately (PART 17).  Every expert that was tried
        # and failed is a legitimate negative (PART 19.2): the alias keys must
        # learn that this task's queries are *not* their capability, which is
        # exactly what keeps the router from recalling them here.
        targets = {int(expert_id): TARGET_NEGATIVE for expert_id in recall}
        for expert_id in tested_singles:
            targets.setdefault(int(expert_id), TARGET_NEGATIVE)
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
]
