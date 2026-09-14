"""V8 acceptance tests: TEST 01 - TEST 30 of the specification's test plan.

Every test here runs for real -- no test is skipped on the grounds that it would
need a GPU.  The heavyweight objects (the 7B backbone, the committed V7 pool)
are represented by small fixtures that exercise the *same* code paths: the
gradient tests use the genuine ``ComposeLinear`` forward, and the migration and
capability tests run against the real committed V7 key store when it is present.

The tests are grouped exactly as the specification enumerates them, and each
function name carries its TEST number so the acceptance report can cite them
one-to-one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import pytest
import torch
from torch import nn

from compose.adapters.lora import ComposeLinear
from compose.adapters.manager import ExpertManager
from compose.adapters.runtime import use_selection
from compose.v8 import audit as v8_audit
from compose.v8.cache import (
    TeacherCacheError,
    assert_roundtrip,
    read_teacher_result,
    write_teacher_result,
)
from compose.v8.checkpoint import (
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
    verify_resume_identity,
)
from compose.v8.config import (
    STATE_BASE_ONLY,
    STATE_RESIDUAL,
    STATE_REUSE1,
    STATE_REUSE2,
    TEACHER_SEARCH_FULL_HISTORY,
    TARGET_CONTEXT_POSITIVE,
    TARGET_IGNORE,
    TARGET_NEGATIVE,
    TARGET_POSITIVE,
    V8Config,
    V8KeyConfig,
    V8TeacherConfig,
)
from compose.v8.gating import (
    GradientGatingError,
    capture_frozen_ledger,
    enforce_freeze_policy,
    gradient_leakage_probe,
    residual_answer_loss,
    state_weight_report,
    verify_frozen_ledger,
)
from compose.v8.inference import (
    V8InferenceRouter,
    assert_inference_purity,
    validate_policy,
)
from compose.v8.key_learning import (
    alias_key_loss,
    assert_no_ignore_is_negative,
    build_key_targets,
    create_alias_keys,
)
from compose.v8.metric_adapter import (
    MetricAdapterError,
    TaskMetricAdapter,
    parse_result_value,
)
from compose.v8.pool import (
    MultiKeyExpertPool,
    MultiKeyPoolError,
    alias_key_init,
    tensor_checksum,
)
from compose.v8.query import assert_query_contract, build_query
from compose.v8.routing import (
    MultiKeyRouter,
    duplicate_row_count,
    distinct_expert_rate,
)
from compose.v8.selection import (
    STATE_CARDINALITY,
    SelectionError,
    build_selection,
    residual_weights,
    uniform_selection,
)
from compose.v8.teacher import AnswerSupervisedTeacher, TeacherError

REPO = Path(__file__).resolve().parents[2]
V7_FORMAL_KEYS = Path(
    "/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_gpu01_cached_query_formal_20260903"
    "/task5/committed/v7_keys.pt"
)
DIM = 1536


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
class TinyComposeNet(nn.Module):
    """A real ``ComposeLinear`` stack at negligible size."""

    def __init__(self, features: int = 8, rank: int = 2, alpha: float = 4.0,
                 depth: int = 2) -> None:
        super().__init__()
        self.stack = nn.ModuleList([
            ComposeLinear(nn.Linear(features, features), rank=rank, alpha=alpha)
            for _ in range(depth)
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.stack:
            inputs = layer(inputs)
        return inputs


def make_manager(expert_ids: Sequence[int], features: int = 8):
    model = TinyComposeNet(features=features)
    manager = ExpertManager(model)
    for expert_id in expert_ids:
        manager.add_expert(int(expert_id))
    return model, manager


def make_pool(num_experts: int = 6, per_task: int = 2, seed: int = 0) -> MultiKeyExpertPool:
    generator = torch.Generator().manual_seed(seed)
    pool = MultiKeyExpertPool()
    for expert_id in range(num_experts):
        origin_task = expert_id // per_task
        pool.add_expert(expert_id=expert_id, origin_task=origin_task,
                        lifecycle="historical")
        direction = torch.randn(DIM, generator=generator)
        pool.add_key(expert_id=expert_id, task_id=origin_task, key_type="origin",
                     value=torch.nn.functional.normalize(direction, dim=-1),
                     lifecycle="historical")
    pool.validate()
    return pool


def unit(index: int, scale: float = 1.0) -> torch.Tensor:
    """A 1536-D basis vector, so geometry in tests is exact."""
    value = torch.zeros(DIM)
    value[index] = scale
    return value


class FakeTeacherTask:
    """Declared answer oracle: solves exactly what the test says it solves."""

    def __init__(
        self,
        base_solved: Sequence[str] = (),
        singles: Mapping[int, Sequence[str]] | None = None,
        pairs: Mapping[Sequence[int], Sequence[str]] | None = None,
        nll: Mapping[str, float] | None = None,
        base_nll: Mapping[str, float] | None = None,
    ) -> None:
        self.base_solved = {str(v) for v in base_solved}
        self.singles = {int(k): {str(v) for v in values}
                        for k, values in (singles or {}).items()}
        self.pairs = {tuple(sorted(int(v) for v in k)): {str(v) for v in values}
                      for k, values in (pairs or {}).items()}
        self.nll = {str(k): float(v) for k, v in (nll or {}).items()}
        self.base_nll = {str(k): float(v) for k, v in (base_nll or {}).items()}

    def route_key(self, experts: Sequence[int]) -> str:
        experts = [int(v) for v in experts]
        if not experts:
            return "base"
        if len(experts) == 1:
            return f"single_{experts[0]:02d}"
        return "pair_{:02d}_{:02d}".format(*sorted(experts))

    def scorer(self, selection: Mapping[str, Sequence[int]]) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for sample_id, experts in selection.items():
            sample_id = str(sample_id)
            experts = [int(v) for v in experts]
            if not experts:
                solved = sample_id in self.base_solved
            elif len(experts) == 1:
                solved = sample_id in self.singles.get(experts[0], set())
            else:
                solved = sample_id in self.pairs.get(tuple(sorted(experts)), set())
            values[sample_id] = 1.0 if solved else 0.0
        return values

    def nll_scorer(self, selection: Mapping[str, Sequence[int]]) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for sample_id, experts in selection.items():
            sample_id = str(sample_id)
            key = self.route_key(experts)
            if not experts:
                values[sample_id] = float(self.base_nll.get(sample_id, 5.0))
            else:
                values[sample_id] = float(self.nll.get(f"{key}|{sample_id}", 5.0))
        return values


def visible_experts_for(
    task: FakeTeacherTask,
    recall: Mapping[str, Sequence[int]],
) -> List[int]:
    """The historical pool visible to the teacher, derived *independently*.

    Deliberately not derived from what the teacher would test: the whole point
    of the Capability Discovery Oracle is that the search universe is a property
    of the pool and the continual scope, not of the teacher's own decisions.  So
    it is built from the declared oracle's expert set (what *could* solve
    something) unioned with the router's Top-M (which is now diagnostic only).
    Deriving it from the teacher's answer would make the coverage assertion in
    ``assert_full_history_coverage`` vacuous.
    """
    experts = {int(v) for values in recall.values() for v in values}
    experts.update(int(k) for k in task.singles)
    for pair in task.pairs:
        experts.update(int(v) for v in pair)
    return sorted(experts)


def run_teacher(
    task: FakeTeacherTask,
    sample_ids: Sequence[str],
    recall: Mapping[str, Sequence[int]],
    *,
    with_nll: bool = True,
    pair_min_metric_gain: float | None = None,
    config: V8TeacherConfig | None = None,
    visible: Sequence[int] | None = None,
):
    teacher = AnswerSupervisedTeacher(TaskMetricAdapter(), config)
    return teacher.run(
        task_id=3,
        sample_ids=list(sample_ids),
        visible_experts=(
            list(visible) if visible is not None
            else visible_experts_for(task, recall)
        ),
        recall_map=recall,
        scorer=task.scorer,
        nll_scorer=task.nll_scorer if with_nll else None,
        pair_min_metric_gain=pair_min_metric_gain,
    )


# ---------------------------------------------------------------------------
# TEST 01 - TEST 04: the freeze contract
# ---------------------------------------------------------------------------
def test_01_fixed_query_has_zero_trainable_parameters():
    query = build_query()
    contract = assert_query_contract(query)
    assert contract["trainable_parameters"] == 0
    assert contract["total_parameters"] == 0
    assert contract["query_dim"] == 1536
    assert all(not parameter.requires_grad for parameter in query.parameters())
    assert not list(query.buffers())
    with pytest.raises(Exception):
        V8Config(query={"trainable_parameter_count": 1})


def test_02_historical_lora_requires_grad_false():
    model, manager = make_manager([0, 1, 2, 3])
    pool = make_pool(num_experts=4)
    report = enforce_freeze_policy(model, manager, pool, current_task=1,
                                   candidate_expert_ids=[2, 3])
    assert report["audit"].ok, report["audit"].render()
    for expert_id in (0, 1):
        for layer in manager.layers.values():
            for parameter in layer.experts[str(expert_id)].parameters():
                assert parameter.requires_grad is False
    for expert_id in (2, 3):
        assert any(parameter.requires_grad
                   for layer in manager.layers.values()
                   for parameter in layer.experts[str(expert_id)].parameters())
    # the base backbone is frozen too
    for parameter in model.parameters():
        if not any(parameter is candidate
                   for expert_id in (2, 3)
                   for layer in manager.layers.values()
                   for candidate in layer.experts[str(expert_id)].parameters()):
            assert parameter.requires_grad is False


def test_03_historical_lora_checksum_unchanged_after_training():
    model, manager = make_manager([0, 1, 2, 3])
    pool = make_pool(num_experts=4)
    enforce_freeze_policy(model, manager, pool, current_task=1,
                          candidate_expert_ids=[2, 3])
    full_lora: Dict[int, Dict[str, torch.Tensor]] = {0: {}, 1: {}}
    for expert_id in (0, 1):
        for layer_name, layer in sorted(manager.layers.items()):
            for name, tensor in layer.experts[str(expert_id)].state_dict().items():
                full_lora[expert_id][f"{layer_name}.{name}"] = tensor
    ledger = capture_frozen_ledger(pool, full_lora)

    optimizer = torch.optim.SGD(
        [p for expert_id in (2, 3)
         for layer in manager.layers.values()
         for p in layer.experts[str(expert_id)].parameters()], lr=0.1)
    inputs = torch.randn(4, 8)
    selection = uniform_selection([2, 3], batch_size=4)
    for _ in range(3):
        optimizer.zero_grad()
        with use_selection(selection):
            output = model(inputs)
        output.pow(2).sum().backward()
        optimizer.step()

    report = verify_frozen_ledger(ledger, pool, full_lora)
    assert report["status"] == "UNCHANGED"
    assert report["changed_lora"] == []

    # the ledger must actually detect a change
    victim = next(iter(full_lora[0].values()))
    victim.data.add_(1.0)
    with pytest.raises(GradientGatingError):
        verify_frozen_ledger(ledger, pool, full_lora)


def test_04_historical_committed_old_key_unchanged():
    pool = make_pool(num_experts=4)
    pool.add_key(expert_id=0, task_id=1, key_type="task_alias",
                 value=unit(100), lifecycle="candidate", trainable=True)
    ledger = capture_frozen_ledger(pool, {}, current_task=1)

    optimizer = torch.optim.SGD(
        [pool.keys[key_id] for key_id in pool.trainable_key_ids()], lr=0.5)
    query = torch.nn.functional.normalize(unit(0), dim=-1)
    for _ in range(3):
        optimizer.zero_grad()
        key = torch.nn.functional.normalize(pool.keys["e0_t1_task_alias"], dim=-1)
        (1.0 - (query @ key)).backward()
        optimizer.step()

    report = verify_frozen_ledger(ledger, pool, {})
    assert report["status"] == "UNCHANGED"
    assert report["changed_keys"] == []
    pool.keys["e0_t0_origin"].data.add_(0.5)
    with pytest.raises(GradientGatingError):
        verify_frozen_ledger(ledger, pool, {})


# ---------------------------------------------------------------------------
# TEST 05 - TEST 07: the multi-key structure and the router
# ---------------------------------------------------------------------------
def test_05_one_expert_can_own_multiple_keys():
    pool = make_pool(num_experts=4)
    pool.add_key(expert_id=1, task_id=2, key_type="task_alias",
                 value=unit(10), lifecycle="candidate", trainable=True)
    pool.validate()
    keys = pool.active_key_ids_for_expert(1)
    assert keys == ["e1_t0_origin", "e1_t2_task_alias"]
    origin, alias = (pool.key_records[k] for k in keys)
    assert origin["key_type"] == "origin" and origin["task_id"] == 0
    assert alias["key_type"] == "task_alias" and alias["task_id"] == 2
    assert alias["expert_id"] == origin["expert_id"] == 1
    audit = pool.audit()
    assert audit["experts_with_multiple_keys"] == [1]
    assert audit["num_keys"] == 5 and audit["num_experts"] == 4


def test_06_router_aggregates_key_scores_by_expert_using_max():
    pool = MultiKeyExpertPool()
    pool.add_expert(expert_id=0, origin_task=0, lifecycle="historical")
    pool.add_expert(expert_id=1, origin_task=0, lifecycle="historical")
    # expert 0: a weak origin key and a strong alias key
    pool.add_key(expert_id=0, task_id=0, key_type="origin", value=unit(1),
                 lifecycle="historical")
    pool.add_key(expert_id=0, task_id=1, key_type="task_alias", value=unit(0),
                 lifecycle="historical")
    # expert 1: a single key scoring strictly between the two
    middle = torch.zeros(DIM)
    middle[0] = 0.7
    middle[2] = (1 - 0.7 ** 2) ** 0.5
    pool.add_key(expert_id=1, task_id=0, key_type="origin", value=middle,
                 lifecycle="historical")
    router = MultiKeyRouter()
    query = unit(0)
    per_expert, pool_ids, _, _ = router.score_matrix(query.unsqueeze(0), pool)
    values = {int(e): float(per_expert[0, i]) for i, e in enumerate(pool_ids.tolist())}
    assert values[0] == pytest.approx(1.0, abs=1e-6)   # max(1.0, 0.0)
    assert values[1] == pytest.approx(0.7, abs=1e-6)
    result = router(query.unsqueeze(0), pool)
    assert result.expert_ids[0, 0].item() == 0
    assert result.key_ids[0][0] == "e0_t1_task_alias"  # the alias is what fired


def test_07_one_expert_cannot_occupy_two_top2_slots():
    pool = MultiKeyExpertPool()
    for expert_id in range(3):
        pool.add_expert(expert_id=expert_id, origin_task=0, lifecycle="historical")
        pool.add_key(expert_id=expert_id, task_id=0, key_type="origin",
                     value=unit(10 + expert_id), lifecycle="historical")
    # expert 0 gets a second key that is the single best match in the whole pool
    pool.add_key(expert_id=0, task_id=1, key_type="task_alias", value=unit(0),
                 lifecycle="historical")
    router = MultiKeyRouter()
    result = router(unit(0).unsqueeze(0), pool)
    slots = result.expert_ids[0].tolist()
    assert len(slots) == 2
    assert slots[0] == 0 and slots[1] != 0
    assert len(set(slots)) == 2
    assert distinct_expert_rate(result.expert_ids) == 1.0
    assert duplicate_row_count(["s0"], result.expert_ids) == 0


# ---------------------------------------------------------------------------
# TEST 08 - TEST 15: the teacher's metric-first decisions
# ---------------------------------------------------------------------------
def test_08_teacher_uses_task_correctness_for_solved_decision():
    samples = ["s0", "s1"]
    # s0: base fails, expert 4 solves.  s1: nothing solves.
    task = FakeTeacherTask(base_solved=[], singles={4: ["s0"]})
    result = run_teacher(task, samples, {"s0": [4], "s1": [4]})
    by_sample = result.by_sample()
    assert by_sample["s0"].state == STATE_REUSE1
    assert by_sample["s0"].selected_experts == [4]
    assert by_sample["s1"].state == STATE_RESIDUAL
    assert by_sample["s1"].base_solved is False
    # correctness, not loss, chose the expert
    assert by_sample["s0"].decision_reason == "single_solved_stop_before_pairs"


def test_09_low_nll_with_wrong_prediction_must_not_become_solved():
    samples = ["s0"]
    # expert 4 is the NLL favourite (0.01) but does NOT solve the sample.
    task = FakeTeacherTask(
        base_solved=[],
        singles={},                       # nothing solves it
        nll={"single_04|s0": 0.01},
        base_nll={"s0": 9.0},
    )
    result = run_teacher(task, samples, {"s0": [4, 5]})
    record = result.by_sample()["s0"]
    assert record.state == STATE_RESIDUAL
    assert 4 not in record.selected_experts
    assert record.base_solved is False
    assert record.single_values["4"] == 0.0
    assert record.tested_singles == [4, 5]


def test_10_correct_prediction_with_higher_nll_remains_solved():
    samples = ["s0"]
    # expert 7 solves it with a mediocre NLL; expert 4 has the best NLL but fails.
    task = FakeTeacherTask(
        base_solved=[],
        singles={7: ["s0"]},
        nll={"single_04|s0": 0.01, "single_07|s0": 3.5},
        base_nll={"s0": 9.0},
    )
    result = run_teacher(task, samples, {"s0": [4, 7]})
    record = result.by_sample()["s0"]
    assert record.state == STATE_REUSE1
    assert record.selected_experts == [7]
    assert record.achieved_nll == pytest.approx(3.5)
    assert record.key_targets[7] == TARGET_POSITIVE
    assert record.key_targets[4] == TARGET_NEGATIVE


def test_11_multiple_solved_experts_best_positive_others_ignore_unsolved_negative():
    samples = ["s0"]
    task = FakeTeacherTask(base_solved=[], singles={2: ["s0"], 5: ["s0"], 9: []})
    result = run_teacher(task, samples, {"s0": [2, 5, 6, 9]})
    record = result.by_sample()["s0"]
    assert record.state == STATE_REUSE1
    chosen = record.selected_experts[0]
    assert chosen in (2, 5)
    other = 5 if chosen == 2 else 2
    assert record.key_targets[chosen] == TARGET_POSITIVE
    assert record.key_targets[other] == TARGET_IGNORE
    assert record.key_targets[6] == TARGET_NEGATIVE
    assert record.key_targets[9] == TARGET_NEGATIVE
    # the tie-break is the declared lexicographic order
    assert chosen == 2


def _decide_reuse1(single_values):
    """Run ``_decide`` on a Reuse1 sample with a recall smaller than the search.

    ``recall`` is the sample's own Top-M and is now *diagnostic only*;
    ``visible_experts`` is the capability-search universe STEP C actually scores,
    so ``tested_singles`` can be a strict superset of the recall -- which is
    exactly the situation the oracle must handle.
    """
    teacher = AnswerSupervisedTeacher.__new__(AnswerSupervisedTeacher)
    teacher.config = V8TeacherConfig()
    return teacher._decide(
        sample_id="s0",
        task_id=0,
        spec_solved_value=1.0,
        base_value=0.0,
        base_is_solved=False,
        recall=[10, 11, 12],
        visible_experts=[10, 11, 12, 13],
        shortlist=[10, 11, 12, 13],
        single_values={expert: {"s0": value} for expert, value in single_values},
        pair_values={},
        nll_cache={},
        gain_floor=0.0,
    )


def test_reuse1_targets_are_total_over_every_scored_expert_not_just_the_recall():
    """An expert outside the recall is still *scored*, so the rule still applies.

    Both edge cases here were wrong while the rule iterated ``recall`` alone and
    defaulted the remainder to NEGATIVE: the alternative solver 13 lost its
    IGNORE label, and in the second case the expert the teacher actually
    selected lost its POSITIVE label.
    """
    # 12 is inside the recall and 13 outside it; both solve, so 12 wins the
    # lexicographic tie-break and 13 is an alternative solver -> IGNORE.
    record = _decide_reuse1([(10, 0.0), (11, 0.0), (12, 1.0), (13, 1.0)])
    assert record.state == STATE_REUSE1
    assert record.selected_experts == [12]
    assert record.key_targets[12] == TARGET_POSITIVE
    assert record.key_targets[13] == TARGET_IGNORE
    assert record.key_targets[10] == TARGET_NEGATIVE
    assert record.key_targets[11] == TARGET_NEGATIVE

    # Only 13 solves, and it is not in this sample's recall: the *selected*
    # expert must still be POSITIVE.
    record = _decide_reuse1([(10, 0.0), (11, 0.0), (12, 0.0), (13, 1.0)])
    assert record.state == STATE_REUSE1
    assert record.selected_experts == [13]
    assert record.key_targets[13] == TARGET_POSITIVE
    assert record.key_targets[10] == TARGET_NEGATIVE
    assert record.key_targets[11] == TARGET_NEGATIVE
    assert record.key_targets[12] == TARGET_NEGATIVE


def test_12_pair_search_not_executed_when_a_single_solved():
    samples = ["s0", "s1", "s2"]
    task = FakeTeacherTask(
        base_solved=[],
        singles={1: ["s0"], 2: ["s1"]},
        pairs={(1, 2): ["s0", "s1", "s2"]},
    )
    result = run_teacher(task, samples, {s: [1, 2] for s in samples})
    for sample_id in ("s0", "s1"):
        record = result.by_sample()[sample_id]
        assert record.state == STATE_REUSE1
        assert record.tested_pairs == []
    # only the sample with no solved single reached the pair stage
    pair_routes = [row for row in result.scored_routes if row["route"].startswith("pair")]
    assert len(pair_routes) == 1
    assert pair_routes[0]["route"] == "pair_01_02"
    assert pair_routes[0]["samples"] == 1
    assert result.by_sample()["s2"].state == STATE_REUSE2


def test_13_pair_cannot_be_selected_using_only_nll_improvement():
    samples = ["s0"]
    # the pair has a dramatically better NLL but does not solve the sample.
    task = FakeTeacherTask(
        base_solved=[],
        singles={},
        pairs={},
        nll={"pair_01_02|s0": 0.001, "single_01|s0": 8.0, "single_02|s0": 8.0},
        base_nll={"s0": 9.0},
    )
    result = run_teacher(task, samples, {"s0": [1, 2]})
    record = result.by_sample()["s0"]
    assert record.state == STATE_RESIDUAL
    assert record.selected_experts != [1, 2]
    assert record.tested_pairs == [[1, 2]]      # it was tried ...
    assert record.achieved_value == 0.0         # ... and rejected on the metric


def test_14_pair_must_satisfy_task_metric_contribution_rule():
    samples = ["s0"]
    task = FakeTeacherTask(base_solved=[], singles={}, pairs={(1, 2): ["s0"]})
    # solved pair, zero required margin -> accepted
    accepted = run_teacher(task, samples, {"s0": [1, 2]}, pair_min_metric_gain=0.0)
    assert accepted.by_sample()["s0"].state == STATE_REUSE2
    assert accepted.by_sample()["s0"].selected_experts == [1, 2]
    # an unreachable margin -> the same solved pair is rejected
    rejected = run_teacher(task, samples, {"s0": [1, 2]}, pair_min_metric_gain=1.5)
    record = rejected.by_sample()["s0"]
    assert record.state == STATE_RESIDUAL
    # And the rejected pair is NOT punished.  On a Residual sample "failed
    # alone" is not evidence of "harmful in a composition" (PART 3): the best
    # scored single becomes the context expert -- an attraction target, so its
    # alias key learns to be recallable here -- and every other tested expert is
    # IGNORE.  No expert on a Residual sample is ever NEGATIVE.
    assert record.residual_context == [1]
    assert record.key_targets[1] == TARGET_CONTEXT_POSITIVE
    assert record.key_targets[2] == TARGET_IGNORE
    assert TARGET_NEGATIVE not in set(record.key_targets.values())


def test_15_base_solved_selects_empty_set_and_stops():
    samples = ["s0", "s1"]
    task = FakeTeacherTask(base_solved=["s0"], singles={3: ["s0", "s1"]})
    result = run_teacher(task, samples, {s: [3] for s in samples})
    record = result.by_sample()["s0"]
    assert record.state == STATE_BASE_ONLY
    assert record.selected_experts == []
    assert record.base_solved is True
    assert record.tested_singles == []
    # STEP A short-circuit: expert 3 was never evaluated on s0
    single_rows = [r for r in result.scored_routes if r["route"] == "single_03"]
    assert single_rows and single_rows[0]["solved"] == 1  # only s1 counts
    assert record.key_targets[3] == TARGET_IGNORE
    assert result.state_counts()[STATE_BASE_ONLY] == 1
    assert result.state_counts()[STATE_REUSE1] == 1


# ---------------------------------------------------------------------------
# TEST 16 - TEST 19, TEST 30: per-sample gradient gating
# ---------------------------------------------------------------------------
def _candidate_grad_norm(model, manager, candidates=(3,)) -> float:
    total = 0.0
    for expert_id in candidates:
        for layer in manager.layers.values():
            for parameter in layer.experts[str(expert_id)].parameters():
                if parameter.grad is not None:
                    total += float(parameter.grad.detach().pow(2).sum())
    return total ** 0.5


def _run_gated_batch(state: str, batch_size: int = 4, candidate: int = 3):
    """Forward a single-state batch and report what reached the candidate.

    A covered batch (BaseOnly / Reuse1 / Reuse2) selects only frozen experts, so
    the loss is not merely weighted to zero -- it is *not differentiable at all*,
    because no trainable parameter appears in the graph.  That is the strongest
    possible form of the isolation the specification asks for, so it is reported
    explicitly rather than being asserted away with a zero-gradient check.
    """
    model, manager = make_manager([0, 1, candidate])
    pool = make_pool(num_experts=3)
    enforce_freeze_policy(model, manager, pool, current_task=1,
                          candidate_expert_ids=[candidate])
    sample_ids = [f"s{i}" for i in range(batch_size)]
    if state == STATE_BASE_ONLY:
        experts = {s: [] for s in sample_ids}
    elif state == STATE_REUSE1:
        experts = {s: [0] for s in sample_ids}
    elif state == STATE_REUSE2:
        experts = {s: [0, 1] for s in sample_ids}
    else:
        experts = {s: [candidate] for s in sample_ids}
    states = {s: state for s in sample_ids}
    selection = build_selection(sample_ids, states, experts)
    weights = residual_weights(sample_ids, states)
    inputs = torch.randn(batch_size, 8)
    model.zero_grad(set_to_none=True)
    with use_selection(selection):
        output = model(inputs)
    per_sample = output.pow(2).sum(dim=1)
    loss = residual_answer_loss(per_sample, weights)
    differentiable = bool(loss.requires_grad)
    if differentiable:
        loss.backward()
    candidate_grads = [
        parameter.grad
        for expert_id in (candidate,)
        for layer in manager.layers.values()
        for parameter in layer.experts[str(expert_id)].parameters()
    ]
    return {
        "loss": loss,
        "weights": weights,
        "differentiable": differentiable,
        "candidate_grad_norm": _candidate_grad_norm(model, manager),
        "candidate_grads_present": any(grad is not None for grad in candidate_grads),
    }


def test_16_base_solved_candidate_lora_grad_is_zero():
    report = _run_gated_batch(STATE_BASE_ONLY)
    assert float(report["weights"].sum()) == 0.0
    assert float(report["loss"]) == 0.0
    assert report["differentiable"] is False
    assert report["candidate_grads_present"] is False
    assert report["candidate_grad_norm"] == 0.0


def test_17_reuse1_candidate_lora_grad_is_zero():
    report = _run_gated_batch(STATE_REUSE1)
    assert float(report["weights"].sum()) == 0.0
    assert float(report["loss"]) == 0.0
    assert report["differentiable"] is False
    assert report["candidate_grads_present"] is False
    assert report["candidate_grad_norm"] == 0.0


def test_18_reuse2_candidate_lora_grad_is_zero():
    report = _run_gated_batch(STATE_REUSE2)
    assert float(report["weights"].sum()) == 0.0
    assert float(report["loss"]) == 0.0
    assert report["differentiable"] is False
    assert report["candidate_grads_present"] is False
    assert report["candidate_grad_norm"] == 0.0


def test_19_residual_candidate_lora_receives_gradient():
    report = _run_gated_batch(STATE_RESIDUAL)
    assert float(report["weights"].sum()) > 0
    assert float(report["loss"].detach()) > 0.0
    assert report["differentiable"] is True
    assert report["candidate_grad_norm"] > 0.0


def test_30_gradient_leakage_on_mixed_baseonly_reuse_residual_batch():
    model, manager = make_manager([0, 1, 3])
    pool = make_pool(num_experts=3)
    enforce_freeze_policy(model, manager, pool, current_task=1,
                          candidate_expert_ids=[3])
    candidate_names = [
        f"{layer_name}.experts.3.{name}"
        for layer_name, layer in sorted(manager.layers.items())
        for name, _ in layer.experts["3"].named_parameters()
    ]

    residual_ids = ["r0", "r1"]
    covered_ids = ["c0", "c1", "c2"]
    residual_states = {s: STATE_RESIDUAL for s in residual_ids}
    covered_states = {s: STATE_BASE_ONLY for s in covered_ids[:1]}
    covered_states.update({s: STATE_REUSE1 for s in covered_ids[1:2]})
    covered_states.update({s: STATE_REUSE2 for s in covered_ids[2:]})
    residual_experts = {s: [3] for s in residual_ids}
    covered_experts = {"c0": [], "c1": [0], "c2": [0, 1]}

    generator = torch.Generator().manual_seed(7)
    inputs_residual = torch.randn(len(residual_ids), 8, generator=generator)
    inputs_covered = torch.randn(len(covered_ids), 8, generator=generator)

    def forward_backward(batch):
        sample_ids = batch["sample_ids"]
        states = batch["states"]
        experts = batch["experts"]
        inputs = batch["inputs"]
        selection = build_selection(sample_ids, states, experts)
        weights = residual_weights(sample_ids, states)
        model.zero_grad(set_to_none=True)
        with use_selection(selection):
            output = model(inputs)
        loss = residual_answer_loss(output.pow(2).sum(dim=1), weights)
        loss.backward()
        return {
            name: parameter.grad.detach().clone() if parameter.grad is not None else None
            for name, parameter in model.named_parameters()
            if name in candidate_names
        }

    report = gradient_leakage_probe(
        model,
        candidate_names,
        {"sample_ids": residual_ids, "states": residual_states,
         "experts": residual_experts, "inputs": inputs_residual},
        {"sample_ids": residual_ids + covered_ids,
         "states": {**residual_states, **covered_states},
         "experts": {**residual_experts, **covered_experts},
         "inputs": torch.cat([inputs_residual, inputs_covered], dim=0)},
        forward_backward,
    )
    assert report["leakage"] is False
    assert report["max_abs_delta"] < 1e-6
    assert len(candidate_names) > 0

    # The probe must be able to detect real leakage: gate the covered rows ON.
    # Only ``c0`` (BaseOnly) and ``c1`` (Reuse1) can be relabelled this way --
    # ``c2`` holds two experts, and Residual is capped at one (PART 2), so the
    # relabel would trip the cardinality rule before it could test the gating.
    def forward_backward_leaky(batch):
        sample_ids = batch["sample_ids"]
        leaks = dict(batch["states"])
        for sample_id in ("c0", "c1"):
            leaks[sample_id] = STATE_RESIDUAL       # wrong on purpose
        selection = build_selection(sample_ids, leaks, batch["experts"])
        weights = residual_weights(sample_ids, leaks)
        model.zero_grad(set_to_none=True)
        with use_selection(selection):
            output = model(inputs=batch["inputs"])
        residual_answer_loss(output.pow(2).sum(dim=1), weights).backward()
        return {
            name: parameter.grad.detach().clone() if parameter.grad is not None else None
            for name, parameter in model.named_parameters()
            if name in candidate_names
        }

    with pytest.raises(GradientGatingError):
        gradient_leakage_probe(
            model, candidate_names,
            {"sample_ids": residual_ids, "states": residual_states,
             "experts": residual_experts, "inputs": inputs_residual},
            {"sample_ids": residual_ids + covered_ids,
             "states": {**residual_states, **covered_states},
             "experts": {**residual_experts, **covered_experts},
             "inputs": torch.cat([inputs_residual, inputs_covered], dim=0)},
            forward_backward_leaky,
        )


# ---------------------------------------------------------------------------
# TEST 20 - TEST 23: alias keys
# ---------------------------------------------------------------------------
def _pool_with_alias(expert_ids=(0, 1), task_id=1):
    pool = make_pool(num_experts=2, per_task=1, seed=3)
    for expert_id in expert_ids:
        pool.add_key(expert_id=expert_id, task_id=task_id, key_type="task_alias",
                     value=unit(200 + expert_id), lifecycle="candidate",
                     trainable=True)
    return pool


def test_20_reuse_sample_can_train_historical_alias_key():
    pool = _pool_with_alias()
    queries = {"s0": unit(0), "s1": unit(50)}
    from compose.v8.teacher import TeacherResult, TeacherSampleRecord

    teacher_result = TeacherResult(
        task_id=1,
        records=[
            TeacherSampleRecord(
                sample_id="s0", state=STATE_REUSE1, selected_experts=[0],
                base_solved=False, base_value=0.0, recall=[0], single_values={"0": 1.0},
                pair_values={}, tested_singles=[0], tested_pairs=[],
                achieved_value=1.0, achieved_nll=None, delta_nll=None,
                teacher_gain=1.0, key_targets={"0": TARGET_POSITIVE},
                decision_reason="single_solved_stop_before_pairs",
            ),
        ],
        config={},
    )
    targets = build_key_targets(teacher_result, pool, task_id=1)
    assert "e0_t1_task_alias" in targets
    assert targets["e0_t1_task_alias"].positive_ids == ["s0"]
    # an expert with no positives gets no target bucket at all
    assert "e1_t1_task_alias" not in targets

    report = alias_key_loss(queries, pool, targets)
    assert report.positive_pairs == 1
    assert report.keys_used == 1
    report.total.backward()
    grad = pool.keys["e0_t1_task_alias"].grad
    assert grad is not None and float(grad.abs().sum()) > 0
    assert float(report.mean_positive_similarity) < 0.5  # starts far from the query


def test_21_alternative_solved_expert_receives_no_negative_gradient():
    pool = _pool_with_alias(expert_ids=(0, 1))
    from compose.v8.teacher import TeacherResult, TeacherSampleRecord

    record = TeacherSampleRecord(
        sample_id="s0", state=STATE_REUSE1, selected_experts=[0],
        base_solved=False, base_value=0.0, recall=[0, 1], single_values={"0": 1.0, "1": 1.0},
        pair_values={}, tested_singles=[0, 1], tested_pairs=[],
        achieved_value=1.0, achieved_nll=None, delta_nll=None,
        teacher_gain=1.0,
        key_targets={"0": TARGET_POSITIVE, "1": TARGET_IGNORE},
        decision_reason="single_solved_stop_before_pairs",
    )
    result = TeacherResult(task_id=1, records=[record], config={})
    assert_no_ignore_is_negative(build_key_targets(result, pool, task_id=1))
    targets = build_key_targets(result, pool, task_id=1)
    assert targets["e1_t1_task_alias"].positive_ids == []
    assert targets["e1_t1_task_alias"].negative_ids == []
    assert targets["e1_t1_task_alias"].ignored_ids == ["s0"]

    queries = {"s0": unit(0)}
    report = alias_key_loss(queries, pool, targets)
    report.total.backward()
    assert pool.keys["e1_t1_task_alias"].grad is None or \
        float(pool.keys["e1_t1_task_alias"].grad.abs().sum()) == 0.0
    assert pool.keys["e0_t1_task_alias"].grad is not None


def test_22_zero_support_historical_expert_gets_no_alias_key():
    pool = make_pool(num_experts=4, per_task=2, seed=5)
    from compose.v8.teacher import TeacherResult, TeacherSampleRecord

    records = [
        TeacherSampleRecord(
            sample_id="s0", state=STATE_REUSE1, selected_experts=[1],
            base_solved=False, base_value=0.0, recall=[1, 2], single_values={"1": 1.0},
            pair_values={}, tested_singles=[1, 2], tested_pairs=[],
            achieved_value=1.0, achieved_nll=None, delta_nll=None, teacher_gain=1.0,
            key_targets={"1": TARGET_POSITIVE, "2": TARGET_NEGATIVE},
            decision_reason="single_solved_stop_before_pairs",
        ),
    ]
    result = TeacherResult(task_id=1, records=records, config={})
    queries = {"s0": unit(0)}
    report = create_alias_keys(result, pool, task_id=1, queries_by_sample=queries)
    assert report["num_created"] == 1
    assert "e1_t1_task_alias" in report["created"]
    # expert 2 had support 0: no key, and the reason is recorded
    assert pool.has_alias(2, 1) is False
    assert not any(key_id.startswith("e2_") and "task_alias" in key_id
                   for key_id in pool.key_ids())
    assert report["num_skipped"] == 0      # support 0 is not even a candidate
    assert report["experts_with_support"] == 1

    # threshold above the available support also blocks creation
    blocked = create_alias_keys(result, pool, task_id=2, queries_by_sample=queries,
                                config=type("C", (), {"alias_support_threshold": 5})())
    assert blocked["num_created"] == 0
    assert blocked["skipped"][1].startswith("support 1 below threshold")


def test_22b_current_task_expert_gets_no_alias_key():
    """A task alias key on the expert's *origin* task is not a key V8 has.

    The pool's own invariant says so -- ``validate`` raises "alias key ... sits
    on the origin task; use the origin key" -- so ``create_alias_keys`` must skip
    such experts instead of building a pool that cannot validate.  The campaign's
    all-experts scope is what made this reachable: it is the only pool that
    contains the current task's own experts.
    """
    from compose.v8.teacher import TeacherResult, TeacherSampleRecord

    pool = make_pool(num_experts=4, per_task=2, seed=5)   # experts 2,3 originate on task 1

    def record_for(expert_id: int) -> TeacherResult:
        return TeacherResult(
            task_id=1,
            records=[
                TeacherSampleRecord(
                    sample_id="s0", state=STATE_REUSE1, selected_experts=[expert_id],
                    base_solved=False, base_value=0.0, recall=[expert_id],
                    single_values={str(expert_id): 1.0}, pair_values={},
                    tested_singles=[expert_id], tested_pairs=[], achieved_value=1.0,
                    achieved_nll=None, delta_nll=None, teacher_gain=1.0,
                    key_targets={str(expert_id): TARGET_POSITIVE},
                    decision_reason="single_solved_stop_before_pairs",
                ),
            ],
            config={},
        )

    # expert 2 originates on task 1: no alias key, and the skip carries the reason
    report = create_alias_keys(record_for(2), pool, task_id=1,
                               queries_by_sample={"s0": unit(0)})
    assert report["num_created"] == 0
    assert report["num_skipped"] == 1
    assert "originates on the current task" in report["skipped"][2]
    assert pool.has_alias(2, 1) is False
    pool.validate()          # the pool the old code built could not pass this

    # the same expert under a *different* task is a legitimate historical reuse
    other = create_alias_keys(record_for(2), pool, task_id=0,
                              queries_by_sample={"s0": unit(0)})
    assert other["num_created"] == 1
    assert "e2_t0_task_alias" in other["created"]
    pool.validate()


def test_22c_full_pool_audit_honours_the_history_only_scope():
    """The audit must not test the experts the run's own scope excludes.

    ``recall.json`` stores the **full** committed pool in ``visible_expert_ids``
    and the scope in ``excluded_expert_ids``.  The first version of
    ``v8_full_pool_recall`` read the former as the scope, so a Task-4
    history-only audit tested experts 16-19 -- Task 4's *own* experts -- and
    counted the samples they solved as retrieval failures, which is exactly the
    self-reuse confound the history-only scope exists to remove.  The recorded
    artefact from that version reported ``retrieval_failure: 22`` with
    ``solved_outside_window: [16, 17, 18, 19]`` on its first sample.
    """
    from compose.experiments.v8_full_pool_recall import audit_plan, resolve_scope

    # the shape ``v8_task_run.py:584-585`` actually writes: visible = the pool
    recall = {
        "visible_expert_ids": list(range(22)) + [23],       # 0..21, 23
        "excluded_expert_ids": [16, 17, 18, 19, 20, 21, 23],
    }
    scope = resolve_scope(recall)
    assert scope["pool"] == list(range(22)) + [23]          # raw field preserved
    assert scope["visible"] == list(range(16))              # 0..15 only
    assert not set(scope["visible"]) & set(scope["excluded"])

    # the confound itself: an expert the run excluded is never a test target,
    # even though it sits in the pool field and the teacher left it untested
    record = {"sample_id": "v7_t4_val_0", "tested_singles": [4, 5, 6, 7],
              "recall": [19, 17, 16, 18, 15, 14, 13, 0]}
    plan = audit_plan(recall, record)
    assert not set(plan) & set(recall["excluded_expert_ids"]), plan
    assert 16 not in plan and 19 not in plan

    # and the test is not vacuous in either direction: in-scope experts the
    # teacher already scored are skipped, in-scope ones it did not are tested
    assert plan == [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]
    assert 4 not in plan and 12 in plan

    # a run with no exclusions (the all-experts scope) is unaffected
    assert audit_plan({"visible_expert_ids": [0, 1, 2]}, {"tested_singles": [1]}) == [0, 2]


def test_23_alias_key_centroid_initialization_correct():
    generator = torch.Generator().manual_seed(11)
    positives = torch.randn(5, DIM, generator=generator)
    key = alias_key_init(positives)
    expected = torch.nn.functional.normalize(positives.mean(dim=0), dim=-1)
    assert torch.allclose(key, expected, atol=1e-6)
    assert float(key.norm()) == pytest.approx(1.0, abs=1e-6)
    # by construction the centroid has maximal mean similarity
    similarity = torch.nn.functional.normalize(positives, dim=-1) @ key
    mean_similarity = float(similarity.mean())
    for _ in range(20):
        other = torch.nn.functional.normalize(torch.randn(DIM, generator=generator), dim=-1)
        assert float((torch.nn.functional.normalize(positives, dim=-1) @ other).mean()) \
            <= mean_similarity + 1e-6
    with pytest.raises(MultiKeyPoolError):
        alias_key_init(torch.zeros(0, DIM))


# ---------------------------------------------------------------------------
# TEST 24 - TEST 28: inference purity, cache, resume, migration, no-history
# ---------------------------------------------------------------------------
def test_24_inference_path_never_reads_ground_truth_answer():
    report = assert_inference_purity(REPO / "compose" / "v8" / "inference.py")
    assert report["pure"] is True
    assert report["forbidden_identifiers_found"] == []
    assert report["forbidden_strings_found"] == []

    # the scanner must actually catch a violation
    import compose.v8.inference as inference_module
    bad = "def f(ground_truth):\n    return oracle(ground_truth)\n"
    identifiers, strings = inference_module.collect_code_identifiers(bad)
    assert "ground_truth" in identifiers
    assert inference_module.FORBIDDEN_INFERENCE_IDENTIFIERS & identifiers

    # and it must not be confused by prose that mentions the banned names
    prose = 'def f(x):\n    """Never use the ground-truth answer or the oracle."""\n    return x\n'
    identifiers, _ = inference_module.collect_code_identifiers(prose)
    assert not inference_module.FORBIDDEN_INFERENCE_IDENTIFIERS & identifiers

    # the exemption for the ban's own definition must not become a loophole:
    # the same literal bound to any other name is still caught
    smuggled = 'FORBIDDEN_ANYTHING_ELSE = {"ground_truth"}\n'
    strings = inference_module.collect_code_identifiers(smuggled)[1]
    assert inference_module.FORBIDDEN_INFERENCE_STRINGS & strings
    smuggled_identifier = "def f(x):\n    return oracle(x)\n"
    identifiers = inference_module.collect_code_identifiers(smuggled_identifier)[0]
    assert "oracle" in inference_module.FORBIDDEN_INFERENCE_IDENTIFIERS & identifiers

    # functional check: routing consumes only queries and committed keys
    pool = make_pool(num_experts=4)
    router = V8InferenceRouter(pool)
    queries = torch.randn(3, DIM)
    policy = router.route_policy(queries)
    assert len(policy["rows"]) == 3
    assert all(set(row) == {"expert_ids", "scores", "key_ids"} for row in policy["rows"])
    validate_policy(pool, policy)
    selection = router.selection(queries, ["a", "b", "c"])
    assert selection.batch_size == 3
    assert duplicate_row_count(["a", "b", "c"], selection.expert_ids) == 0


def test_25_teacher_cache_reload_deterministic():
    task = FakeTeacherTask(
        base_solved=["s0"], singles={4: ["s1"]}, pairs={(4, 5): ["s2"]},
        nll={"single_04|s1": 1.5},
    )
    samples = ["s0", "s1", "s2"]
    result = run_teacher(task, samples, {s: [4, 5] for s in samples})
    root = REPO / "experiments" / "runs" / "0911_v8" / "cache_test"
    report = assert_roundtrip(result, root)
    assert report["deterministic"] is True
    reloaded = read_teacher_result(root)
    assert reloaded.state_counts() == result.state_counts()
    assert reloaded.positive_counts() == result.positive_counts()
    assert [r.sample_id for r in reloaded.records] == sorted(samples)

    # ground truth may never be stored in the training cache
    with pytest.raises(TeacherCacheError):
        write_teacher_result(root / "leak", result, provenance={"ground_truth": "x"})
    manifest = json.loads((root / "teacher_manifest.json").read_text())
    assert manifest["stores_ground_truth"] is False


def test_26_resume_preserves_expert_key_mapping():
    pool = make_pool(num_experts=6, per_task=2, seed=13)
    pool.add_key(expert_id=1, task_id=3, key_type="task_alias", value=unit(7),
                 lifecycle="candidate", trainable=True)
    pool.add_key(expert_id=5, task_id=3, key_type="task_alias", value=unit(9),
                 lifecycle="candidate", trainable=True)
    config = V8Config()
    root = REPO / "experiments" / "runs" / "0911_v8" / "resume_test"
    manifest = save_checkpoint(root / "v8_state.pt", pool, config, current_task=3,
                               global_step=17)
    loaded = load_checkpoint(root / "v8_state.pt")
    report = verify_resume_identity(loaded, manifest)
    assert report["status"] == "IDENTICAL"
    assert report["experts"] == 6
    assert report["keys"] == 8
    assert report["alias_keys"] == 2
    resumed_pool = loaded["pool"]
    assert resumed_pool.origin_key_id(1) == pool.origin_key_id(1)
    assert resumed_pool.key_records["e1_t3_task_alias"]["support_count"] == \
        pool.key_records["e1_t3_task_alias"]["support_count"]
    assert resumed_pool.trainable_key_ids() == pool.trainable_key_ids()
    # the resumed key tensors are identical
    for key_id in pool.key_ids():
        assert torch.equal(resumed_pool.keys[key_id], pool.keys[key_id])

    # a tampered manifest must be refused
    tampered = dict(manifest)
    tampered["expert_ids"] = [0, 1, 2]
    with pytest.raises(CheckpointError):
        verify_resume_identity(loaded, tampered)


def test_27_v7_checkpoint_migration_creates_origin_keys_correctly():
    if not V7_FORMAL_KEYS.exists():
        pytest.skip("committed V7 key store is not available on this machine")
    state = torch.load(V7_FORMAL_KEYS, map_location="cpu")
    assert state["schema_version"] == 1
    assert state["query_dim"] == 1536
    # the store is keyed by string expert id in both `keys` and `metadata`
    assert {type(k).__name__ for k in state["keys"]} == {"str"}
    pool = MultiKeyExpertPool.load_v7_pool(state)
    pool.validate()
    assert pool.query_dim == 1536
    assert len(pool.expert_records) == len(state["keys"]) == 24
    for raw_id, tensor in state["keys"].items():
        expert_id = int(raw_id)
        key_id = pool.origin_key_id(expert_id)
        assert torch.equal(pool.keys[key_id], tensor.float())
        record = pool.key_records[key_id]
        assert record["key_type"] == "origin"
        assert record["task_id"] == int(state["metadata"][raw_id]["origin_task"])
        assert record["lifecycle"] == state["metadata"][raw_id]["lifecycle"]
        assert pool.expert_records[expert_id]["origin_task"] == \
            int(state["metadata"][raw_id]["origin_task"])
        # the RMS calibration travels with the expert, not the key
        assert pool.expert_records[expert_id]["rms_state"] is not None
    audit = pool.audit()
    assert audit["num_alias_keys"] == 0
    assert audit["lifecycle_counts"]["pruned"] == 1
    assert audit["lifecycle_counts"]["historical"] == 23
    # 4 experts per origin task, and no expert originates from two tasks
    origins = {e: pool.expert_records[e]["origin_task"] for e in pool.expert_ids()}
    assert sorted(origins.values()) == [t for t in range(6) for _ in range(4)]
    # the pruned expert keeps its origin key but is not live
    pruned = [key_id for key_id, record in pool.key_records.items()
              if record["lifecycle"] == "pruned"]
    assert len(pruned) == 1
    pruned_expert = pool.key_records[pruned[0]]["expert_id"]
    assert pruned_expert not in pool.live_expert_ids()
    # a migration is exact: re-deriving from the same store is byte-identical
    again = MultiKeyExpertPool.load_v7_pool(state)
    for key_id in pool.key_ids():
        assert torch.equal(pool.keys[key_id], again.keys[key_id])


def test_28_task0_no_history_path_runs():
    samples = ["s0", "s1", "s2"]
    # no history at all: recall is empty for every sample
    task = FakeTeacherTask(base_solved=["s0"], singles={}, pairs={})
    result = run_teacher(task, samples, {s: [] for s in samples})
    counts = result.state_counts()
    assert counts[STATE_BASE_ONLY] == 1
    assert counts[STATE_RESIDUAL] == 2
    for record in result.records:
        assert record.tested_singles == []
        assert record.tested_pairs == []
        if record.state == STATE_RESIDUAL:
            assert record.selected_experts == []      # no context to keep
            assert record.achieved_value == 0.0
    # empty recall plus a non-empty recall list must not crash
    result2 = run_teacher(task, samples, {"s0": [], "s1": [], "s2": [0]})
    assert result2.state_counts()[STATE_BASE_ONLY] == 1


def test_29_v7_original_tests_still_pass():
    completed = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/compose/test_v7_global_coevolution.py", "-q"],
        cwd=str(REPO), capture_output=True, text=True, timeout=1800,
    )
    assert completed.returncode == 0, (
        f"V7 regression failed\nstdout:\n{completed.stdout[-4000:]}\n"
        f"stderr:\n{completed.stderr[-2000:]}"
    )
    assert "passed" in completed.stdout


# ---------------------------------------------------------------------------
# supporting invariants the tests above rely on
# ---------------------------------------------------------------------------
def test_metric_adapter_reproduces_official_accuracy_format():
    adapter = TaskMetricAdapter()
    text = "Samples: 3000\nAccuracy: 66.00%\n"
    assert parse_result_value(text, "Accuracy") == pytest.approx(66.0)
    values = [1.0] * 1980 + [0.0] * 1020
    gate = adapter.assert_matches_official(4, values, text, tolerance=0.005)
    assert gate["within_tolerance"] is True
    assert gate["adapter_value"] == pytest.approx(66.0, abs=1e-9)
    # exact match mirrors the official comparison, including whitespace
    assert adapter.is_solved(4, "  3  ", "3") is False
    assert adapter.is_solved(4, "yes", "YES") is True
    # caption tasks are refused rather than approximated
    with pytest.raises(MetricAdapterError):
        adapter.is_solved(5, "a caption", "a caption")
    proxy_name, official_name = adapter.capability_proxy(5)
    assert proxy_name == "v8_caption_capability_proxy_v1"
    assert official_name == "Average"


def test_selection_states_map_to_the_right_composition_scale():
    sample_ids = ["a", "b", "c", "d"]
    states = {"a": STATE_BASE_ONLY, "b": STATE_REUSE1, "c": STATE_REUSE2,
              "d": STATE_RESIDUAL}
    experts = {"a": [], "b": [3], "c": [3, 7], "d": [7]}
    selection = build_selection(sample_ids, states, experts)
    assert selection.expert_ids.tolist() == [[-1, -1, -1, -1], [3, -1, -1, -1],
                                             [3, 7, -1, -1], [7, -1, -1, -1]]
    assert residual_weights(sample_ids, states).tolist() == [0.0, 0.0, 0.0, 1.0]
    with pytest.raises(SelectionError):
        build_selection(["x"], {"x": STATE_REUSE2}, {"x": [3]})
    with pytest.raises(SelectionError):
        build_selection(["x"], {"x": STATE_BASE_ONLY}, {"x": [3]})
    with pytest.raises(SelectionError):
        build_selection(["x"], {"x": STATE_REUSE1}, {"x": [3, 3]})


def test_candidate_recall_and_gap_metrics_are_computable():
    solved = {"s0": [1, 2], "s1": [3]}
    recall = {"s0": [1, 5, 6], "s1": [4, 5, 3]}
    curve = v8_audit.teacher_expert_recall_at_k(solved, recall, ks=(1, 2, 4))
    # Denominator is the number of oracle *solving expert* pairs, not samples:
    # s0 contributes {1, 2} and s1 contributes {3}, so 3 in total.
    assert curve[1] == pytest.approx(1 / 3)   # only expert 1 is inside recall[:1]
    assert curve[2] == pytest.approx(1 / 3)   # expert 2 is absent from s0's list
    assert curve[4] == pytest.approx(2 / 3)   # expert 3 enters at s1's rank 3
    # an empty oracle has no pairs to recall, so the curve is defined and zero
    assert v8_audit.teacher_expert_recall_at_k({"s0": []}, recall, ks=(1,))[1] == 0.0
    full = v8_audit.full_pool_oracle_recall_at_k(solved, recall, ks=(1, 2))
    assert full[2] == curve[2]
    gap = v8_audit.gap_closed(67.97, 74.0, 95.31)
    assert gap["v8_minus_v7"] == pytest.approx(6.03)
    assert gap["teacher_minus_v7"] == pytest.approx(27.34)
    assert gap["gap_closed"] == pytest.approx(6.03 / 27.34, abs=1e-6)
    assert v8_audit.gap_closed(10.0, 10.0, 10.0)["gap_closed"] == 0.0
    diagnosis = v8_audit.diagnose(full_pool_oracle_solves=0, teacher_positives=0,
                                 candidate_recall=0.0, v7_metric=1.0, v8_metric=1.0,
                                 samples=100)
    assert diagnosis["case"] == "CASE_C"


def test_teacher_rejects_nll_solved_threshold_configuration():
    with pytest.raises(ValueError):
        V8TeacherConfig(nll_use_as_solved_threshold=True)
    with pytest.raises(ValueError):
        V8TeacherConfig(solved_signal="answer_nll")
    with pytest.raises(ValueError):
        V8TeacherConfig(use_base_first=False)
    config = V8TeacherConfig()
    assert config.pair_budget(2) == 1
    assert config.pair_budget(4) == 6
    assert config.pair_budget(23) == 6      # bounded by max_pairs


def test_pool_rejects_illegal_states():
    pool = make_pool(num_experts=2, per_task=2, seed=21)
    with pytest.raises(MultiKeyPoolError):
        pool.add_key(expert_id=0, task_id=0, key_type="origin", value=unit(0))
    with pytest.raises(MultiKeyPoolError):
        pool.add_key(expert_id=99, task_id=0, key_type="origin", value=unit(0))
    with pytest.raises(MultiKeyPoolError):
        pool.add_key(expert_id=0, task_id=1, key_type="task_alias", value=torch.zeros(10))
    with pytest.raises(MultiKeyPoolError):
        pool.add_key(expert_id=0, task_id=1, key_type="nonsense", value=unit(0))
    # an alias key may not be placed on the expert's own origin task
    pool.add_key(expert_id=0, task_id=1, key_type="origin", value=unit(1))
    with pytest.raises(MultiKeyPoolError):
        pool.validate()

    pool2 = make_pool(num_experts=2, per_task=2, seed=22)
    pool2.add_key(expert_id=0, task_id=0, key_type="task_alias", value=unit(3))
    with pytest.raises(MultiKeyPoolError):
        pool2.validate()


# ---------------------------------------------------------------------------
# pruning and commit: the end-of-task path
# ---------------------------------------------------------------------------
def test_pruning_retires_zero_support_alias_but_never_strands_an_expert():
    from compose.v8.pruning import (
        PruningError,
        apply_pruning,
        plan_key_pruning,
    )

    pool = make_pool(num_experts=2, per_task=2, seed=31)
    pool.add_key(expert_id=0, task_id=1, key_type="task_alias", value=unit(2),
                 lifecycle="candidate", trainable=True, support_count=40)
    pool.add_key(expert_id=1, task_id=1, key_type="task_alias", value=unit(3),
                 lifecycle="candidate", trainable=True, support_count=0)

    decisions = plan_key_pruning(pool)
    targets = {decision.key_id: decision for decision in decisions}
    assert "e1_t1_task_alias" in targets
    assert "support 0 below threshold 1" in targets["e1_t1_task_alias"].reason
    # the supported key is kept
    assert "e0_t1_task_alias" not in targets
    # origin keys are never proposed: they are the expert's identity
    assert not any(decision.key_type == "origin" for decision in decisions)

    report = apply_pruning(pool, decisions)
    assert report["pruned"] == ["e1_t1_task_alias"]
    assert pool.key_records["e1_t1_task_alias"]["lifecycle"] == "pruned"
    assert pool.key_records["e1_t1_task_alias"]["trainable"] is False
    assert pool.keys["e1_t1_task_alias"].requires_grad is False
    assert "e1_t1_task_alias" not in pool.active_key_ids()
    # the expert keeps its origin key, so it is still reachable
    assert pool.active_key_ids_for_expert(1) == ["e1_t0_origin"]
    assert pool.live_expert_ids() == [0, 1]

    # a redundant alias key (nearly identical to its sibling) is also retired
    pool2 = make_pool(num_experts=1, per_task=1, seed=32)
    twin = pool2.keys["e0_t0_origin"].detach().clone()
    pool2.add_key(expert_id=0, task_id=1, key_type="task_alias", value=twin,
                  lifecycle="candidate", support_count=10)
    redundant = plan_key_pruning(pool2)
    assert [decision.key_id for decision in redundant] == ["e0_t1_task_alias"]
    assert "redundant with a sibling key" in redundant[0].reason
    assert redundant[0].max_similarity_to_sibling > 0.99

    # An expert can never be stranded *by this planner*: every expert must own
    # an origin key (pool.validate), and origin keys are never proposed.  The
    # guard exists for any other caller, so it is exercised directly.
    from compose.v8.pool import LIFECYCLE_PRUNED
    from compose.v8.pruning import assert_pool_not_emptied

    pool3 = make_pool(num_experts=2, per_task=2, seed=33)
    with pytest.raises(PruningError):
        pool3.set_key_lifecycle("e0_t0_origin", LIFECYCLE_PRUNED)
        assert_pool_not_emptied(pool3)
    assert_pool_not_emptied(pool)       # the healthy pool passes

    # V7 migration retains pruned expert/key tombstones for provenance.  They
    # are outside the routing pool and therefore do not violate this invariant.
    pool4 = make_pool(num_experts=2, per_task=2, seed=34)
    pool4.expert_records[1]["lifecycle"] = LIFECYCLE_PRUNED
    pool4.set_key_lifecycle("e1_t0_origin", LIFECYCLE_PRUNED)
    assert_pool_not_emptied(pool4)

    # what a redundant sibling looks like from the candidate side
    from compose.v8.pruning import plan_candidate_pruning
    duplicated = plan_candidate_pruning({"new": unit(0)}, {"old": unit(0)})
    assert duplicated["num_redundant"] == 1
    assert duplicated["redundant"]["new"]["duplicates"] == "old"
    distinct = plan_candidate_pruning({"new": unit(5)}, {"old": unit(0)})
    assert distinct["num_redundant"] == 0


def test_commit_freezes_the_task_and_exports_v7_compatible_keys():
    from compose.v8.commit import (
        CommitError,
        assert_commit_immutable,
        commit_task,
        validate_precommit,
        write_v7_compatible_keys,
    )

    # A task starts with only earlier tasks' keys: both experts originate on
    # task 0, and task 1 is what creates new state.
    pool = make_pool(num_experts=2, per_task=2, seed=41)
    # task 1 creates a candidate expert (2) with its origin key plus an alias
    # key on the historical expert 0
    pool.add_expert(expert_id=2, origin_task=1, lifecycle="candidate")
    pool.add_key(expert_id=2, task_id=1, key_type="origin", value=unit(20),
                 lifecycle="candidate", trainable=True)
    pool.add_key(expert_id=0, task_id=1, key_type="task_alias", value=unit(21),
                 lifecycle="candidate", trainable=True, support_count=7)
    assert set(pool.trainable_key_ids()) == {"e2_t1_origin", "e0_t1_task_alias"}

    assert validate_precommit(pool, 1, [2])["ok"] is True
    # a candidate with no origin key cannot be committed
    pool.add_expert(expert_id=3, origin_task=1, lifecycle="candidate")
    with pytest.raises(CommitError):
        validate_precommit(pool, 1, [2, 3])
    del pool.expert_records[3]

    report = commit_task(pool, task_id=1, candidate_expert_ids=[2])
    assert report.committed_experts == [2]
    assert report.committed_alias_keys == ["e0_t1_task_alias"]
    assert report.committed_origin_keys == ["e2_t1_origin"]
    assert set(report.frozen_keys) == {"e0_t0_origin", "e1_t0_origin"}
    assert report.num_experts_after == 3
    assert report.num_keys_after == 4

    # everything from task 1 is now frozen history
    assert pool.trainable_key_ids() == []
    assert pool.expert_records[2]["lifecycle"] == "historical"
    assert pool.expert_records[2]["creation_task"] == 1
    assert pool.key_records["e2_t1_origin"]["trainable"] is False
    assert pool.keys["e0_t1_task_alias"].requires_grad is False
    # the alias key survived the commit: 1 Expert : N Keys
    assert pool.active_key_ids_for_expert(0) == ["e0_t0_origin", "e0_t1_task_alias"]
    assert assert_commit_immutable(report, pool)["status"] == "IMMUTABLE"
    with pytest.raises(CommitError):
        pool.keys["e0_t1_task_alias"].data.add_(1.0)
        assert_commit_immutable(report, pool)
    pool.keys["e0_t1_task_alias"].data.add_(-1.0)

    # a V7 store carries origin keys only: the evaluator assumes 1 key per expert
    v7_path = REPO / "experiments" / "runs" / "0911_v8" / "commit_test" / "v7_keys.pt"
    summary = write_v7_compatible_keys(pool, v7_path)
    assert summary["experts"] == 3
    assert summary["alias_keys_written"] == 0
    reloaded = torch.load(v7_path, map_location="cpu")
    assert sorted(reloaded["keys"]) == ["0", "1", "2"]
    assert reloaded["metadata"]["0"]["origin_task"] == 0
    assert reloaded["metadata"]["2"]["origin_task"] == 1
    assert torch.equal(reloaded["keys"]["2"], pool.keys["e2_t1_origin"])
    # and the migrated store reads back as the same pool minus the alias keys
    migrated = MultiKeyExpertPool.load_v7_pool(reloaded)
    assert migrated.expert_ids() == pool.expert_ids()
    assert migrated.audit()["num_alias_keys"] == 0


# ---------------------------------------------------------------------------
# Pipeline integration: teacher -> cache -> alias keys -> gated step -> commit
#
# The tests above pin each stage in isolation.  This one pins the *wiring*,
# which no stage-local test can see: a teacher verdict has to survive the
# canonical cache, become a trainable alias key, drive a gated training step
# that reaches the candidate expert and nothing else, survive pruning, and end
# up frozen in a committed pool with every historical tensor byte-identical.
# ---------------------------------------------------------------------------
def test_v8_pipeline_chains_teacher_verdicts_to_a_committed_pool(tmp_path):
    from compose.v8.commit import (
        assert_commit_immutable,
        commit_task,
        write_v7_compatible_keys,
    )
    from compose.v8.pruning import apply_pruning, plan_key_pruning

    sample_ids = ["s0", "s1", "s2", "s3", "s4"]
    recall = {sample_id: [0, 1, 2] for sample_id in sample_ids}

    # Declared capability: expert 0 solves s0/s1 alone, expert 1 *also* solves
    # s0, the (0,1) pair solves s3 (with no single solving it), the base solves
    # s2, and nothing the pool has solves s4.  Expert 2 is recalled everywhere
    # and solves nothing -- the zero-support case.
    task = FakeTeacherTask(
        base_solved=["s2"],
        singles={0: ["s0", "s1"], 1: ["s0"]},
        pairs={(0, 1): ["s3"]},
    )
    teacher = AnswerSupervisedTeacher(TaskMetricAdapter(), V8TeacherConfig())
    result = teacher.run(
        task_id=1,
        sample_ids=sample_ids,
        visible_experts=[0, 1, 2],
        recall_map=recall,
        scorer=task.scorer,
        nll_scorer=task.nll_scorer,
    )

    # -- the teacher's four states, and its four target roles ---------------
    assert result.state_counts() == {
        STATE_BASE_ONLY: 1,
        STATE_REUSE1: 2,
        STATE_REUSE2: 1,
        STATE_RESIDUAL: 1,
    }
    by_sample = result.by_sample()
    assert by_sample["s0"].key_targets[0] == TARGET_POSITIVE
    assert by_sample["s0"].key_targets[1] == TARGET_IGNORE      # also recalls, solved elsewhere
    assert by_sample["s0"].key_targets[2] == TARGET_NEGATIVE    # never solved it
    assert by_sample["s3"].key_targets[0] == TARGET_POSITIVE    # both members of the winning pair
    assert by_sample["s3"].key_targets[1] == TARGET_POSITIVE
    assert by_sample["s4"].state == STATE_RESIDUAL
    assert by_sample["s4"].residual_context == [0], (
        "Residual keeps at most one historical context, chosen by "
        "(metric, NLL, expert_id)"
    )
    assert by_sample["s4"].key_targets[0] == TARGET_CONTEXT_POSITIVE
    assert by_sample["s4"].key_targets[1] == TARGET_IGNORE
    assert by_sample["s4"].key_targets[2] == TARGET_IGNORE
    assert TARGET_NEGATIVE not in set(by_sample["s4"].key_targets.values())

    # -- 1. canonical cache: write, reload, digest-checked -------------------
    cache = tmp_path / "teacher_cache"
    manifest = write_teacher_result(cache, result, provenance={"test": "pipeline"})
    assert manifest["stores_ground_truth"] is False
    restored = read_teacher_result(cache)
    assert restored.state_counts() == result.state_counts()
    assert sorted(restored.by_sample()) == sorted(sample_ids)
    assert restored.positive_counts() == result.positive_counts()

    # -- 2. lazy alias creation ---------------------------------------------
    pool = make_pool(num_experts=3, per_task=3, seed=51)
    queries_by_sample = {
        sample_id: torch.nn.functional.normalize(unit(100 + index), dim=-1)
        for index, sample_id in enumerate(sample_ids)
    }
    created = create_alias_keys(
        restored, pool, task_id=1, queries_by_sample=queries_by_sample
    )
    # Expert 0 solved s0, s1 and the pair with 1 on s3 (solver support 3) and is
    # the context expert kept for the Residual sample s4 -> AliasSupport = 4.
    # Expert 1 solved s3 and is not s4's context -> support 1, solver-only.
    # Expert 2 solved nothing and is not a context -> no key at all.
    assert created["num_created"] == 2, created
    assert created["created"]["e0_t1_task_alias"]["support"] == 4
    assert created["created"]["e0_t1_task_alias"]["solver_support_count"] == 3
    assert created["created"]["e0_t1_task_alias"]["context_support_count"] == 1
    assert created["created"]["e0_t1_task_alias"]["support_source"] == "mixed"
    assert created["created"]["e1_t1_task_alias"]["support"] == 1
    assert created["created"]["e1_t1_task_alias"]["support_source"] == "solver_only"
    assert not pool.has_alias(2, 1), "zero support must not create an alias key"
    assert pool.active_key_ids_for_expert(0) == ["e0_t0_origin", "e0_t1_task_alias"]

    # a key that ended with no support (the pruning path) and a candidate expert
    pool.add_key(expert_id=2, task_id=1, key_type="task_alias", value=unit(2),
                 lifecycle="candidate", trainable=True, support_count=0)
    pool.add_expert(expert_id=3, origin_task=1, lifecycle="candidate")
    pool.add_key(expert_id=3, task_id=1, key_type="origin", value=unit(3),
                 lifecycle="candidate", trainable=True)

    # -- 3. freeze policy, then ledger before any gradient ------------------
    model, manager = make_manager([0, 1, 2, 3])
    raw_lora = {
        expert_id: {
            "{}.{}".format(layer_name, name): tensor
            for layer_name, layer in sorted(manager.layers.items())
            for name, tensor in layer.experts[str(expert_id)].state_dict().items()
        }
        for expert_id in (0, 1, 2)
    }
    report = enforce_freeze_policy(model, manager, pool, current_task=1,
                                   candidate_expert_ids=[3])
    assert report["audit"].ok, report["audit"].render()
    ledger = capture_frozen_ledger(pool, raw_lora, current_task=1)
    assert ledger.key_checksums, "the ledger must cover historical keys"

    # -- 4. one gated training step ----------------------------------------
    states = {
        "s0": STATE_REUSE1, "s1": STATE_REUSE1, "s2": STATE_BASE_ONLY,
        "s3": STATE_REUSE2, "s4": STATE_RESIDUAL,
    }
    experts = {"s0": [0], "s1": [0], "s2": [], "s3": [0, 1], "s4": [3]}
    selection = build_selection(sample_ids, states, experts)
    inputs = torch.randn(len(sample_ids), 8,
                         generator=torch.Generator().manual_seed(3))
    with use_selection(selection):
        outputs = model(inputs)
    weights = residual_weights(sample_ids, states)
    assert weights.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
    loss = residual_answer_loss(outputs.pow(2).mean(dim=1), weights)
    loss.backward()

    candidate = [
        parameter
        for layer in manager.layers.values()
        for parameter in layer.experts["3"].parameters()
    ]
    assert any(parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
               for parameter in candidate), "the candidate expert must receive gradient"
    for expert_id in (0, 1, 2):
        for layer in manager.layers.values():
            for parameter in layer.experts[str(expert_id)].parameters():
                assert parameter.grad is None or not bool(
                    torch.count_nonzero(parameter.grad)
                ), "historical expert {} received gradient".format(expert_id)

    # the alias-key loss reaches the alias keys and only the alias keys
    targets = build_key_targets(restored, pool, 1)
    assert sorted(targets) == [
        "e0_t1_task_alias", "e1_t1_task_alias", "e2_t1_task_alias",
    ]
    assert targets["e0_t1_task_alias"].positive_ids == ["s0", "s1", "s3"]
    # s4's context expert is an attraction target of its own kind, and is NOT
    # also ignored or negative on that sample (PART 3)
    assert targets["e0_t1_task_alias"].context_positive_ids == ["s4"]
    assert targets["e0_t1_task_alias"].support_source == "mixed"
    assert targets["e1_t1_task_alias"].positive_ids == ["s3"]
    # the alternative solver is ignored on s0 (not pushed away from a query it
    # solves), every recalled key is ignored on the base-solved s2, and every
    # non-context expert is ignored on the Residual s4
    assert targets["e1_t1_task_alias"].ignored_ids == ["s0", "s2", "s4"]
    # s1 is the one sample where "tested as a single and did not solve it" *is*
    # evidence -- a Reuse1 sample the pool already covers, so pushing e1's key
    # away from that query removes nothing.  The same label is never applied to
    # the Residual s4, where failure alone is not evidence.
    assert targets["e1_t1_task_alias"].negative_ids == ["s1"]
    # a bucket with no support contributes no loss term (and is pruned below)
    assert targets["e2_t1_task_alias"].positive_ids == []
    assert targets["e2_t1_task_alias"].context_positive_ids == []
    alias_report = alias_key_loss(queries_by_sample, pool, targets)
    assert alias_report.keys_used == 2
    assert float(alias_report.total.detach()) > 0.0
    alias_report.total.backward()
    for key_id in ("e0_t1_task_alias", "e1_t1_task_alias"):
        grad = pool.keys[key_id].grad
        assert grad is not None and bool(torch.count_nonzero(grad)), key_id
    assert pool.keys["e0_t0_origin"].grad is None

    # -- 5. pruning retires the zero-support key ----------------------------
    decisions = plan_key_pruning(pool)
    assert [decision.key_id for decision in decisions] == ["e2_t1_task_alias"]
    apply_pruning(pool, decisions)
    assert pool.key_records["e2_t1_task_alias"]["lifecycle"] == "pruned"
    assert "e2_t1_task_alias" not in pool.active_key_ids()

    # -- 6. commit freezes the task and exports V7-compatible keys ----------
    commit_report = commit_task(pool, task_id=1, candidate_expert_ids=[3])
    assert commit_report.committed_experts == [3]
    assert "e0_t1_task_alias" in commit_report.committed_alias_keys
    assert pool.trainable_key_ids() == []
    assert assert_commit_immutable(commit_report, pool)["status"] == "IMMUTABLE"
    export = write_v7_compatible_keys(pool, tmp_path / "v7_keys.pt")
    assert export["alias_keys_written"] == 0, "a V7 store carries origin keys only"
    assert export["experts"] == 4

    # -- 7. nothing frozen moved -------------------------------------------
    verdict = verify_frozen_ledger(ledger, pool, raw_lora)
    assert verdict["status"] == "UNCHANGED"
    assert verdict["changed_keys"] == [] and verdict["changed_lora"] == []


def test_the_pipeline_test_would_catch_a_freeze_violation(tmp_path):
    """The chain above is only evidence if a violation can fail it."""
    pool = make_pool(num_experts=2, per_task=2, seed=61)
    ledger = capture_frozen_ledger(pool, {}, current_task=0)
    pool.keys["e0_t0_origin"].data.add_(0.25)
    with pytest.raises(GradientGatingError):
        verify_frozen_ledger(ledger, pool, {})


def test_alias_key_ranking_loss_is_a_gradient_carrying_hinge():
    """``L_rank`` must be a hinge on the key, not a constant built from floats.

    Built with basis-vector geometry so every cosine below is exact.  An earlier
    implementation computed the hinge from ``float(...)`` values, which detached
    it from the graph; with ``L_pos`` already at its stationary point (the
    centroid the key is initialised with), ``L_key`` then had a gradient of
    exactly zero and could not train anything.
    """
    from compose.v8.config import V8KeyConfig
    from compose.v8.key_learning import KeyTargets
    from compose.v8.pool import LIFECYCLE_HISTORICAL

    def build(margin: float, competitor: int, negative_query: torch.Tensor):
        pool = MultiKeyExpertPool()
        for expert_id in (0, 1):
            pool.add_expert(expert_id=expert_id, origin_task=0,
                            lifecycle=LIFECYCLE_HISTORICAL)
        pool.add_key(expert_id=0, task_id=0, key_type="origin",
                     value=unit(200), lifecycle=LIFECYCLE_HISTORICAL)
        pool.add_key(expert_id=1, task_id=0, key_type="origin",
                     value=unit(competitor), lifecycle=LIFECYCLE_HISTORICAL)
        key_id = pool.add_key(
            expert_id=0, task_id=1, key_type="task_alias",
            value=torch.nn.functional.normalize(unit(0) + unit(2), dim=-1),
            lifecycle="candidate", trainable=True,
        )
        targets = {key_id: KeyTargets(key_id=key_id, expert_id=0, task_id=1,
                                      positive_ids=["neg"], negative_ids=["neg2"])}
        queries = {"neg": unit(0), "neg2": negative_query}
        return pool, key_id, targets, queries, V8KeyConfig(ranking_margin=margin)

    # margin violated: the negative query sits exactly on the competitor key,
    # while the current key is 45 degrees off the positive query.
    pool, key_id, targets, queries, config = build(0.2, 201, unit(201))
    report = alias_key_loss(queries, pool, targets, config=config)
    assert report.keys_used == 1 and report.ranking_pairs == 1
    assert float(report.positive) == pytest.approx(1.0 - 0.5 ** 0.5, abs=1e-6)
    assert float(report.ranking) == pytest.approx(0.2 - 0.0 + 1.0, abs=1e-6)
    assert float(report.total) == pytest.approx(
        1.0 * (1.0 - 0.5 ** 0.5) + 0.1 * 1.2, abs=1e-6
    )
    report.total.backward()
    grad = pool.keys[key_id].grad
    assert grad is not None and bool(torch.count_nonzero(grad))
    # the competitor is what the current key is pushed away from, but it is
    # historical: it is detached, so it receives no gradient at all
    assert pool.keys["e1_t0_origin"].grad is None
    assert pool.keys["e0_t0_origin"].grad is None

    # margin already satisfied: the key claims the negative query outright and
    # the nearest competitor is orthogonal, so the hinge is clamped off (exactly
    # zero, not merely small) and a correctly-separated key is left alone
    aligned = torch.nn.functional.normalize(unit(0) + unit(2), dim=-1)
    pool, key_id, targets, queries, config = build(0.2, 300, aligned)
    satisfied = alias_key_loss(queries, pool, targets, config=config)
    assert satisfied.ranking_pairs == 1
    assert float(satisfied.ranking) == 0.0


# ---------------------------------------------------------------------------
# The training loop itself: compose/v8/trainer.py
# ---------------------------------------------------------------------------
def _trainer_scenario(tmp_path):
    """A 4-expert pool, one candidate, five samples, one alias key per solver."""
    from compose.v8.trainer import TrainBatch

    sample_ids = ["s0", "s1", "s2", "s3", "s4"]
    recall = {sample_id: [0, 1, 2] for sample_id in sample_ids}
    task = FakeTeacherTask(
        base_solved=["s2"],
        singles={0: ["s0", "s1"], 1: ["s0"]},
        pairs={(0, 1): ["s3"]},
    )
    teacher = AnswerSupervisedTeacher(TaskMetricAdapter(), V8TeacherConfig())
    result = teacher.run(
        task_id=1, sample_ids=sample_ids, visible_experts=[0, 1, 2],
        recall_map=recall, scorer=task.scorer, nll_scorer=task.nll_scorer,
    )
    pool = make_pool(num_experts=3, per_task=3, seed=51)
    queries_by_sample = {
        sample_id: torch.nn.functional.normalize(unit(100 + index), dim=-1)
        for index, sample_id in enumerate(sample_ids)
    }
    create_alias_keys(result, pool, task_id=1, queries_by_sample=queries_by_sample)
    # a key that ends up with no support: created here, pruned in finalize()
    pool.add_key(expert_id=2, task_id=1, key_type="task_alias", value=unit(2),
                 lifecycle="candidate", trainable=True, support_count=0)
    pool.add_expert(expert_id=3, origin_task=1, lifecycle="candidate")
    pool.add_key(expert_id=3, task_id=1, key_type="origin", value=unit(3),
                 lifecycle="candidate", trainable=True)

    model, manager = make_manager([0, 1, 2, 3])
    states = {
        "s0": STATE_REUSE1, "s1": STATE_REUSE1, "s2": STATE_BASE_ONLY,
        "s3": STATE_REUSE2, "s4": STATE_RESIDUAL,
    }
    experts = {"s0": [0], "s1": [0], "s2": [], "s3": [0, 1], "s4": [3]}
    inputs = torch.randn(len(sample_ids), 8,
                         generator=torch.Generator().manual_seed(7))
    order = {sample_id: index for index, sample_id in enumerate(sample_ids)}

    def forward_fn(requested):
        rows = [order[str(sample_id)] for sample_id in requested]
        return model(inputs[rows]).pow(2).mean(dim=1)

    batch = TrainBatch(list(sample_ids), states, experts)
    return dict(model=model, manager=manager, pool=pool, result=result,
                queries=queries_by_sample, states=states, experts=experts,
                batch=batch, forward_fn=forward_fn, teacher=result)


def test_v8_trainer_trains_the_candidate_and_the_alias_keys_only(tmp_path):
    from compose.v8.trainer import V8TaskTrainer
    from compose.v8.pruning import plan_key_pruning

    scenario = _trainer_scenario(tmp_path)
    pool, manager, model = scenario["pool"], scenario["manager"], scenario["model"]
    historical = {
        expert_id: {
            "{}.{}".format(layer_name, name): tensor.detach().clone()
            for layer_name, layer in sorted(manager.layers.items())
            for name, tensor in layer.experts[str(expert_id)].state_dict().items()
        }
        for expert_id in (0, 1, 2)
    }
    candidate_before = {
        name: tensor.detach().clone()
        for layer in manager.layers.values()
        for name, tensor in layer.experts["3"].state_dict().items()
    }

    trainer = V8TaskTrainer(
        model=model, manager=manager, pool=pool, config=V8Config(),
        current_task=1, candidate_expert_ids=[3],
        forward_fn=scenario["forward_fn"],
        queries_by_sample=scenario["queries"],
    )
    report = trainer.train_epoch([scenario["batch"]], teacher_result=scenario["result"])

    # gating: only the residual sample carried weight, the other four did not
    assert report.residual_samples == 1 and report.covered_samples == 4
    assert report.states == {
        STATE_BASE_ONLY: 1, STATE_REUSE1: 2, STATE_REUSE2: 1, STATE_RESIDUAL: 1,
    }
    assert all(entry["mismatched_samples"] == [] for entry in report.gating)
    assert report.gating[0]["gradient_active_counts"] == {STATE_RESIDUAL: 1}

    # the trainable surface actually moved, and only it
    assert report.gradient_experts == [3], report.gradient_experts
    # Both supported keys are refined -- e1's has a single positive and is
    # moved purely by the ranking term, which is the signal the detached-hinge
    # bug in §9.1 removed.  e2's key has no positives, so it gets nothing.
    assert report.gradient_keys == [
        "e0_t1_task_alias", "e1_t1_task_alias",
    ], report.gradient_keys
    assert report.frozen_gradient_offenders == []
    assert any(
        not torch.equal(candidate_before[name], tensor)
        for layer in manager.layers.values()
        for name, tensor in layer.experts["3"].state_dict().items()
    ), "the candidate expert must actually be updated"
    for expert_id in (0, 1, 2):
        for layer_name, layer in sorted(manager.layers.items()):
            for name, tensor in layer.experts[str(expert_id)].state_dict().items():
                key = "{}.{}".format(layer_name, name)
                assert torch.equal(historical[expert_id][key], tensor), key

    # finalize: prune the support-less key, verify the ledger, commit
    assert [decision.key_id for decision in plan_key_pruning(pool)] == ["e2_t1_task_alias"]
    outcome = trainer.finalize(checkpoint_path=tmp_path / "v8_checkpoint.pt")
    assert outcome["pruned_keys"] == ["e2_t1_task_alias"]
    assert outcome["ledger"]["status"] == "UNCHANGED"
    assert outcome["ledger"]["changed_keys"] == []
    assert outcome["ledger"]["changed_lora"] == []
    assert outcome["commit"]["committed_experts"] == [3]
    assert "e0_t1_task_alias" in outcome["commit"]["committed_alias_keys"]
    assert pool.trainable_key_ids() == []
    # a reopened checkpoint restores the same pool identity
    assert Path(outcome["checkpoint"] and tmp_path / "v8_checkpoint.pt").is_file()
    from compose.v8.checkpoint import load_checkpoint, verify_resume_identity
    loaded = load_checkpoint(tmp_path / "v8_checkpoint.pt", current_task=1)
    assert verify_resume_identity(loaded, outcome["checkpoint"])["status"] == "IDENTICAL"


def test_v8b_residual_route_appends_candidate_and_trains_its_origin_key(tmp_path):
    """The real V8-B route is old context + current candidate, never context alone."""
    from compose.v8.trainer import TrainBatch, V8TaskTrainer

    scenario = _trainer_scenario(tmp_path)
    trainer = V8TaskTrainer(
        model=scenario["model"], manager=scenario["manager"], pool=scenario["pool"],
        config=V8Config(), current_task=1, candidate_expert_ids=[3],
        forward_fn=scenario["forward_fn"], queries_by_sample=scenario["queries"],
    )
    batch = TrainBatch(
        ["s4"], {"s4": STATE_RESIDUAL}, {"s4": [0]},
        candidate_by_sample={"s4": 3},
    )
    selection = trainer._training_selection(batch)
    assert selection.expert_ids.tolist() == [[0, 3, -1, -1]]
    before = scenario["pool"].keys["e3_t1_origin"].detach().clone()
    report = trainer.train_epoch([batch], teacher_result=scenario["result"])
    assert "e3_t1_origin" in report.gradient_keys
    assert not torch.equal(before, scenario["pool"].keys["e3_t1_origin"].detach())


def test_v8b_checkpoint_replay_keeps_candidate_selection_active(tmp_path):
    """Reentrant checkpoint replay must not silently fall back to backbone-only."""
    from torch.utils.checkpoint import checkpoint
    from compose.v8.trainer import TrainBatch, V8TaskTrainer

    scenario = _trainer_scenario(tmp_path)
    eager_forward = scenario["forward_fn"]

    def checkpointed_forward(sample_ids):
        sentinel = torch.ones((), requires_grad=True)
        return checkpoint(
            lambda _sentinel: eager_forward(sample_ids),
            sentinel,
            use_reentrant=True,
        )

    trainer = V8TaskTrainer(
        model=scenario["model"], manager=scenario["manager"], pool=scenario["pool"],
        config=V8Config(), current_task=1, candidate_expert_ids=[3],
        forward_fn=checkpointed_forward, queries_by_sample=scenario["queries"],
    )
    batch = TrainBatch(
        ["s4"], {"s4": STATE_RESIDUAL}, {"s4": [0]},
        candidate_by_sample={"s4": 3},
    )
    report = trainer.train_epoch([batch], teacher_result=scenario["result"])

    assert report.gradient_experts == [3]
    assert all(
        layer._checkpoint_replay_selection is None
        for layer in scenario["manager"].layers.values()
    )


def test_v8b_candidate_origin_key_is_in_the_explicit_whitelist(tmp_path):
    scenario = _trainer_scenario(tmp_path)
    from compose.v8.trainer import V8TaskTrainer

    trainer = V8TaskTrainer(
        model=scenario["model"], manager=scenario["manager"], pool=scenario["pool"],
        config=V8Config(), current_task=1, candidate_expert_ids=[3],
        forward_fn=scenario["forward_fn"], queries_by_sample=scenario["queries"],
    )
    assert "e3_t1_origin" in trainer.whitelist
    assert trainer.pool.keys["e3_t1_origin"].requires_grad
    assert trainer.freeze_report["audit"].trainable_candidate_keys == ["e3_t1_origin"]


def test_v8_tensor_checksum_accepts_bfloat16_without_dtype_conversion():
    from compose.v8.pool import tensor_checksum

    value = torch.arange(8, dtype=torch.float32).to(torch.bfloat16)
    assert tensor_checksum(value) == tensor_checksum(value.clone())
    changed = value.clone()
    changed[0] = 9
    assert tensor_checksum(value) != tensor_checksum(changed)


def test_v8_trainer_leakage_probe_shows_mixed_batches_change_nothing(tmp_path):
    from compose.v8.trainer import TrainBatch, V8TaskTrainer

    scenario = _trainer_scenario(tmp_path)
    trainer = V8TaskTrainer(
        model=scenario["model"], manager=scenario["manager"], pool=scenario["pool"],
        config=V8Config(), current_task=1, candidate_expert_ids=[3],
        forward_fn=scenario["forward_fn"],
    )
    residual_only = TrainBatch(["s4"], {"s4": STATE_RESIDUAL}, {"s4": [3]})
    mixed = TrainBatch(
        ["s4", "s2", "s0"],
        {"s4": STATE_RESIDUAL, "s2": STATE_BASE_ONLY, "s0": STATE_REUSE1},
        {"s4": [3], "s2": [], "s0": [0]},
    )
    report = trainer.leakage_probe(residual_only, mixed)
    assert report["leakage"] is False
    # Not exactly 0: covering samples adds zero-weight terms, and float addition
    # reassociates.  1.5e-08 is the noise floor, four orders below the 1e-6
    # threshold -- an actual leak (see below) moves the gradient by ~1e-1.
    assert report["max_abs_delta"] < 1e-6
    # a probe that cannot fail would be worthless: giving a covered sample the
    # residual weight by hand must move the candidate gradient, and the probe
    # must raise rather than report
    with pytest.raises(GradientGatingError, match="gradient leakage"):
        trainer.leakage_probe(
            residual_only,
            TrainBatch(
                ["s4", "s0"],
                {"s4": STATE_RESIDUAL, "s0": STATE_RESIDUAL},
                {"s4": [3], "s0": [3]},
            ),
        )


# ---------------------------------------------------------------------------
# PART 10 (oracle-semantics round): tests A - L
#
# The first 45 tests encode the *mechanism*; these encode the three finalised
# method decisions and are named by letter so the acceptance report can cite
# them one-to-one.  Each one fails if its decision is silently reverted.
# ---------------------------------------------------------------------------
def test_oracle_A_full_history_search_finds_an_expert_the_keys_rank_outside_top_m():
    """A: key recall must not decide who is eligible for answer supervision.

    Expert 9 solves ``s0`` and is ranked *outside* the router's Top-M -- the
    defining case: if the search were bounded by the recall, 9 would never be
    scored, never be selected, and never earn the alias key that fixes exactly
    this ranking deficit.  The failure would be invisible and self-confirming.
    """
    samples = ["s0"]
    recall = {"s0": [0, 1, 2, 3, 4]}          # 9 is deliberately absent
    task = FakeTeacherTask(base_solved=[], singles={9: ["s0"]})
    visible = [0, 1, 2, 3, 4, 9]
    result = run_teacher(task, samples, recall, visible=visible)

    record = result.by_sample()["s0"]
    assert 9 not in record.recall, "the fixture must keep 9 outside the Top-M"
    assert record.historical_experts_tested == visible
    assert record.state == STATE_REUSE1
    assert record.selected_experts == [9]
    assert record.key_targets[9] == TARGET_POSITIVE
    assert record.key_targets[0] == TARGET_NEGATIVE
    # the recall is still recorded -- it is the quantity key quality is judged
    # by -- but only as a diagnostic
    assert record.router_top_m == [0, 1, 2, 3, 4]
    assert result.coverage_report()["full_coverage"] is True


def test_oracle_B_every_visible_single_must_be_scored_or_the_run_hard_fails():
    """B: the oracle invariant is asserted per sample, not argued from the code."""
    from compose.v8.teacher import (
        TeacherSampleRecord,
        assert_full_history_coverage,
    )

    def record(sample_id, base_solved, tested):
        return TeacherSampleRecord(
            sample_id=sample_id, state=STATE_RESIDUAL, selected_experts=[],
            base_solved=base_solved, base_value=0.0, recall=[], single_values={},
            pair_values={}, tested_singles=list(tested), tested_pairs=[],
            achieved_value=0.0, achieved_nll=None, delta_nll=None,
            teacher_gain=0.0, key_targets={}, decision_reason="test",
            historical_experts_visible=[0, 1, 2, 3],
            historical_experts_tested=list(tested),
        )

    # the honest case passes
    assert_full_history_coverage(
        [record("s0", False, [0, 1, 2, 3]), record("s1", True, [])], [0, 1, 2, 3]
    )

    # one expert left unscored -> the search was bounded, and it must not pass
    with pytest.raises(TeacherError, match="full visible"):
        assert_full_history_coverage([record("s0", False, [0, 1, 2])], [0, 1, 2, 3])

    # scoring something outside the visible pool is equally a bug
    with pytest.raises(TeacherError, match="full visible"):
        assert_full_history_coverage(
            [record("s0", False, [0, 1, 2, 3, 7])], [0, 1, 2, 3]
        )

    # STEP A stops before the search: a BaseOnly sample must have tested nothing
    with pytest.raises(TeacherError, match="BaseOnly"):
        assert_full_history_coverage([record("s0", True, [0])], [0, 1, 2, 3])


def test_oracle_C_residual_context_expert_gains_alias_support_and_is_never_negative():
    """C: a Residual sample's context expert is an attraction target (DECISION-2)."""
    from compose.v8.teacher import TeacherResult, TeacherSampleRecord

    pool = make_pool(num_experts=2, per_task=1, seed=3)
    queries = {"s0": unit(5), "s1": unit(60)}
    result = TeacherResult(
        task_id=1,
        records=[
            TeacherSampleRecord(
                sample_id="s0", state=STATE_RESIDUAL, selected_experts=[],
                base_solved=False, base_value=0.0, recall=[0, 1],
                single_values={"0": 0.0, "1": 0.0}, pair_values={},
                tested_singles=[0, 1], tested_pairs=[[0, 1]],
                achieved_value=0.0, achieved_nll=None, delta_nll=None,
                teacher_gain=0.0,
                key_targets={"0": TARGET_CONTEXT_POSITIVE, "1": TARGET_IGNORE},
                residual_context=[0],
                decision_reason="no_single_or_pair_solved_residual",
                historical_experts_visible=[0, 1],
                historical_experts_tested=[0, 1],
            ),
        ],
        config={},
    )

    assert_no_ignore_is_negative(build_key_targets(result, pool, task_id=1))

    created = create_alias_keys(result, pool, task_id=1, queries_by_sample=queries)
    assert created["num_created"] == 1
    payload = created["created"]["e0_t1_task_alias"]
    assert payload["support"] == 1
    assert payload["context_support_count"] == 1
    assert payload["solver_support_count"] == 0
    assert payload["support_source"] == "context_only"
    assert pool.has_alias(0, 1), "the context expert must earn a callable alias key"
    assert not pool.has_alias(1, 1), "a non-context expert gets no support"

    targets = build_key_targets(result, pool, task_id=1)
    bucket = targets["e0_t1_task_alias"]
    assert bucket.context_positive_ids == ["s0"]
    assert bucket.positive_ids == []
    assert bucket.support == 1
    assert "s0" not in bucket.negative_ids
    assert "s0" not in bucket.ignored_ids

    # a context-positive expert must also be rejected if it is simultaneously
    # labelled negative on the same sample: attraction and repulsion on one
    # (query, key) pair is a contradiction, not a trade-off
    bad = TeacherResult(
        task_id=1,
        records=[
            TeacherSampleRecord(
                sample_id="s0", state=STATE_RESIDUAL, selected_experts=[],
                base_solved=False, base_value=0.0, recall=[], single_values={},
                pair_values={}, tested_singles=[0], tested_pairs=[],
                achieved_value=0.0, achieved_nll=None, delta_nll=None,
                teacher_gain=0.0,
                key_targets={"0": TARGET_CONTEXT_POSITIVE},
                decision_reason="test",
            ),
        ],
        config={},
    )
    conflicted = build_key_targets(bad, pool, task_id=1)
    conflicted["e0_t1_task_alias"].negative_ids.append("s0")
    with pytest.raises(Exception, match="positive and"):
        assert_no_ignore_is_negative(conflicted)


def test_oracle_D_residual_context_produces_real_alias_key_gradient():
    """D: the gradient, not the presence of the parameter, is the evidence.

    "The alias tensor has a gradient in this batch" proves nothing -- a batch
    containing any solver positive would do that.  What has to hold is that this
    sample's context role alone moves the key, and that the move is *towards*
    the sample's query.
    """
    pool = make_pool(num_experts=2, per_task=1, seed=3)
    pool.add_key(expert_id=0, task_id=1, key_type="task_alias",
                 value=torch.nn.functional.normalize(unit(300), dim=-1),
                 lifecycle="candidate", trainable=True)
    origin_before = pool.keys["e0_t0_origin"].detach().clone()
    query = unit(0)
    targets = build_key_targets(
        _residual_result(context_expert=0), pool, task_id=1
    )

    key_before = pool.keys["e0_t1_task_alias"].detach().clone()
    before = float(torch.nn.functional.cosine_similarity(
        torch.nn.functional.normalize(key_before, dim=-1), query, dim=-1))
    report = alias_key_loss({"s0": query, "s1": unit(90)}, pool, targets,
                            config=V8KeyConfig(lambda_context_positive=1.0,
                                               lambda_solver_positive=1.0,
                                               lambda_rank=0.0))
    assert report.context_positive_pairs == 1
    assert float(report.context_positive.detach()) > 0.0
    assert float(report.solver_positive.detach()) == 0.0
    report.total.backward()

    grad = pool.keys["e0_t1_task_alias"].grad
    assert grad is not None and float(grad.abs().sum()) > 0.0, (
        "the context expert's alias key must receive a real gradient"
    )
    # the origin key is historical: in the graph as a detached constant or not
    # at all, never a leaf that accumulates
    assert pool.keys["e0_t0_origin"].grad is None
    assert torch.equal(origin_before, pool.keys["e0_t0_origin"].detach())

    # and the gradient points the right way: a step down the loss raises the
    # similarity to the query
    with torch.no_grad():
        pool.keys["e0_t1_task_alias"].add_(grad, alpha=-0.1)
    after = float(torch.nn.functional.cosine_similarity(
        torch.nn.functional.normalize(pool.keys["e0_t1_task_alias"].detach(), dim=-1),
        query, dim=-1))
    assert after > before, (before, after)


def _residual_result(context_expert: int, task_id: int = 1):
    from compose.v8.teacher import TeacherResult, TeacherSampleRecord

    return TeacherResult(
        task_id=task_id,
        records=[
            TeacherSampleRecord(
                sample_id="s0", state=STATE_RESIDUAL, selected_experts=[],
                base_solved=False, base_value=0.0, recall=[0, 1],
                single_values={"0": 0.0, "1": 0.0}, pair_values={},
                tested_singles=[0, 1], tested_pairs=[[0, 1]],
                achieved_value=0.0, achieved_nll=None, delta_nll=None,
                teacher_gain=0.0,
                key_targets={
                    str(context_expert): TARGET_CONTEXT_POSITIVE,
                    **{
                        str(other): TARGET_IGNORE
                        for other in (0, 1) if other != context_expert
                    },
                },
                residual_context=[int(context_expert)],
                decision_reason="no_single_or_pair_solved_residual",
                historical_experts_visible=[0, 1],
                historical_experts_tested=[0, 1],
            ),
        ],
        config={},
    )


def _alias_grad_case(state: str, selected: Sequence[int], positives: Mapping[int, str]):
    """Run the key loss for one declared sample and report the alias gradients."""
    from compose.v8.teacher import TeacherResult, TeacherSampleRecord

    pool = _pool_with_alias(expert_ids=(0, 1))
    queries = {"s0": unit(0), "s1": unit(70)}
    record = TeacherSampleRecord(
        sample_id="s0", state=state, selected_experts=list(selected),
        base_solved=False, base_value=0.0, recall=[0, 1],
        single_values={"0": 1.0, "1": 1.0}, pair_values={},
        tested_singles=[0, 1], tested_pairs=[],
        achieved_value=1.0, achieved_nll=None, delta_nll=None,
        teacher_gain=1.0,
        key_targets={str(k): v for k, v in positives.items()},
        decision_reason="test",
    )
    result = TeacherResult(task_id=1, records=[record], config={})
    targets = build_key_targets(result, pool, task_id=1)
    report = alias_key_loss(queries, pool, targets)
    if float(report.total.detach()) > 0.0:
        report.total.backward()
    return pool, targets, report


def test_oracle_E_reuse1_selected_solver_alias_key_receives_gradient():
    """E: the state table's Reuse1 row: candidate 0, selected solver alias +."""
    pool, targets, report = _alias_grad_case(
        STATE_REUSE1, selected=[0],
        positives={0: TARGET_POSITIVE, 1: TARGET_IGNORE},
    )
    assert targets["e0_t1_task_alias"].positive_ids == ["s0"]
    grad = pool.keys["e0_t1_task_alias"].grad
    assert grad is not None and float(grad.abs().sum()) > 0.0
    # the alternative solver is IGNORE, so its key is not in the graph
    assert "e1_t1_task_alias" not in targets or not targets["e1_t1_task_alias"].positive_ids
    assert pool.keys["e1_t1_task_alias"].grad is None


def test_oracle_F_reuse2_both_selected_aliases_receive_gradient():
    """F: the state table's Reuse2 row: candidate 0, both selected aliases +."""
    pool, targets, report = _alias_grad_case(
        STATE_REUSE2, selected=[0, 1],
        positives={0: TARGET_POSITIVE, 1: TARGET_POSITIVE},
    )
    assert targets["e0_t1_task_alias"].positive_ids == ["s0"]
    assert targets["e1_t1_task_alias"].positive_ids == ["s0"]
    for key_id in ("e0_t1_task_alias", "e1_t1_task_alias"):
        grad = pool.keys[key_id].grad
        assert grad is not None and float(grad.abs().sum()) > 0.0, key_id


def test_oracle_G_baseonly_batch_trains_no_key_and_leaves_the_candidate_alone(tmp_path):
    """G: the state table's BaseOnly row: 0 key gradient, 0 candidate update.

    This is the test that requires the key loss to be restricted to the batch.
    With a global objective the BaseOnly batch below would still move a key,
    through some other sample's positive, and the table would be a statement
    about intent rather than about the run.
    """
    from compose.v8.trainer import TrainBatch, V8TaskTrainer

    scenario = _trainer_scenario(tmp_path)
    pool, manager = scenario["pool"], scenario["manager"]
    keys_before = {
        key_id: pool.keys[key_id].detach().clone() for key_id in pool.key_ids()
    }
    # keyed by layer *and* name: the fixture has two layers with identically
    # named LoRA tensors, so a name-only dict would collide and compare layer 0
    # against layer 0 twice
    candidate_before = {
        "{}.{}".format(layer_name, name): tensor.detach().clone()
        for layer_name, layer in sorted(manager.layers.items())
        for name, tensor in layer.experts["3"].state_dict().items()
    }

    trainer = V8TaskTrainer(
        model=scenario["model"], manager=manager, pool=pool, config=V8Config(),
        current_task=1, candidate_expert_ids=[3],
        forward_fn=scenario["forward_fn"],
        queries_by_sample=scenario["queries"],
    )
    only_base = TrainBatch(["s2"], {"s2": STATE_BASE_ONLY}, {"s2": []})
    report = trainer.train_epoch([only_base], teacher_result=scenario["result"])

    assert report.states == {STATE_BASE_ONLY: 1}
    assert report.gradient_keys == [], report.gradient_keys
    assert report.gradient_experts == []
    assert report.key_batches == 0, "a BaseOnly batch has no key targets at all"
    assert report.key_role_samples.get("context_positive", 0) == 0
    for key_id, value in keys_before.items():
        assert torch.equal(value, pool.keys[key_id].detach()), key_id
    for layer_name, layer in sorted(manager.layers.items()):
        for name, tensor in layer.experts["3"].state_dict().items():
            key = "{}.{}".format(layer_name, name)
            assert torch.equal(candidate_before[key], tensor), key


def test_oracle_H_residual_cardinality_is_at_most_one_context_expert():
    """H: capped at one historical context, so at most two experts in total."""
    from compose.v8.selection import (
        MAX_RESIDUAL_ACTIVE_EXPERTS,
        validate_state,
    )

    assert MAX_RESIDUAL_ACTIVE_EXPERTS == 2
    assert STATE_CARDINALITY[STATE_RESIDUAL] == (0, 1)
    validate_state(STATE_RESIDUAL, [])
    validate_state(STATE_RESIDUAL, [3])
    with pytest.raises(SelectionError, match="requires"):
        validate_state(STATE_RESIDUAL, [0, 1])
    with pytest.raises(SelectionError, match="requires"):
        validate_state(STATE_RESIDUAL, [0, 1, 2])

    # a Residual batch row is the context expert plus nothing: the candidate's
    # second slot comes from the training route, not from the state
    from compose.v8.trainer import TrainBatch
    batch = TrainBatch(["s0"], {"s0": STATE_RESIDUAL}, {"s0": [0]})
    assert len(batch.experts_by_sample["s0"]) == 1

    # and the teacher itself never emits a pair as context
    result = run_teacher(
        FakeTeacherTask(base_solved=[], singles={}, pairs={}),
        ["s0", "s1"], {"s0": [0, 1], "s1": [0, 1]}, visible=[0, 1],
    )
    for record in result.records:
        assert record.state == STATE_RESIDUAL
        assert len(record.residual_context) <= 1


def test_oracle_I_three_keys_on_one_expert_still_occupy_one_top2_slot():
    """I: the inference budget is per *expert*, so an extra key buys no slot."""
    pool = make_pool(num_experts=3, per_task=3, seed=21)
    pool.add_key(expert_id=0, task_id=1, key_type="task_alias", value=unit(10),
                 lifecycle="candidate", trainable=False)
    pool.add_key(expert_id=0, task_id=2, key_type="task_alias", value=unit(11),
                 lifecycle="candidate", trainable=False)
    assert len(pool.active_key_ids_for_expert(0)) == 3

    router = MultiKeyRouter()
    queries = torch.stack([unit(10), unit(11), unit(12)])
    result = router(queries, pool)
    assert result.expert_ids.shape[1] == 2
    for row in result.expert_ids.tolist():
        assert len(set(row)) == 2, "one expert must not fill both slots"
        assert 0 in row  # three keys make expert 0 impossible to miss
    assert duplicate_row_count(["a", "b", "c"], result.expert_ids) == 0
    assert distinct_expert_rate(result.expert_ids) == 1.0

    # the router is still exactly max-per-expert: the repeated-key expert scores
    # its *best* key, never the sum
    per_expert = result.per_expert_scores
    column = result.pool_expert_ids.tolist().index(0)
    keys = torch.stack([
        torch.nn.functional.normalize(pool.keys[key_id].float(), dim=-1)
        for key_id in pool.active_key_ids_for_expert(0)
    ])
    best = (torch.nn.functional.normalize(queries, dim=-1) @ keys.T).max(dim=1).values
    assert torch.allclose(per_expert[:, column], best, atol=1e-5)


def test_oracle_J_the_whole_inference_chain_is_supervision_free():
    """J: purity over the modules the deployable path actually runs."""
    for name in ("inference.py", "routing.py", "query.py", "generate.py"):
        report = assert_inference_purity(REPO / "compose" / "v8" / name)
        assert report["pure"] is True, (name, report)
        assert report["forbidden_identifiers_found"] == []
        assert report["forbidden_strings_found"] == []

    import inspect

    from compose.v8.inference import V8InferenceRouter

    pool = make_pool(num_experts=3, per_task=3, seed=5)
    router = V8InferenceRouter(pool)
    # The only tensors the router holds are the committed keys themselves --
    # frozen data, not a learnable surface (PART 7) -- and nothing is buffered.
    assert not list(router.buffers())
    assert {id(parameter) for parameter in router.parameters()} == {
        id(pool.keys[key_id]) for key_id in pool.key_ids()
    }
    assert not any(parameter.requires_grad for parameter in router.parameters())
    for method in ("route_policy", "selection", "forward"):
        signature = inspect.signature(getattr(router, method))
        assert not any(
            token in name.lower()
            for name in signature.parameters
            for token in ("answer", "ground_truth", "label", "teacher", "nll")
        ), method


def test_oracle_K_teacher_oracle_and_actual_top2_metrics_are_formally_distinct():
    """K: the harness must not be able to report one route's number as the other's."""
    from compose.experiments.v8_task_run import (
        TOP2_MEASURED_NOTE,
        TOP2_NOT_MEASURED_NOTE,
        _parser,
        metric_report,
    )

    # the deployable route is off unless it is explicitly asked for, so no run
    # can quietly publish an oracle number under the wrong name
    args = _parser().parse_args(["--task", "1", "--root", "/tmp/v8-test-root"])
    assert args.top2_inference_eval is False

    unmeasured = metric_report(94.14, None)
    assert unmeasured["teacher_oracle_metric"] == 94.14
    assert unmeasured["actual_top2_inference_metric"] is None
    assert unmeasured["actual_top2_inference_metric_note"] == TOP2_NOT_MEASURED_NOTE
    assert "ORACLE" in unmeasured["teacher_oracle_metric_note"]
    assert unmeasured["metrics_are_separate"] is True

    measured = metric_report(94.14, {"value": 87.11})
    assert measured["teacher_oracle_metric"] == 94.14
    assert measured["actual_top2_inference_metric"] == 87.11
    assert measured["actual_top2_inference_metric_note"] == TOP2_MEASURED_NOTE
    assert measured["teacher_oracle_metric"] != measured["actual_top2_inference_metric"]


def test_oracle_L_mixed_batch_leakage_is_still_probed():
    """L: TEST30's guarantee is unchanged by the per-batch key-loss change."""
    assert hasattr(
        sys.modules[__name__],
        "test_30_gradient_leakage_on_mixed_baseonly_reuse_residual_batch",
    )


def test_oracle_M_teacher_result_artifact_is_buildable_without_a_run():
    """M: the ``teacher_result.json`` body is constructible off-line.

    Regression for a defect the first oracle smoke found and no unit test could:
    ``run_teacher`` assembled this dict inline and called ``teacher_coverage_report``,
    a function that does not exist -- the real API is ``TeacherResult.coverage_report()``.
    It is a ``NameError``, so the module imported fine, every unit test passed, and
    the failure appeared only after ~13 minutes of GPU generation, at the moment the
    teacher tried to write its artefact.  ``teacher_result_payload`` is that dict as a
    pure function, so the shape is assertable without a model: this test fails with
    ``NameError`` on the pre-fix code and passes on the fixed one.
    """
    from compose.experiments.v8_task_run import (
        TEACHER_RESULT_SCHEMA_VERSION,
        teacher_result_payload,
    )
    from compose.v8.teacher import TeacherResult, TeacherSampleRecord

    record = TeacherSampleRecord(
        sample_id="s0", state=STATE_REUSE1, selected_experts=[1],
        base_solved=False, base_value=0.0, recall=[1, 2], single_values={"1": 1.0},
        pair_values={}, tested_singles=[1, 2], tested_pairs=[],
        achieved_value=1.0, achieved_nll=None, delta_nll=None, teacher_gain=1.0,
        key_targets={"1": TARGET_POSITIVE, "2": TARGET_CONTEXT_POSITIVE},
        decision_reason="single_solved_stop_before_pairs",
        residual_context=[2],
        historical_experts_visible=[1, 2],
        historical_experts_tested=[1, 2],
    )
    # ``search_mode`` and ``visible_experts`` are *views* on the result, read off
    # ``config`` and the per-record visible sets -- not constructor fields.  The
    # config is built exactly as ``AnswerSupervisedTeacher.run`` builds it.
    teacher_config = V8TeacherConfig()
    result = TeacherResult(
        task_id=4, records=[record], config=asdict(teacher_config),
    )
    payload = teacher_result_payload(result, teacher_config)

    # the artefact must carry the evidence that lets a reader audit the search
    # without re-running it, and the schema that says which semantics produced it
    assert payload["schema_version"] == TEACHER_RESULT_SCHEMA_VERSION
    assert payload["teacher_search_mode"] == TEACHER_SEARCH_FULL_HISTORY
    assert payload["historical_experts_visible"] == [1, 2]
    coverage = payload["coverage"]
    assert coverage["teacher_search_mode"] == TEACHER_SEARCH_FULL_HISTORY
    assert coverage["historical_experts_visible"] == [1, 2]
    # both visible experts were tested on the unsolved sample -> full coverage
    assert coverage["full_coverage"] is True
    assert coverage["historical_experts_tested"] == [1, 2]
    assert coverage["never_tested"] == []
    assert coverage["searched_samples"] == 1
    assert payload["config"]["search_mode"] == TEACHER_SEARCH_FULL_HISTORY
    assert payload["records"][0]["sample_id"] == "s0"
    assert payload["states"][STATE_REUSE1] == 1
    # and the payload carries no answer string, by the cache's own rule
    from compose.v8.cache import _forbid_answer_fields
    _forbid_answer_fields(payload)


def test_oracle_N_incomplete_coverage_is_visible_in_the_artifact():
    """N: a search that missed a visible expert must be legible in the payload."""
    from compose.experiments.v8_task_run import teacher_result_payload
    from compose.v8.teacher import TeacherResult, TeacherSampleRecord

    record = TeacherSampleRecord(
        sample_id="s0", state=STATE_RESIDUAL, selected_experts=[1],
        base_solved=False, base_value=0.0, recall=[1], single_values={"1": 1.0},
        pair_values={}, tested_singles=[1], tested_pairs=[],
        achieved_value=1.0, achieved_nll=None, delta_nll=None, teacher_gain=0.0,
        key_targets={"1": TARGET_CONTEXT_POSITIVE},
        decision_reason="no_single_or_pair_solved",
        residual_context=[1],
        historical_experts_visible=[1, 2, 3],
        historical_experts_tested=[1],          # 2 and 3 were never scored
    )
    result = TeacherResult(
        task_id=4, records=[record], config=asdict(V8TeacherConfig()),
    )
    coverage = teacher_result_payload(result, V8TeacherConfig())["coverage"]
    assert coverage["full_coverage"] is False
    assert coverage["historical_experts_visible"] == [1, 2, 3]
    assert coverage["historical_experts_tested"] == [1]
    # the artefact must *name* the experts the search skipped, not just report
    # that some set size differed -- this is the list a reader audits
    assert coverage["never_tested"] == [2, 3]
    assert coverage["searched_samples"] == 1
    assert coverage["fully_covered_samples"] == 0


def test_oracle_O_a_subset_run_may_not_borrow_the_full_split_v7_baseline(tmp_path):
    """O: a ``--limit`` run must not publish V7's full-split number as its baseline.

    Found by the first oracle smoke.  ``analyse`` read V7's metric from the
    *diagnostic root's* ``COMPLETE.json``, which is measured on the diagnostic's
    own 256-sample split -- so a 32-sample smoke reported ``v7=67.97``, a number
    it had not measured on its own samples, and fed it to ``gap_closed`` against
    a 32-sample V8 number.  The measurement is fine; treating it as this run's
    baseline is not.  Comparability is now checked against the diagnostic's own
    prediction file, and a mismatch routes the value to a reference field.
    """
    from compose.experiments.v8_task_run import _prediction_ids

    # V7 generation files key samples as ``question_id``; the teacher uses
    # ``sample_id``.  Both must be readable, or the check cannot be made at all.
    by_question = tmp_path / "v7_q.jsonl"
    by_question.write_text("\n".join(json.dumps({"question_id": "v7_t4_val_{}".format(i),
                                                 "text": "1"}) for i in range(4)) + "\n",
                           encoding="utf-8")
    assert _prediction_ids(by_question) == {"v7_t4_val_0", "v7_t4_val_1",
                                            "v7_t4_val_2", "v7_t4_val_3"}
    by_sample = tmp_path / "v7_s.jsonl"
    by_sample.write_text(json.dumps({"sample_id": "a", "text": "1"}) + "\n", encoding="utf-8")
    assert _prediction_ids(by_sample) == {"a"}

    # an id-less file must not be able to "match" anything -- the previous draft
    # collected ``str(row.get("sample_id"))``, i.e. the literal string "None",
    # which is a match waiting to happen
    id_less = tmp_path / "v7_none.jsonl"
    id_less.write_text(json.dumps({"text": "1"}) + "\n", encoding="utf-8")
    assert _prediction_ids(id_less) is None
    assert _prediction_ids(tmp_path / "missing.jsonl") is None
    assert _prediction_ids(tmp_path) is None

    # the prediction-file id sets of a limit-32 run and of the full split differ,
    # which is exactly the condition that must refuse the baseline
    subset = {"v7_t4_val_{}".format(i) for i in range(32)}
    full = {"v7_t4_val_{}".format(i) for i in range(256)}
    assert subset != full

    # and with no comparable baseline, diagnose must not fall into the A/B/D
    # cases -- all three are comparisons *against V7*
    diagnosis = v8_audit.diagnose(full_pool_oracle_solves=20, teacher_positives=20,
                                  candidate_recall=0.75, v7_metric=None,
                                  v8_metric=37.5, samples=32)
    assert diagnosis["case"] == "CASE_UNCLASSIFIED_NO_V7_BASELINE"
    assert diagnosis["v7_metric"] is None
    assert diagnosis["v7_metric_comparable"] is False
    # capability 0 is still decidable without a baseline: that case is about the
    # pool, not about the comparison
    assert v8_audit.diagnose(full_pool_oracle_solves=0, teacher_positives=0,
                             candidate_recall=0.0, v7_metric=None, v8_metric=12.5,
                             samples=32)["case"] == "CASE_C"
    # and a run that *does* have a baseline still classifies as before
    assert v8_audit.diagnose(full_pool_oracle_solves=20, teacher_positives=20,
                             candidate_recall=0.75, v7_metric=67.97,
                             v8_metric=37.5, samples=256)["case"] == "CASE_B"


def test_oracle_P_run_config_declares_the_real_scope(tmp_path):
    """P: the provenance file must not claim a narrower exclusion than the run used.

    ``write_config`` runs before ``recall``, and read the scope with
    ``getattr(self, "excluded_expert_ids", [])`` -- so every history-only run
    wrote ``excluded_expert_ids: []`` into its own ``run_config.json`` while the
    run actually excluded seven experts.  That is the §16.1 trap (read the scope
    from a field that does not hold it) reappearing in a second artefact, and it
    is worse than the first: ``run_config.json`` is the file a reader opens to
    find out what a run *was*.  The scope is now computed from
    ``_excluded_experts``, which is pure and available this early.
    """
    from compose.experiments.v8_task_run import V8TaskRun, _parser

    args = _parser().parse_args([
        "--task", "4", "--root", str(tmp_path), "--history-only",
    ])
    runner = V8TaskRun(args)
    # the two things ``_excluded_experts`` reads, as ``setup`` would leave them:
    # 23 live experts, task 4's own are 16-19 and later tasks own 20/21/23
    runner.expert_ids = list(range(22)) + [23]
    runner.pool = type("P", (), {"expert_records": {
        expert_id: {"origin_task": (0 if expert_id < 4 else
                                    1 if expert_id < 8 else
                                    2 if expert_id < 12 else
                                    3 if expert_id < 16 else
                                    4 if expert_id < 20 else 5)}
        for expert_id in runner.expert_ids
    }})()
    runner.checkpoint_dir = tmp_path / "ckpt"
    runner.question_file = tmp_path / "val_full.json"
    runner.query_path = tmp_path / "queries.pt"
    runner.query_contract_hash = "0" * 64
    runner.write_config()

    written = json.loads((runner.out / "run_config.json").read_text(encoding="utf-8"))
    assert written["history_only"] is True
    assert written["excluded_expert_ids"] == [16, 17, 18, 19, 20, 21, 23]
    # and it must agree with what the scope resolution actually produces
    assert written["excluded_expert_ids"] == runner._excluded_experts()

    # the all-experts scope genuinely excludes nothing, and still says so
    all_args = _parser().parse_args(["--task", "4", "--root", str(tmp_path / "all")])
    all_runner = V8TaskRun(all_args)
    all_runner.expert_ids = runner.expert_ids
    all_runner.pool = runner.pool
    all_runner.checkpoint_dir = tmp_path / "ckpt"
    all_runner.question_file = tmp_path / "val_full.json"
    all_runner.query_path = tmp_path / "queries.pt"
    all_runner.query_contract_hash = "0" * 64
    all_runner.write_config()
    assert json.loads((all_runner.out / "run_config.json").read_text(
        encoding="utf-8"))["excluded_expert_ids"] == []
