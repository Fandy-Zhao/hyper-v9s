"""Static acceptance checks for V9-S (spec §38, §40, §42).

Everything here runs on CPU in seconds and needs no LLaVA checkpoint.  The suite
exists because the method's central claims are checkable without a GPU:

* **the answer trains the Candidate LoRA and nothing else directly** -- the gate
  that drives the composition carries no key gradient, so ``L_ans`` reaches no
  key parameter, and the only answer-side path into a key is
  ``contribution -> responsibility -> L_key``;
* **one forward per sample** -- one backbone pass carries every offered expert,
  and the quantity that supervises the keys is the derivative of the
  ground-truth answer loss along the gate, not an enumeration over experts;
* **the deployment path is unchanged** -- routing at test time is cosine over
  the pool's own effective keys, with no answer, no task id and no training
  recall cache reachable from the module.

Run with ``python -m pytest tests/compose/test_v9_static.py``.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F

from compose.adapters.types import ComposeSelection
from compose.v8.config import V8RoutingConfig
from compose.v8.pool import (
    KEY_TYPE_TASK_ALIAS,
    LIFECYCLE_CANDIDATE,
    LIFECYCLE_HISTORICAL,
    LIFECYCLE_PRUNED,
)
from compose.v9.audit import (
    apply_historical_task_key_audit,
    audit_historical_task_keys,
)
from compose.v9.config import (
    V9Config,
    V9InferenceConfig,
    V9ResponsibilityConfig,
    assert_frozen_contract,
    load_v9_config,
)
from compose.v9.contribution import (
    answer_derived_responsibility,
    calibration_report,
    gate_gradient,
    local_conditional_contribution,
    pair_rerank_report,
)
from compose.v9.inference import (
    V9InferenceError,
    V9InferenceRouter,
    assert_v9_inference_purity,
    validate_inference_policy,
)
from compose.v9.keys import V9KeyPool, initialize_candidate_keys
from compose.v9.losses import budget_loss, compose_total_loss, sparse_loss
from compose.v9.multi_key import aggregatable_expert_ids, memory_key_ids
from compose.v9.retrieval import (
    HistoricalTopC,
    V9RetrievalError,
    build_historical_topc,
    is_wide_step,
    load_or_build_historical_topc,
    retrieval_diagnostics,
)
from compose.v9.router import V9Router
from compose.v9.schedule import (
    STAGE_BOOTSTRAP,
    STAGE_HARD,
    STAGE_SOFT,
    V9StageScheduler,
)
from compose.v9.trainer import answer_loss_key_gradient

QUERY_DIM = 1536

#: The task every fixture routes on.  Above every fixture expert's origin task,
#: which is what ``add_task_key`` requires: a task key on or before an expert's
#: origin would shadow the identity it was committed with.
ROUTING_TASK = 1


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
def _query_matrix(count: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    return F.normalize(torch.randn(count, QUERY_DIM, generator=generator), dim=-1)


#: Keeps the pool's random draws off the query matrix's.  Two generators on the
#: same seed produce the same vectors, which would hand expert 0 a cosine of
#: exactly 1 with query 0 and quietly make every fixture degenerate.
_POOL_SEED_OFFSET = 7919


def _pool(config: V9Config, historical: int, candidates: int, seed: int = 0):
    """A pool with ``historical`` frozen experts and ``candidates`` fresh ones."""
    generator = torch.Generator().manual_seed(int(seed) + _POOL_SEED_OFFSET)
    pool = V9KeyPool(query_dim=config.query.query_dim)
    next_id = 0
    for _ in range(historical):
        pool.add_expert_with_base(
            next_id,
            F.normalize(torch.randn(QUERY_DIM, generator=generator), dim=-1),
            origin_task=0,
            lifecycle=LIFECYCLE_HISTORICAL,
        )
        next_id += 1
    candidate_ids = []
    for _ in range(candidates):
        pool.add_expert_with_base(
            next_id,
            F.normalize(torch.randn(QUERY_DIM, generator=generator), dim=-1),
            origin_task=1,
            lifecycle=LIFECYCLE_CANDIDATE,
            trainable=True,
        )
        candidate_ids.append(next_id)
        next_id += 1
    pool.validate()
    return pool, candidate_ids


def _with_task_keys(pool, task_id: int = ROUTING_TASK):
    """Give every historical expert its own current-task key, as a task does."""
    for expert_id in pool.historical_ids:
        pool.add_task_key(expert_id, task_id=task_id)
    return pool


def _recall(pool, queries, config: V9Config, sample_ids=None) -> HistoricalTopC:
    return build_historical_topc(
        queries,
        pool.historical_ids,
        pool.base_key_matrix(pool.historical_ids, detach=True),
        config.historical_retrieval,
        wide=config.wide_retrieval,
        task_index=ROUTING_TASK,
        seed=1,
        sample_ids=sample_ids,
    )


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
def test_config_round_trip_is_exact():
    config = V9Config()
    assert V9Config.from_dict(config.to_dict()) == config
    # The cached recall is as wide as the widest step ever used, and no wider:
    # the base Top-C is a prefix of the wide Top-C, so one cache serves both.
    assert config.historical_slots == max(
        config.historical_retrieval.top_c, config.wide_retrieval.top_c
    )
    assert config.selection_slots == config.historical_slots + config.candidate_count
    assert config.active_historical_slots(False) == config.historical_retrieval.top_c
    assert config.active_historical_slots(True) == config.wide_retrieval.top_c


def test_the_declared_defaults_are_the_V9S_recipe():
    """Spec §35/§37: the shipped defaults *are* the main method."""
    config = V9Config()
    assert config.candidate_count == 2, "M = 2 is the main method, M = 4 the ablation"
    assert config.routing.type == "independent_sigmoid"
    assert config.routing.direct_answer_gradient_to_key is False
    assert config.routing.max_inference_experts == 2
    assert config.wide_retrieval.enabled and 0.0 < config.wide_retrieval.ratio < 1.0
    assert config.responsibility == V9ResponsibilityConfig()
    assert config.loss.use_budget_loss is False, "budget loss is the ablation"
    assert config.inference == V9InferenceConfig()
    assert config.inference.task_id is False
    assert config.inference.global_multi_key is True
    assert config.inference.pair_rerank is False
    assert config.exact_oracle.training is False
    assert config.exact_oracle.validation_calibration_only is True
    # The removed V9 v1 ingredients must not be configurable at all: a knob for
    # a mechanism the method does not have is a way to believe it is running.
    assert not hasattr(config, "gamma")
    assert not hasattr(config.key, "gamma")
    assert not hasattr(config, "pair_rerank")
    assert not hasattr(config.bootstrap, "exploration_gate")
    assert not hasattr(config.routing, "final_max_experts")


def _load_and_assert(raw: dict, message: str) -> None:
    """The recipe drift must die somewhere on the load path.

    Some constraints are structural enough to catch while parsing the section
    they belong to; the rest are re-asserted against the assembled config.  A
    YAML that can be *loaded* while violating either layer is the failure this
    guards against, so both are exercised in one path.
    """
    with pytest.raises(ValueError) as error:
        assert_frozen_contract(V9Config.from_dict(raw))
    assert message in str(error.value)


@pytest.mark.parametrize(
    "mutate, message",
    [
        # Caught while parsing the section it belongs to.
        (lambda value: {"routing": dict(value.routing.__dict__, type="softmax")}, "sigmoid"),
        (
            lambda value: {
                "schedule": dict(
                    value.schedule.__dict__,
                    bootstrap_ratio=0.08,
                    soft_ratio=0.92,
                    hard_ratio=0.0,
                )
            },
            "discretisation",
        ),
        (
            lambda value: {"key": dict(value.key.__dict__, aggregation="mean")},
            "aggregation",
        ),
        (lambda value: {"query": dict(value.query.__dict__, query_dim=1024)}, "1536"),
        (
            lambda value: {
                "historical_retrieval": dict(
                    value.historical_retrieval.__dict__, aggregation="mean"
                )
            },
            "frozen base keys",
        ),
        # The two supervision routes into a key must not both be open.
        (
            lambda value: {
                "routing": dict(value.routing.__dict__, direct_answer_gradient_to_key=True)
            },
            "only through contribution",
        ),
        (
            lambda value: {
                "responsibility": dict(value.responsibility.__dict__, positive_only=False)
            },
            "positive-only",
        ),
        (
            lambda value: {"inference": dict(value.inference.__dict__, task_id=True)},
            "task id",
        ),
        (
            lambda value: {
                "inference": dict(value.inference.__dict__, global_multi_key=False)
            },
            "Global Multi-Key",
        ),
        (
            lambda value: {
                "exact_oracle": dict(value.exact_oracle.__dict__, training=True)
            },
            "training loop",
        ),
        (
            lambda value: {
                "wide_retrieval": dict(value.wide_retrieval.__dict__, ratio=0.0)
            },
            "ratio 0",
        ),
        # Structurally legal, so only the contract re-assertion can catch it.
        (
            lambda value: {
                "composition": dict(value.composition.__dict__, cardinality_scale="v8")
            },
            "cardinality",
        ),
        (lambda value: {"query": dict(value.query.__dict__, cache=False)}, "cached"),
        (
            lambda value: {
                "wide_retrieval": dict(value.wide_retrieval.__dict__, top_c=2)
            },
            "narrower",
        ),
    ],
)
def test_frozen_contract_rejects_recipe_drift(mutate, message):
    raw = V9Config().to_dict()
    raw.update(mutate(V9Config()))
    _load_and_assert(raw, message)


# ----------------------------------------------------------------------
# schedule
# ----------------------------------------------------------------------
def test_schedule_covers_three_stages_with_a_monotone_temperature():
    config = V9Config()
    scheduler = V9StageScheduler(config.schedule, config.routing, 1000)
    stages = [scheduler.state(step).stage for step in range(1000)]
    assert stages[0] == STAGE_BOOTSTRAP
    assert STAGE_SOFT in stages and STAGE_HARD in stages
    # Order is preserved: no stage may reappear after it has ended.
    order = [STAGE_BOOTSTRAP, STAGE_SOFT, STAGE_HARD]
    assert stages == sorted(stages, key=order.index)
    temperatures = [scheduler.state(step).temperature for step in range(1000)]
    assert temperatures == sorted(temperatures, reverse=True)
    assert temperatures[0] == config.routing.temperature_start
    # The ramp is defined on the step axis, so it reaches its endpoint at the
    # first step *past* the run and must clamp rather than overshoot beyond it.
    assert scheduler.state(1000).temperature == config.routing.temperature_end
    assert scheduler.state(5000).temperature == config.routing.temperature_end
    assert temperatures[-1] > config.routing.temperature_end
    # The discretisation stage is the last one and must be non-empty: under the
    # detach contract it *is* the deployed rule, not a surrogate for it.
    assert stages[-1] == STAGE_HARD
    assert config.schedule.hard_ratio > 0


# ----------------------------------------------------------------------
# keys
# ----------------------------------------------------------------------
def test_a_new_task_key_starts_at_the_experts_base_key():
    config = V9Config()
    pool, _ = _pool(config, historical=3, candidates=0)
    for expert_id in pool.historical_ids:
        base = pool.effective_key_matrix([pool.base_key_id(expert_id)], detach=True)
        pool.add_task_key(expert_id, task_id=ROUTING_TASK)
        routed = pool.effective_key_matrix(
            [pool.routing_key_id(expert_id, ROUTING_TASK)], detach=True
        )
        assert torch.equal(base, routed), (
            "an expert enters a new task routing exactly as itself: a fresh "
            "task key must be bit-identical to the expert's base key"
        )
        assert pool.routing_key_id(expert_id, ROUTING_TASK) != pool.base_key_id(
            expert_id
        )


def test_a_task_key_is_absolute_not_a_delta_on_the_base():
    """V9-S deleted ``normalize(base + gamma * delta)``; the key stands alone."""
    config = V9Config()
    pool, _ = _pool(config, historical=1, candidates=0)
    expert_id = pool.historical_ids[0]
    key_id = pool.add_task_key(expert_id, task_id=ROUTING_TASK)
    base = pool.keys[pool.base_key_id(expert_id)].detach().clone()
    replacement = F.normalize(torch.full((QUERY_DIM,), 0.05), dim=-1)
    with torch.no_grad():
        pool.keys[key_id].copy_(replacement)
    effective = pool.effective_key_matrix([key_id], detach=True)[0]
    # Absolute: the stored value is the direction, up to the normalisation every
    # routing comparison applies.
    assert torch.allclose(effective, replacement, atol=1e-6)
    # And not a residual: the base plays no part in where the expert now points.
    assert not torch.allclose(
        effective, F.normalize(base + replacement, dim=-1), atol=1e-6
    )
    # Moving the base moves nothing, which is the property that used to hold and
    # no longer does.
    with torch.no_grad():
        pool.keys[pool.base_key_id(expert_id)].add_(torch.randn(QUERY_DIM))
    assert torch.allclose(
        pool.effective_key_matrix([key_id], detach=True)[0], effective, atol=1e-6
    )


def test_only_current_task_and_candidate_keys_are_trainable():
    config = V9Config()
    pool, candidate_ids = _pool(config, historical=3, candidates=2)
    _with_task_keys(pool)
    pool.freeze_historical(current_task=ROUTING_TASK)
    trainable = set(pool.trainable_key_ids())
    assert trainable == {
        pool.routing_key_id(expert_id, ROUTING_TASK)
        for expert_id in pool.historical_ids
    } | {pool.origin_key_id(expert_id) for expert_id in candidate_ids}
    # The freeze audit must not claim the candidates' own keys: they are the
    # keys this task is training, and the end-of-task checksum compares against
    # the moment training started.
    frozen = set(pool.historical_key_ids())
    assert not frozen & {pool.origin_key_id(expert_id) for expert_id in candidate_ids}
    assert all(not pool.keys[key_id].requires_grad for key_id in frozen)
    # Every trainable key is a key the router will actually route through.
    routed = set(pool.routing_key_ids(pool.expert_ids(), task_id=ROUTING_TASK))
    assert trainable <= routed
    # Every task key is an absolute V8 alias: the one key role that carries a
    # later task's direction, and the only key type V9-S adds.
    task_key_ids = [
        pool.routing_key_id(expert_id, ROUTING_TASK)
        for expert_id in pool.historical_ids
    ]
    assert all(
        pool.key_record(key_id)["key_type"] == KEY_TYPE_TASK_ALIAS
        for key_id in task_key_ids
    )


def test_historical_key_ids_cover_every_retained_key():
    """The freeze audit must see the keys a later task is most likely to touch."""
    config = V9Config()
    pool, _ = _pool(config, historical=2, candidates=0)
    key_id = pool.add_task_key(pool.historical_ids[0], task_id=ROUTING_TASK)
    pool.set_key_lifecycle(key_id, LIFECYCLE_HISTORICAL)
    assert key_id in pool.historical_key_ids()
    assert key_id in memory_key_ids(pool, pool.historical_ids[0])


def test_candidate_init_is_deterministic_and_units():
    queries = _query_matrix(64, seed=3)
    first = initialize_candidate_keys(queries, 4, seed=7)
    second = initialize_candidate_keys(queries, 4, seed=7)
    third = initialize_candidate_keys(queries, 4, seed=8)
    assert first.shape == (4, QUERY_DIM)
    assert torch.allclose(first, second)
    assert not torch.allclose(first, third)
    assert torch.allclose(first.norm(dim=1), torch.ones(4), atol=1e-5)
    # Well separated, which is the entire reason for clustering here: a shared
    # task mean plus a small perturbation leaves the candidates collinear.
    cosine = first @ first.T
    off_diagonal = cosine - torch.eye(4) * cosine.diagonal()
    assert float(off_diagonal.max()) < 0.9


# ----------------------------------------------------------------------
# retrieval
# ----------------------------------------------------------------------
def test_historical_topc_rows_are_disjoint_and_reach_the_frozen_base_keys():
    config = V9Config()
    pool, _ = _pool(config, historical=10, candidates=0)
    queries = _query_matrix(32, seed=5)
    topc = _recall(pool, queries, config)
    assert topc.slots == config.historical_slots
    assert topc.top_c == config.historical_retrieval.top_c
    assert topc.wide_top_c == config.wide_retrieval.top_c
    assert topc.rows == 32
    for row in topc.expert_ids:
        assert len(set(int(value) for value in row.tolist())) == row.numel()
    # The ordering is by frozen-base-key cosine, so the top-1 must genuinely be
    # the highest-scoring historical expert for that query.
    similarity = queries @ pool.base_key_matrix(pool.historical_ids, detach=True).T
    expected = pool.historical_ids[int(similarity[0].argmax())]
    assert int(topc.expert_ids[0, 0]) == expected


def test_the_wide_recall_extends_the_base_recall_rather_than_replacing_it():
    """One cache serves both widths -- the invariant the whole design rests on."""
    config = V9Config()
    pool, _ = _pool(config, historical=10, candidates=0)
    queries = _query_matrix(24, seed=17)
    topc = _recall(pool, queries, config)
    base = topc.expert_ids[:, : topc.active_columns(False)]
    wide = topc.expert_ids[:, : topc.active_columns(True)]
    assert torch.equal(wide[:, : base.shape[1]], base)
    diagnostics = retrieval_diagnostics(topc)
    assert diagnostics["distinct_in_wide_recall"] >= diagnostics["distinct_in_topc"]
    assert diagnostics["wide_top_c"] == config.wide_retrieval.top_c


def test_wide_steps_are_deterministic_and_hit_the_declared_ratio():
    """Every rank must agree on the row width without a collective."""
    ratio = 0.15
    first = [is_wide_step(step, ratio, seed=42) for step in range(4000)]
    second = [is_wide_step(step, ratio, seed=42) for step in range(4000)]
    assert first == second, "the draw is a pure function of (seed, step)"
    assert first != [is_wide_step(step, ratio, seed=43) for step in range(4000)]
    observed = sum(first) / len(first)
    assert abs(observed - ratio) < 0.02, observed
    # The degenerate ends are total functions, not special cases at the call
    # site: ratio 0 must not widen, ratio 1 must widen every step.
    assert not any(is_wide_step(step, 0.0, seed=42) for step in range(64))
    assert all(is_wide_step(step, 1.0, seed=42) for step in range(64))


def test_historical_topc_round_trip_realigns_by_sample_id(tmp_path):
    config = V9Config()
    pool, _ = _pool(config, historical=4, candidates=0)
    queries = _query_matrix(6, seed=11)
    ids = ["s{}".format(index) for index in range(6)]
    topc = _recall(pool, queries, config, sample_ids=ids)
    path = tmp_path / "topc.pt"
    topc.save(str(path))
    loaded = HistoricalTopC.load(str(path))
    assert loaded.sample_ids == ids
    shuffled = ids[::-1]
    aligned = loaded.aligned_rows(shuffled)
    assert torch.equal(aligned[0], loaded.expert_ids[-1])
    with pytest.raises(V9RetrievalError):
        loaded.aligned_rows(ids[:-1] + ["missing"])


def test_historical_topc_cache_rebuilds_when_the_recipe_changes(tmp_path):
    config = V9Config()
    pool, _ = _pool(config, historical=6, candidates=0)
    queries = _query_matrix(8, seed=13)
    keys = pool.base_key_matrix(pool.historical_ids, detach=True)
    path = str(tmp_path / "topc.pt")
    first = load_or_build_historical_topc(
        path,
        queries,
        pool.historical_ids,
        keys,
        config.historical_retrieval,
        wide=config.wide_retrieval,
        task_index=ROUTING_TASK,
        seed=1,
    )
    widened = dataclasses.replace(config.historical_retrieval, top_c=3)
    second = load_or_build_historical_topc(
        path,
        queries,
        pool.historical_ids,
        keys,
        widened,
        wide=config.wide_retrieval,
        task_index=ROUTING_TASK,
        seed=1,
    )
    assert first.top_c == config.historical_retrieval.top_c
    assert second.top_c == 3


# ----------------------------------------------------------------------
# routing
# ----------------------------------------------------------------------
def _router(config: V9Config, pool):
    return V9Router(
        config=config,
        key_pool=pool,
        candidate_ids=pool.current_ids,
        historical_ids=pool.historical_ids,
        task_index=ROUTING_TASK,
    )


def _route(
    config: V9Config,
    pool,
    rows: int,
    stage: str,
    seed: int = 0,
    bias: float | None = None,
    wide: bool = False,
):
    router = _router(config, pool)
    if bias is not None:
        with torch.no_grad():
            router.bias.fill_(float(bias))
    queries = _query_matrix(rows, seed=seed)
    topc = _recall(pool, queries, config)
    route = router.route(queries, topc.expert_ids, 1.0, stage, wide=wide)
    return router, route, queries, topc


def test_routing_row_is_historical_block_then_candidates():
    config = V9Config()
    pool, candidate_ids = _pool(config, historical=5, candidates=config.candidate_count)
    _with_task_keys(pool)
    router, route, _, topc = _route(config, pool, 4, STAGE_SOFT)
    # The row is as wide as the recall the pool could actually supply, and the
    # cache is built at the pool's own bound rather than at the configured one.
    assert topc.slots == min(config.historical_slots, len(pool.historical_ids))
    assert route.expert_ids.shape == (4, topc.slots + config.candidate_count)
    # Every historical column holds a real expert -- the recall filled the whole
    # row -- but a base step only *reads* the first ``top_c`` of them, which is
    # how one cached row serves both widths.
    live = config.active_historical_slots(False)
    assert live == config.historical_retrieval.top_c < topc.slots
    expected_mask = torch.cat(
        [
            torch.arange(topc.slots).unsqueeze(0).lt(live).expand(4, -1),
            torch.ones(4, config.candidate_count, dtype=torch.bool),
        ],
        dim=1,
    )
    assert route.slot_mask.dtype == torch.bool
    assert torch.equal(route.slot_mask, expected_mask)
    assert torch.equal(
        route.probabilities[~expected_mask],
        torch.zeros(int((~expected_mask).sum())),
    )
    tail = route.expert_ids[:, topc.slots:]
    expected = torch.tensor(candidate_ids).unsqueeze(0).expand(4, -1)
    assert torch.equal(tail, expected)
    assert not route.expert_ids.requires_grad
    assert router.routing_key_ids() == router.routing_key_ids()


def test_a_base_step_masks_the_wide_tail_and_a_wide_step_opens_it():
    """Wide recall is a masked slice of the same cached row, not a new row."""
    config = V9Config()
    pool, candidate_ids = _pool(config, historical=10, candidates=config.candidate_count)
    _with_task_keys(pool)
    router = _router(config, pool)
    queries = _query_matrix(6, seed=23)
    topc = _recall(pool, queries, config)

    base = router.route(queries, topc.expert_ids, 1.0, STAGE_SOFT, wide=False)
    wide = router.route(queries, topc.expert_ids, 1.0, STAGE_SOFT, wide=True)
    base_cut = config.active_historical_slots(False)
    wide_cut = config.active_historical_slots(True)

    assert base.slot_mask.sum() == 6 * (base_cut + config.candidate_count)
    assert wide.slot_mask.sum() == 6 * (wide_cut + config.candidate_count)
    # A masked slot costs nothing: it is a zero gate the composition skips.
    assert float(base.forward_gates[~base.slot_mask].abs().sum()) == 0.0
    # The tail is still there in the tensor, which is what keeps the row width
    # fixed for the whole task.
    assert base.expert_ids.shape == wide.expert_ids.shape
    tails = base.expert_ids[:, base_cut : config.historical_slots]
    assert bool((tails != -1).any()), "the masked tail holds the recall it computed"
    assert base.wide is False and wide.wide is True


def test_routing_is_independent_sigmoid_not_softmax():
    config = V9Config()
    pool, candidate_ids = _pool(config, historical=5, candidates=config.candidate_count)
    _with_task_keys(pool)
    router, route, queries, _ = _route(config, pool, 8, STAGE_SOFT)
    key_ids = router.routing_key_ids()
    keys = pool.effective_key_matrix(key_ids, detach=False)
    cosine = F.normalize(queries, dim=-1) @ keys.T
    columns = router._expert_columns(route.expert_ids)
    expected = torch.sigmoid(
        cosine.gather(1, columns)
        - router.bias.detach().gather(0, columns.reshape(-1)).reshape(
            route.expert_ids.shape
        )
    )
    # A slot the recall did not fill carries no gate at all, which is what lets
    # the wide tail be masked out of the composition.
    expected = torch.where(route.slot_mask, expected, torch.zeros_like(expected))
    assert torch.allclose(route.probabilities, expected, atol=1e-6)
    # A softmax row sums to 1 by construction.  An independent sigmoid does not,
    # and that is the point: a sample may need none, one, or two experts.
    totals = route.probabilities.sum(dim=1)
    assert not torch.allclose(totals, torch.ones_like(totals), atol=1e-3)


def test_bootstrap_floors_every_gate_but_keeps_the_contribution_measurable():
    config = V9Config()
    pool, candidate_ids = _pool(config, historical=4, candidates=config.candidate_count)
    _with_task_keys(pool)
    _, route, _, _ = _route(config, pool, 6, STAGE_BOOTSTRAP)
    floor = config.bootstrap.gate_floor
    assert float(route.forward_gates.min()) >= floor - 1e-6
    assert route.hard_gates is None
    # The floor is a forward-only device: the trainable quantity is still the
    # sigmoid, and it is still differentiable w.r.t. the keys.
    assert torch.equal(route.probabilities, route.soft_gates)
    assert route.soft_gates.requires_grad
    assert (route.forward_gates >= route.probabilities.detach() - 1e-6).all()


def test_hard_stage_forward_is_exactly_the_deployed_top2():
    config = V9Config()
    pool, candidate_ids = _pool(config, historical=6, candidates=config.candidate_count)
    _with_task_keys(pool)
    router, route, _, _ = _route(config, pool, 16, STAGE_HARD)
    k = config.routing.max_inference_experts
    assert k == 2, "deployment is Top-2"
    assert set(route.forward_gates.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(route.forward_gates.sum(dim=1), torch.full((16,), float(k)))
    # Under the detach contract the straight-through form collapses to the
    # indicator itself: `hard - p.detach() + p` with `p` already detached is
    # `hard`.  The final stage is therefore literally the inference rule.
    assert not route.forward_gates.requires_grad
    assert not route.forward_gates.grad_fn
    # Exactly the Top-2 by probability, per row.
    top = torch.topk(route.probabilities.detach(), k, dim=1).indices
    selected = route.forward_gates.bool().nonzero()[:, 1].reshape(16, k)
    assert torch.equal(selected.sort(dim=1).values, top.sort(dim=1).values)


def test_the_forward_gate_carries_no_key_gradient():
    """Spec §38: the answer reaches a key only through L_key, never the gate."""
    config = V9Config()
    pool, candidate_ids = _pool(config, historical=4, candidates=config.candidate_count)
    _with_task_keys(pool)
    pool.freeze_historical(current_task=ROUTING_TASK)
    router, route, _, _ = _route(config, pool, 8, STAGE_SOFT)
    keys = router.trainable_parameters()["key"]
    assert keys, "the fixture must expose at least one trainable key"

    # The forward gate is what the composition multiplies the expert outputs
    # by.  It is not merely `allow_unused` for the keys -- it carries no
    # autograd history at all, so there is no graph along which any downstream
    # answer loss could reach a key through it.
    assert not route.forward_gates.requires_grad
    assert route.forward_gates.grad_fn is None

    # ...while `probabilities` stays differentiable w.r.t. those same keys: that
    # is what makes the contribution measurable at all.
    gradient = torch.autograd.grad(
        route.probabilities.sum(),
        keys[0],
        retain_graph=True,
        allow_unused=True,
    )[0]
    assert gradient is not None and float(gradient.abs().sum()) > 0.0


def test_key_and_bias_gradients_reach_the_router_parameters():
    """L_key's route into the keys exists, is wired, and is not dead."""
    config = V9Config()
    pool, candidate_ids = _pool(config, historical=4, candidates=config.candidate_count)
    _with_task_keys(pool)
    pool.freeze_historical(current_task=ROUTING_TASK)
    router, route, _, _ = _route(config, pool, 8, STAGE_SOFT)
    loss = route.probabilities.sum()
    gradients = torch.autograd.grad(
        loss, router.trainable_parameters()["key"] + [router.bias], retain_graph=False
    )
    assert all(gradient is not None for gradient in gradients)
    keys = router.trainable_parameters()["key"]
    assert any(float(gradient.abs().sum()) > 0 for gradient in gradients[: len(keys)])
    assert float(gradients[-1].abs().sum()) > 0


def test_the_answer_loss_trains_the_candidate_lora_and_no_key():
    """Spec §41 (A) and (B), at the level of a real autograd graph.

    The composition is ``h = base + sum_k a_k * u_k(theta_k)``.  A candidate's
    LoRA output must carry the answer gradient; the gates that scale it must
    not, because their key-dependent part is precisely what would otherwise
    hand the answer a second, unsupervised route into the key vector.
    """
    config = V9Config()
    pool, candidate_ids = _pool(config, historical=4, candidates=config.candidate_count)
    _with_task_keys(pool)
    pool.freeze_historical(current_task=ROUTING_TASK)
    router, route, queries, _ = _route(config, pool, 8, STAGE_SOFT)

    generator = torch.Generator().manual_seed(5)
    expert_outputs = {
        expert_id: torch.randn(8, 16, generator=generator) * 0.1
        for expert_id in pool.expert_ids()
    }
    lora_parameters = {
        expert_id: torch.nn.Parameter(expert_outputs[expert_id].clone())
        for expert_id in candidate_ids
    }
    # One forward, every offered expert, scaled by the gate the composition saw.
    mixed = torch.zeros(8, 16)
    for slot in range(route.slots):
        for row in range(route.batch_size):
            expert_id = int(route.expert_ids[row, slot])
            if expert_id < 0 or not bool(route.slot_mask[row, slot]):
                continue
            gate = route.forward_gates[row, slot]
            output = lora_parameters.get(expert_id, expert_outputs[expert_id])
            mixed[row] = mixed[row] + gate * output[row]
    answer_loss = (mixed ** 2).mean()

    assert answer_loss.requires_grad
    candidate_gradients = torch.autograd.grad(
        answer_loss,
        list(lora_parameters.values()),
        retain_graph=True,
        allow_unused=True,
    )
    assert any(
        gradient is not None and float(gradient.abs().sum()) > 0
        for gradient in candidate_gradients
    ), "the answer must train the Candidate LoRA"

    key_gradients = answer_loss_key_gradient(
        answer_loss, _key_parameter_map(router)
    )
    assert key_gradients, "the audit must have looked at least one key"
    assert max(key_gradients.values()) == 0.0, (
        "L_ans reached a key directly: {}".format(
            {k: v for k, v in key_gradients.items() if v}
        )
    )


def _key_parameter_map(router: V9Router):
    return {
        key_id: router.key_pool.keys[key_id]
        for key_id in router.key_pool.trainable_key_ids()
    }


def test_answer_loss_key_gradient_reports_a_max_per_key():
    first = torch.nn.Parameter(torch.zeros(4))
    second = torch.nn.Parameter(torch.full((4,), 0.5))
    loss = second.sum()
    report = answer_loss_key_gradient(loss, {"a": first, "b": second})
    assert set(report) == {"a", "b"}
    assert report["a"] == 0.0
    assert report["b"] > 0.0


# ----------------------------------------------------------------------
# contribution and responsibility
# ----------------------------------------------------------------------
def test_local_contribution_is_the_gate_gradient_with_stop_gradient():
    gates = torch.tensor([[0.7, 0.2, 0.4]], requires_grad=True)
    weights = torch.tensor([[2.0, -1.0, 0.5]])
    loss = (gates * weights).sum()

    gradient = gate_gradient(loss, gates)
    assert torch.allclose(gradient, weights)
    contribution = local_conditional_contribution(loss, gates, retain_graph=False)
    assert torch.allclose(contribution, -(gates.detach() * weights))
    # create_graph=False: the teacher must not carry a second-order graph.
    assert not contribution.requires_grad


def test_local_contribution_uses_the_participation_the_forward_used():
    """Bootstrap floors the gate; the credit must follow the floored value."""
    config = V9Config()
    pool, candidate_ids = _pool(config, historical=4, candidates=config.candidate_count)
    _with_task_keys(pool)
    # Untouched random unit keys in 1536-D have near-zero cosine, so every gate
    # sits at ~0.5 and the floor never binds.  A large negative bias is what a
    # freshly initialised expert's gate looks like before the answer has ever
    # reached it, and it is the regime the floor exists for.
    router, route, _, _ = _route(
        config, pool, 4, STAGE_BOOTSTRAP, bias=9.0
    )
    floor = config.bootstrap.gate_floor
    assert float(route.probabilities.max()) < floor
    assert torch.allclose(
        route.forward_gates, torch.full_like(route.forward_gates, floor), atol=1e-6
    )
    loss = route.probabilities.sum()
    with_values = local_conditional_contribution(
        loss, route.soft_gates, retain_graph=True, values=route.forward_gates
    )
    without = local_conditional_contribution(loss, route.soft_gates, retain_graph=False)
    assert torch.allclose(without, -(route.probabilities.detach() * 1.0))
    assert torch.allclose(with_values, -(route.forward_gates.detach() * 1.0))
    assert not torch.allclose(with_values, without)


def test_responsibility_is_normalised_and_never_invented():
    contribution = torch.tensor([[2.0, 1.0, -3.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -3.0]])
    result = answer_derived_responsibility(contribution)
    assert torch.allclose(result.responsibility[0], torch.tensor([2 / 3, 1 / 3, 0.0]), atol=1e-6)
    assert result.valid.tolist() == [True, False, False]
    # A row the answer could not rank keeps a zero target and is excluded; it is
    # never handed a uniform or a zero "teacher" it did not earn.
    assert torch.equal(result.responsibility[1], torch.zeros(3))
    assert torch.equal(result.responsibility[2], torch.zeros(3))


def test_responsibility_supervision_is_bce_on_the_independent_gate():
    """The loss is BCE against the teacher, on the rows that have one."""
    from compose.v9.losses import key_responsibility_loss

    probabilities = torch.tensor([[0.9, 0.1], [0.5, 0.5]])
    responsibility = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    valid = torch.tensor([True, False])
    mask = torch.ones_like(probabilities, dtype=torch.bool)
    loss = key_responsibility_loss(probabilities, responsibility, valid, mask)
    # Row 1 is excluded, so the value is row 0's two terms and nothing else.
    expected = -(math.log(0.9) + math.log(1.0 - 0.1)) / 2
    assert float(loss) == pytest.approx(expected, abs=1e-6)
    # Dropping the valid row changes nothing about the value; it is excluded by
    # the mask, not merely down-weighted.
    only_row = key_responsibility_loss(
        probabilities[:1], responsibility[:1], valid[:1], mask[:1]
    )
    assert float(only_row) == pytest.approx(expected, abs=1e-6)


def test_calibration_report_detects_a_correlated_and_an_uncorrelated_proxy():
    exact = torch.tensor([3.0, 2.0, 1.0, 0.0])
    correlated = calibration_report(exact * 0.5, exact, top_k=2)
    assert correlated["pearson"] > 0.99
    assert correlated["sign_agreement"] == 1.0
    assert correlated["top1_agreement"] == 1.0
    anti = calibration_report(-exact, exact, top_k=2)
    assert anti["pearson"] < -0.99
    assert anti["top1_agreement"] == 0.0


def test_pair_rerank_reports_the_regret_of_the_gate_chosen_pair():
    """Spec §35: the main experiment's `false` must be a measured claim."""
    deployed = torch.tensor([1.0, 4.0])
    # Row 0: the gate-chosen pair is the best of the measured set.
    # Row 1: a challenger is strictly better, so the gate ranking lost by 1.0.
    alternatives = torch.tensor([[1.0, 1.5, 2.0], [4.0, 3.0, 5.0]])
    report = pair_rerank_report(
        deployed, alternatives, torch.zeros(2, dtype=torch.long)
    )
    assert report["samples"] == 2
    assert report["pairs_measured"] == 3
    assert report["mean_regret"] == pytest.approx(0.5)
    assert report["best_pair_is_deployed_rate"] == pytest.approx(0.5)
    assert report["mean_best_pair_loss"] == pytest.approx((1.0 + 3.0) / 2)
    with pytest.raises(ValueError):
        pair_rerank_report(deployed, alternatives[:1], torch.zeros(2, dtype=torch.long))


def test_pair_rerank_defaults_off_and_cannot_be_declared_without_calibration():
    assert V9Config().inference.pair_rerank is False
    assert V9Config(inference=V9InferenceConfig(pair_rerank=True)).inference.pair_rerank
    raw = V9Config().to_dict()
    raw["validation"] = dict(raw["validation"], contribution_calibration=False)
    raw["inference"] = dict(raw["inference"], pair_rerank=True)
    with pytest.raises(ValueError) as error:
        assert_frozen_contract(V9Config.from_dict(raw))
    assert "pair_rerank" in str(error.value)


@pytest.mark.parametrize(
    "path",
    ["configs/v9s_main.yaml", "configs/v9s_preflight.yaml"],
)
def test_shipped_configs_load_and_declare_the_main_recipe(path):
    import yaml

    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    with open(root / path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = V9Config.from_dict(raw)
    assert_frozen_contract(config)
    assert config.method == "v9s"
    assert config.routing.type == "independent_sigmoid"
    assert config.routing.max_inference_experts == 2
    assert config.routing.direct_answer_gradient_to_key is False
    assert 0.05 <= config.bootstrap.gate_floor <= 0.1
    assert config.inference.task_id is False
    assert config.inference.global_multi_key is True
    assert config.exact_oracle.training is False
    assert config.loss.use_budget_loss is False
    # M = 2 and the wide recall are the method, not a tuning choice; the
    # preflight must exercise the same ones or it proves nothing.
    assert config.candidate_count == 2
    assert config.wide_retrieval.enabled is True
    assert config.wide_retrieval.top_c > config.historical_retrieval.top_c
    if "preflight" not in path:
        # Spec §35 and §16 apply to the reported experiment, not to the
        # deliberately compressed preflight that only exercises the code path.
        assert config.inference.pair_rerank is False, "spec §35 main experiment"
        assert config.schedule.bootstrap_ratio <= 0.10, "the formal bootstrap share"
        assert 0.60 <= config.schedule.soft_ratio <= 0.75
        assert 0.20 <= config.schedule.hard_ratio <= 0.30
    assert path in {
        "configs/v9s_main.yaml",
        "configs/v9s_preflight.yaml",
    }


# ----------------------------------------------------------------------
# objective
# ----------------------------------------------------------------------
def test_total_loss_is_the_declared_objective():
    config = V9Config()
    probabilities = torch.tensor([[0.9, 0.6, 0.1, 0.05]], requires_grad=True)
    responsibility = torch.tensor([[0.8, 0.2, 0.0, 0.0]])
    valid = torch.tensor([True])
    mask = torch.tensor([[True, True, True, True]])
    answer = torch.tensor(2.5)
    terms = compose_total_loss(
        answer, probabilities, responsibility, valid, mask, config.loss
    )
    expected = (
        answer
        + config.loss.lambda_key * terms.key
        + config.loss.lambda_sparse * terms.sparse
    )
    assert torch.allclose(terms.total, expected)
    assert torch.allclose(terms.sparse, probabilities.sum())
    # Off by default: the budget term is still *computed* -- so the logged value
    # stays comparable across runs -- but it contributes no gradient, which is
    # the difference between reporting a quantity and optimising it.
    assert config.loss.use_budget_loss is False
    assert torch.allclose(
        terms.budget, (probabilities.sum() - config.loss.sparse_budget).clamp_min(0) ** 2
    )
    gradient = torch.autograd.grad(terms.total, probabilities, retain_graph=True)[0]
    without = torch.autograd.grad(
        answer
        + config.loss.lambda_key * terms.key
        + config.loss.lambda_sparse * terms.sparse,
        probabilities,
        retain_graph=True,
    )[0]
    assert torch.allclose(gradient, without)


def test_the_budget_term_joins_the_objective_only_when_enabled():
    config = V9Config()
    enabled = dataclasses.replace(config.loss, use_budget_loss=True)
    probabilities = torch.tensor([[0.9, 0.6, 0.5, 0.5]])
    responsibility = torch.tensor([[0.8, 0.2, 0.0, 0.0]])
    valid = torch.tensor([True])
    mask = torch.tensor([[True, True, True, True]])
    answer = torch.tensor(2.5)
    terms = compose_total_loss(
        answer, probabilities, responsibility, valid, mask, enabled
    )
    assert terms.budget.item() > 0.0
    expected = (
        answer
        + enabled.lambda_key * terms.key
        + enabled.lambda_sparse * terms.sparse
        + enabled.lambda_budget * terms.budget
    )
    assert torch.allclose(terms.total, expected)


def test_the_sparse_penalty_does_not_depend_on_the_row_width():
    """The same gates must cost the same on task 0 and on a wider task."""
    probabilities = torch.tensor([[0.4, 0.3, 0.2]])
    narrow = torch.ones_like(probabilities, dtype=torch.bool)
    wide = torch.cat([narrow, torch.zeros(1, 5, dtype=torch.bool)], dim=1)
    padded = torch.cat([probabilities, torch.zeros(1, 5)], dim=1)
    assert torch.allclose(sparse_loss(padded, wide), sparse_loss(probabilities, narrow))


def test_budget_penalty_is_one_sided():
    below = torch.tensor([[0.1, 0.1, 0.1]])
    mask = torch.ones_like(below, dtype=torch.bool)
    assert float(budget_loss(below, mask, 2.0)) == 0.0
    assert float(budget_loss(torch.tensor([[0.9, 0.9, 0.9]]), mask, 2.0)) > 0.0


# ----------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------
def test_multi_key_aggregation_is_a_max_over_the_experts_own_memory():
    config = V9Config()
    pool, _ = _pool(config, historical=3, candidates=0)
    expert_id = pool.historical_ids[0]
    base = pool.effective_key_matrix([pool.base_key_id(expert_id)], detach=True)[0]
    key_id = pool.add_task_key(expert_id, task_id=ROUTING_TASK)
    with torch.no_grad():
        pool.keys[key_id].copy_(torch.full((QUERY_DIM,), 0.2))
    retained = pool.effective_key_matrix([key_id], detach=True)[0]

    query = F.normalize(base, dim=-1).unsqueeze(0)
    router = V9InferenceRouter(pool)
    result = router(query)
    assert expert_id in result.pool_expert_ids.tolist()
    index = result.pool_expert_ids.tolist().index(expert_id)
    score = float(result.per_expert_scores[0, index])
    expected = max(float((query[0] @ base).item()), float((query[0] @ retained).item()))
    assert score == pytest.approx(expected, abs=1e-6)
    assert score == pytest.approx(1.0, abs=1e-6)  # the query is the base key itself
    # Both keys survive in the expert's memory: that is what lets an expert
    # stay reachable after the intervening tasks are pruned.
    assert set(memory_key_ids(pool, expert_id)) == {
        pool.base_key_id(expert_id),
        key_id,
    }


def test_inference_never_routes_to_a_candidate_that_was_not_committed():
    config = V9Config()
    pool, candidate_ids = _pool(config, historical=2, candidates=2)
    assert set(aggregatable_expert_ids(pool)) == set(pool.historical_ids)
    assert not set(candidate_ids) & set(aggregatable_expert_ids(pool))
    result = V9InferenceRouter(pool)(_query_matrix(4, seed=2))
    assert not set(result.expert_ids.reshape(-1).tolist()) & set(candidate_ids)


def test_inference_takes_only_the_query_and_never_a_task_id():
    config = V9Config()
    pool, _ = _pool(config, historical=3, candidates=0)
    _with_task_keys(pool)
    router = V9InferenceRouter(pool)
    queries = _query_matrix(6, seed=31)
    # The router's signature is the answer: one tensor, no task index, no
    # ground-truth answer, no training recall cache.
    result = router(queries)
    assert result.expert_ids.shape[0] == 6
    policy = router.route_policy(queries)
    assert set(policy) == {"rows", "policy"}
    for row in policy["rows"]:
        assert set(row) == {"expert_ids", "scores", "key_ids"}
    assert config.inference.task_id is False


def test_inference_policy_rejects_a_non_live_expert():
    config = V9Config()
    pool, _ = _pool(config, historical=2, candidates=1)
    live = pool.historical_ids[0]
    validate_inference_policy(pool, {"rows": [{"expert_ids": [live]}]})
    with pytest.raises(V9InferenceError):
        validate_inference_policy(pool, {"rows": [{"expert_ids": [9999]}]})


def test_inference_module_is_purity_clean():
    audit = assert_v9_inference_purity()
    assert audit["pure"] is True, audit
    assert audit["forbidden_identifiers_found"] == []
    assert audit["forbidden_strings_found"] == []


# ----------------------------------------------------------------------
# lifecycle
# ----------------------------------------------------------------------
def test_pruned_expert_leaves_the_routing_pool_entirely():
    config = V9Config()
    pool, _ = _pool(config, historical=3, candidates=0)
    pruned = pool.historical_ids[0]
    for key_id in pool.key_ids(expert_id=pruned):
        pool.set_key_lifecycle(key_id, LIFECYCLE_PRUNED)
        pool.set_key_trainable(key_id, False)
    pool.expert_record(pruned)["lifecycle"] = LIFECYCLE_PRUNED
    assert pruned not in pool.historical_ids
    assert pruned not in memory_key_ids(pool, pruned)
    assert pruned not in aggregatable_expert_ids(pool)


def test_task_key_audit_resets_only_the_rejected_key():
    """A rejected key returns to the expert's base identity, not to zero.

    Resetting to zero would be a different statement: the expert would still be
    recallable, but through a key it never had.  The task taught this key
    nothing worth keeping, which is not the same as the expert being useless.
    """
    config = V9Config()
    pool, _ = _pool(config, historical=2, candidates=0)
    keep, drop = pool.historical_ids
    for expert_id in (keep, drop):
        key_id = pool.add_task_key(expert_id, task_id=ROUTING_TASK)
        with torch.no_grad():
            pool.keys[key_id].copy_(torch.full((QUERY_DIM,), 0.3))
    base_before = {
        expert_id: pool.keys[pool.base_key_id(expert_id)].detach().clone()
        for expert_id in (keep, drop)
    }
    statistics = {
        str(keep): {"usage_rate": 0.5, "mean_positive_contribution": 0.02},
        str(drop): {"usage_rate": 0.0, "mean_positive_contribution": 0.0},
    }
    decisions = audit_historical_task_keys(pool, statistics, config.audit, task_index=ROUTING_TASK)
    assert decisions["reset_task_key"] == [drop]
    assert decisions["retain_task_key"] == [keep]
    assert decisions["new_key_retention_rate"] == pytest.approx(0.5)
    apply_historical_task_key_audit(pool, decisions, task_index=ROUTING_TASK)

    dropped = pool.task_key_id(drop, ROUTING_TASK)
    assert torch.equal(pool.keys[dropped], pool.keys[pool.base_key_id(drop)])
    assert not pool.keys[dropped].requires_grad
    assert pool.key_record(dropped)["lifecycle"] == LIFECYCLE_PRUNED
    # The expert itself is untouched: still live, still recallable at its base,
    # and the reset moved the task key rather than the identity it was built on.
    assert drop in pool.historical_ids
    for expert_id in (keep, drop):
        assert torch.equal(
            pool.keys[pool.base_key_id(expert_id)], base_before[expert_id]
        )
    # And it routes through the key it had before this task ran.
    assert pool.routing_key_id(drop, ROUTING_TASK) == pool.base_key_id(drop)

    retained = pool.task_key_id(keep, ROUTING_TASK)
    assert pool.key_record(retained)["lifecycle"] == LIFECYCLE_HISTORICAL
    assert not pool.keys[retained].requires_grad
    assert retained in pool.historical_key_ids()
    assert pool.routing_key_id(keep, ROUTING_TASK) == retained
    pool.validate()


def test_key_state_round_trip_preserves_trainability_and_bytes():
    config = V9Config()
    pool, candidate_ids = _pool(config, historical=2, candidates=2)
    _with_task_keys(pool)
    pool.freeze_historical(current_task=ROUTING_TASK)
    trainable_before = set(pool.trainable_key_ids())
    payload = pool.export_state()
    assert payload["pool_kind"] == "v9s_multi_key"
    audit = V9KeyPool.from_state(payload, current_task=ROUTING_TASK).audit()
    assert audit["pool_kind"] == "v9s_multi_key"
    assert audit["num_legacy_residual_keys"] == 0
    assert audit["num_task_keys"] == len(pool.historical_ids)
    restored = V9KeyPool.from_state(payload, current_task=ROUTING_TASK)
    assert set(restored.trainable_key_ids()) == trainable_before
    records = payload["metadata"]["key_records"]
    assert restored.export_state()["metadata"]["key_records"] == records
    assert restored.historical_key_ids() == pool.historical_key_ids()
    for key_id in records:
        assert torch.equal(restored.keys[key_id], pool.keys[key_id])
        assert restored.keys[key_id].requires_grad == pool.keys[key_id].requires_grad
    # A round-trip through the pool the trainer reloads must leave the freeze
    # audit with nothing to report.
    assert restored.historical_checksums() == pool.historical_checksums()


def test_a_legacy_v9_residual_state_migrates_to_absolute_task_keys():
    """``legacy_v9_only``: an interrupted V9 v1 run resumes instead of restarting.

    The conversion must be *routing-preserving*: the migrated pool has to select
    for every query exactly what the old ``normalize(base + gamma * delta)``
    selected, or resuming would silently change the method mid-run.
    """
    config = V9Config()
    pool, _ = _pool(config, historical=2, candidates=0)
    expert_id = pool.historical_ids[0]
    untouched_expert = pool.historical_ids[1]
    base = pool.keys[pool.base_key_id(expert_id)].detach().clone()
    gamma = 0.7
    delta = torch.randn(QUERY_DIM) * 0.2

    payload = pool.export_state()
    # V9 v1 kept the residual *beside* the origin key rather than over it, so
    # the state has three keys where V9-S would have three of a different kind.
    residual_id = "e{}_t{}_task_residual".format(expert_id, ROUTING_TASK)
    payload["keys"][residual_id] = delta
    payload["metadata"]["key_records"][residual_id] = {
        "key_id": residual_id,
        "expert_id": expert_id,
        "task_id": ROUTING_TASK,
        "key_type": "task_residual",
        "lifecycle": LIFECYCLE_HISTORICAL,
        "trainable": True,
        "support_count": 0,
        "teacher_gain": 0.0,
        "validation_gain": 0.0,
        "extra": {},
    }
    payload["pool_kind"] = "v9_task_residual"
    payload["gamma"] = gamma

    migrated = V9KeyPool.from_state(payload, current_task=None)
    alias_id = migrated.routing_key_id(expert_id, ROUTING_TASK)
    assert alias_id == "e{}_t{}_task_alias".format(expert_id, ROUTING_TASK)
    resolved = F.normalize(base + gamma * delta, dim=-1)
    assert torch.allclose(migrated.keys[alias_id], resolved, atol=1e-6)
    # The base key survives the conversion byte for byte: the residual moved,
    # the identity the expert was committed with did not.
    assert torch.equal(migrated.keys[migrated.base_key_id(expert_id)], base)
    # An expert that had no residual keeps routing at its base, so the
    # migration is a no-op for every part of the pool V9 v1 left alone.
    assert (
        migrated.routing_key_id(untouched_expert, ROUTING_TASK)
        == migrated.base_key_id(untouched_expert)
    )
    assert migrated.key_record(alias_id)["extra"]["legacy_v9_only"][
        "converted_from"
    ] == "task_residual"

    queries = _query_matrix(8, seed=41)
    before = queries @ resolved.unsqueeze(0).T
    after = queries @ migrated.effective_key_matrix([alias_id], detach=True).T
    assert torch.allclose(before, after, atol=1e-6)
    # Nothing in the V9-S package writes the legacy marker, so a fresh pool can
    # never come back through this path.
    assert pool.export_state()["pool_kind"] == "v9s_multi_key"


# ----------------------------------------------------------------------
# diagnostics (spec §34)
# ----------------------------------------------------------------------
def test_the_deployed_gate_is_the_top_two_indicator():
    """The rule the calibration scores against must be the deployed one.

    If the measurement used a *different* hard rule than the one inference
    serves, the soft/hard gap would be a property of the discrepancy rather than
    of the method.
    """
    config = V9Config()
    pool, _ = _pool(config, historical=6, candidates=config.candidate_count)
    _with_task_keys(pool)
    router, route, _, _ = _route(config, pool, 8, STAGE_SOFT)

    deployed = router.deployed_gates(route.probabilities, route.slot_mask)
    assert deployed.shape == route.probabilities.shape
    # Exactly ``max_inference_experts`` ones per row, and never on a masked slot.
    assert torch.equal(
        deployed.eq(1.0).sum(dim=1),
        torch.full((8,), float(config.routing.max_inference_experts)).long(),
    )
    assert not bool((deployed * ~route.slot_mask).any().item())
    # The chosen set is the top-2 of the *probability* the soft stage produced.
    top = route.probabilities.masked_fill(~route.slot_mask, -1.0).topk(
        config.routing.max_inference_experts, dim=1
    ).indices
    chosen = deployed.gt(0).nonzero(as_tuple=False)[:, 1].reshape(8, -1)
    assert torch.equal(chosen.sort(dim=1).values, top.sort(dim=1).values)
    # It is a decision, not a differentiable gate: it carries no key gradient.
    assert not deployed.requires_grad
    # The soft stage keeps no hard gate of its own -- there is nothing to serve
    # from -- so the deployed rule is derived here rather than read off.
    assert route.hard_gates is None


def test_the_hard_stage_gate_is_the_deployed_rule_not_a_surrogate():
    """ST Hard Top-2 must *be* deployment, not an approximation of it."""
    config = V9Config()
    pool, _ = _pool(config, historical=6, candidates=config.candidate_count)
    _with_task_keys(pool)
    router, _, queries, topc = _route(config, pool, 8, STAGE_SOFT)
    hard_route = router.route(queries, topc.expert_ids, 1.0, STAGE_HARD)
    assert hard_route.hard_gates is not None
    assert torch.equal(hard_route.forward_gates, hard_route.hard_gates)
    assert hard_route.forward_gates.requires_grad is False
    assert torch.equal(
        hard_route.forward_gates,
        router.deployed_gates(hard_route.probabilities, hard_route.slot_mask),
    )


def test_the_reported_validators_cover_every_spec_34_quantity():
    """The diagnostics list is a contract, not a suggestion.

    Each name below is read by an operator or by the task-end audit; a silent
    rename would leave the run looking healthy while reporting nothing.
    """
    from compose.v9.contribution import contribution_statistics

    statistics = contribution_statistics(
        torch.tensor([[0.5, -0.25, 0.75, -1.0]]), torch.ones(1, 4, dtype=torch.bool)
    )
    assert set(statistics) >= {
        "mean",
        "mean_positive",
        "mean_negative",
        "positive_rate",
        "std",
        "responsibility_mean",
    }
    # The responsibility mean is taken over the positive part only: a negative
    # contribution is evidence against the expert, not negative credit.
    assert statistics["responsibility_mean"] == pytest.approx((0.5 + 0.75) / 4)

    config = V9Config()
    assert config.validation.contribution_calibration is True, (
        "§34's soft/hard validation scores are produced by the calibration pass; "
        "turning it off removes them"
    )
