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
import json
import math
import types

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
from compose.v9.data import build_task_retrieval
from compose.v9.retrieval import (
    HistoricalTopC,
    V9RetrievalError,
    build_historical_topc,
    is_wide_step,
    load_or_build_historical_topc,
    retrieval_diagnostics,
)
from compose.v9.router import V9RouteOutput, V9Router
from compose.v9.schedule import (
    STAGE_BOOTSTRAP,
    STAGE_HARD,
    STAGE_SOFT,
    V9StageScheduler,
)
from compose.v9.trainer import (
    V9ComposeTrainer,
    V9TrainerError,
    answer_loss_key_gradient,
)

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


def test_forcing_a_rebuild_writes_the_cache_it_was_asked_for(tmp_path):
    """``force_build`` means "do not read one", never "do not write one".

    The validation recall set is built with ``force_build=True`` and its path is
    then handed to the training process, which loads it half an hour later.  A
    rebuild that derives into memory and stops there leaves the manifest
    pointing at a file nothing created -- a crash in a later stage, in a
    different process, reporting a path the stage that failed never named.
    """
    config = V9Config()
    pool, _ = _pool(config, historical=8, candidates=2)
    queries = _query_matrix(12, seed=23)
    ids = ["s{}".format(index) for index in range(12)]
    path = tmp_path / "historical_topc_val.pt"
    built = build_task_retrieval(
        key_pool=pool,
        queries=queries,
        sample_ids=ids,
        config=config,
        task_index=ROUTING_TASK,
        cache_path=str(path),
        force_build=True,
    )
    assert path.is_file(), "the caller was handed a path with nothing behind it"
    loaded = HistoricalTopC.load(str(path))
    assert loaded.sample_ids == ids
    assert torch.equal(loaded.expert_ids, built.expert_ids)
    # And the file is what a later reader gets: forcing a rebuild over an
    # existing cache replaces it rather than leaving the stale rows in place.
    later = build_task_retrieval(
        key_pool=pool,
        queries=queries,
        sample_ids=ids,
        config=config,
        task_index=ROUTING_TASK,
        cache_path=str(path),
        force_build=False,
    )
    assert torch.equal(later.expert_ids, built.expert_ids)


def test_the_dataset_names_a_sample_the_way_the_encoder_did():
    """The two halves of the query cache must agree on what a sample is called.

    The encoder reads ``id``, then ``question_id``, and the training split
    carries the first while the validation split carries the second.  A dataset
    that reads only ``id`` cannot address a validation cache at all: all 64
    samples report as missing, in the stage that loads them, on a cache that
    exists and is correct.  Checked against the encoder's own function rather
    than against a copy of its rule.
    """
    from compose.eval.query_features import shard_expected_ids
    from compose.train.data import _query_cache_sample_id

    for record in (
        {"id": "v7_t0_train_7", "question_id": "ignored"},
        {"question_id": "v7_t0_val_7"},
        {"id": None, "question_id": "v7_t0_val_8"},
    ):
        index = 3
        assert _query_cache_sample_id(record, index) == shard_expected_ids(
            [record], 1, 0
        )[0], record
    # Position is the last resort, and only the last: a cache whose ids are
    # missing everywhere is addressed by position, which is what the encoder
    # itself cannot express and therefore never writes.
    assert _query_cache_sample_id({}, 3) == "3"


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
    assert torch.equal(route.probabilities, route.answer_gates)
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
    # The forward gate is the straight-through form `hard - p.detach() + p`,
    # whose value is exactly the indicator and whose derivative w.r.t. `p` is
    # the identity.  Both halves matter: the first is what deployment serves,
    # the second is what keeps the last stage of training able to measure a
    # contribution at all -- a bare indicator would report dL_ans/da = 0 for
    # every expert and supervise nothing.
    assert route.forward_gates.requires_grad
    assert route.forward_gates.grad_fn is not None
    # Exactly the Top-2 by probability, per row.
    top = torch.topk(route.probabilities.detach(), k, dim=1).indices
    selected = route.forward_gates.bool().nonzero()[:, 1].reshape(16, k)
    assert torch.equal(selected.sort(dim=1).values, top.sort(dim=1).values)


def test_the_answer_gate_is_differentiable_but_reaches_no_key():
    """Spec §38, and the two halves of the contract it protects.

    The gate the composition consumes must **carry a graph** -- the answer has
    to be differentiable w.r.t. it, or the contribution is identically zero and
    the keys are supervised by nothing.  It must **not** carry a key: a key
    behind that graph would be a second, disagreeing answer-side gradient on the
    same parameter.

    The distinction is ``p.detach()`` versus *building p from detached inputs*.
    The first satisfies the second half and silently destroys the first; only
    the second satisfies both.
    """
    config = V9Config()
    pool, candidate_ids = _pool(config, historical=4, candidates=config.candidate_count)
    _with_task_keys(pool)
    pool.freeze_historical(current_task=ROUTING_TASK)
    router, route, _, _ = _route(config, pool, 8, STAGE_SOFT)
    keys = router.trainable_parameters()["key"]
    assert keys, "the fixture must expose at least one trainable key"

    # Half one: the answer is a function of the gate, so a contribution exists.
    # It is a *leaf* that requires grad -- an input variable the loss is written
    # in terms of, which is the only way a quantity with no parameter history
    # can still be differentiated against.
    assert route.answer_gates.requires_grad
    assert route.answer_gates.is_leaf
    probe = (route.answer_gates * torch.arange(
        route.answer_gates.numel(), dtype=route.answer_gates.dtype
    ).reshape(route.answer_gates.shape)).sum()
    assert float(torch.autograd.grad(probe, route.answer_gates, retain_graph=True)[0].abs().sum()) > 0.0

    # Half two: no key is behind it -- not partially, not through the bias.
    for key in keys:
        assert torch.autograd.grad(
            probe, key, retain_graph=True, allow_unused=True
        )[0] is None
    assert torch.autograd.grad(
        probe, router.bias, retain_graph=True, allow_unused=True
    )[0] is None

    # ...while `probabilities` stays differentiable w.r.t. those same keys: that
    # is the copy ``L_key`` trains, and the value the two copies agree on.
    assert torch.allclose(route.probabilities.detach(), route.answer_gates.detach())
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


def test_answer_loss_key_gradient_reports_a_frozen_key_rather_than_differentiating_it():
    """A frozen key cannot be an input to ``torch.autograd.grad``.

    It is not a differentiable leaf, so autograd raises rather than returning
    zero -- the gradient is undefined, not absent.  The audit still has to cover
    such a key, so it reports the structural answer: there is no edge to carry an
    answer gradient into it.
    """
    trainable = torch.nn.Parameter(torch.full((4,), 0.5))
    frozen = torch.nn.Parameter(torch.full((4,), 2.0), requires_grad=False)
    loss = trainable.sum() + frozen.detach().sum()
    report = answer_loss_key_gradient(loss, {"new": trainable, "old": frozen})
    assert set(report) == {"new", "old"}
    assert report["new"] > 0.0
    assert report["old"] == 0.0
    # And the same call without the fix is an error, not a zero: this is the
    # assertion that the two are not interchangeable.
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(loss, [frozen], allow_unused=True)


def test_the_isolation_audit_covers_a_frozen_key_instead_of_crashing_on_it():
    """The production call, on the task shape that first has a frozen key.

    ``_assert_answer_loss_isolated_from_keys`` hands autograd the *whole* key
    set -- the trainable candidates and task keys and the frozen base keys of the
    historical experts -- to show that ``L_ans`` reaches none of them.  Task 0
    owns no historical expert, so that set is entirely trainable there and every
    task-0 preflight passed; the first task with a committed pool raised ``One of
    the differentiated Tensors does not require grad`` on its first training
    step, after the handoff.  What the audit must do with a frozen key is
    *report* it: it is covered, and the report says on what grounds.
    """
    config = V9Config()
    pool, candidate_ids = _pool(config, historical=4, candidates=config.candidate_count)
    _with_task_keys(pool)
    pool.freeze_historical(current_task=ROUTING_TASK)
    router, route, _, _ = _route(config, pool, 4, STAGE_SOFT)

    generator = torch.Generator().manual_seed(11)
    mixed = torch.zeros(4, 8)
    for slot in range(route.slots):
        for row in range(route.batch_size):
            expert_id = int(route.expert_ids[row, slot])
            if expert_id < 0 or not bool(route.slot_mask[row, slot]):
                continue
            mixed[row] = mixed[row] + route.forward_gates[row, slot] * torch.randn(
                8, generator=generator
            )
    answer_loss = (mixed ** 2).mean()

    stub = types.SimpleNamespace(v9_key_pool=pool, v9_router=router)
    V9ComposeTrainer._assert_answer_loss_isolated_from_keys(stub, answer_loss, route)
    report = stub.v9_answer_key_isolation
    frozen = set(pool.historical_key_ids())
    assert frozen, "the fixture must own frozen keys or it cannot reproduce the crash"
    assert set(report["frozen_keys_not_differentiated"]) == frozen
    assert report["measured_keys"] == report["keys_checked"] - len(frozen)
    assert report["measured_keys"] > 0, "the trainable keys must still be measured"
    assert report["max_abs_gradient"] == 0.0
    assert report["nonzero_keys"] == []
    # The report is written to the task's output directory at the end of the
    # run, so it has to survive ``json.dump``.  A tensor or a parameter in here
    # would raise *after* the training loop, which is the worst possible moment:
    # the work is done and the run still ends without its artefacts.
    assert json.loads(json.dumps(report, sort_keys=True)) == report


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


def test_a_gate_that_cannot_be_differentiated_is_an_error_not_a_zero():
    """Regression: the silent zero that hid a full-length unsupervised run.

    ``gate_gradient`` used to return ``zeros_like(gates)`` when the gate did not
    require grad.  Every downstream quantity -- contribution, responsibility,
    ``L_key`` -- then logged a clean ``0.0``, which reads exactly like "measured,
    and the answer had no preference".  It means the opposite: nothing was
    measured.  Both disconnections raise now.
    """
    detached = torch.tensor([[0.7, 0.2]], requires_grad=False)
    loss = (detached * torch.tensor([[2.0, -1.0]])).sum()
    with pytest.raises(RuntimeError, match="does not require grad"):
        gate_gradient(loss, detached)
    with pytest.raises(RuntimeError, match="does not require grad"):
        local_conditional_contribution(loss, detached)

    # A gate that requires grad but that the loss never consumed is the same
    # failure wearing the other face: the graph is not connected to it.
    unused = torch.tensor([[0.7, 0.2]], requires_grad=True)
    unrelated = torch.tensor(3.0, requires_grad=True)
    with pytest.raises(RuntimeError, match="does not depend on the gate"):
        gate_gradient(unrelated * 2.0, unused)


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
        loss, route.probabilities, retain_graph=True, values=route.forward_gates
    )
    without = local_conditional_contribution(loss, route.probabilities, retain_graph=False)
    assert torch.allclose(without, -(route.probabilities.detach() * 1.0))
    assert torch.allclose(with_values, -(route.forward_gates.detach() * 1.0))
    assert not torch.allclose(with_values, without)


def test_a_composition_through_the_answer_gate_has_a_contribution_and_no_key():
    """The regression test for the failure this file was written to catch.

    A gate that is *detached* is not the same thing as a gate that is *built
    from detached inputs*.  The first severs the answer loss from the gate
    entirely: the composition consumes a tensor the loss is not a function of,
    ``dL_ans/da`` is exactly zero, the responsibility has no signal, and the run
    trains the candidate LoRAs with the keys frozen at their k-means seeding --
    while every metric that would have shown it reads a clean 0.0.

    So the test composes the way the model does, through ``answer_gates``, and
    asserts both halves at once: the contribution is non-zero, and no key has a
    gradient from it.
    """
    config = V9Config()
    pool, candidate_ids = _pool(config, historical=4, candidates=config.candidate_count)
    _with_task_keys(pool)
    pool.freeze_historical(current_task=ROUTING_TASK)
    router, route, _, _ = _route(config, pool, 8, STAGE_SOFT)
    keys = router.trainable_parameters()["key"]

    generator = torch.Generator().manual_seed(11)
    outputs = {
        int(expert_id): torch.randn(8, 16, generator=generator)
        for expert_id in route.expert_ids.unique().tolist()
    }
    stacked = torch.stack(
        [outputs[int(value)] for value in route.expert_ids.reshape(-1).tolist()]
    )
    mixture = (route.answer_gates.reshape(-1, 1) * stacked.reshape(8, -1, 16)).sum(dim=1)
    answer_loss = (mixture ** 2).mean()

    contribution = local_conditional_contribution(
        answer_loss, route.answer_gates, retain_graph=True, values=route.answer_gates
    )
    assert float(contribution.abs().sum()) > 0.0, (
        "the answer loss is a function of no gate; the responsibility teacher "
        "would be identically zero"
    )
    # The keys are still outside that graph, which is the other half.
    for key in keys:
        assert torch.autograd.grad(
            answer_loss, key, retain_graph=True, allow_unused=True
        )[0] is None


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
    # ``pair_is_deployed`` is the probe's *indicator* (1.0 = column 0 is what the
    # deployed rule serves), which is what ``_exact_pair_losses`` returns.  It is
    # not a column index, and the two were once mixed up: comparing an arg-min
    # position against a 1.0 flag reported 0/N for a run in which column 0 was
    # served on every sample.
    report = pair_rerank_report(
        deployed, alternatives, torch.ones(2, dtype=torch.float32)
    )
    assert report["samples"] == 2
    assert report["comparable_samples"] == 2
    assert report["pairs_measured"] == 3
    assert report["mean_regret"] == pytest.approx(0.5)
    assert report["best_pair_is_deployed_rate"] == pytest.approx(0.5)
    assert report["mean_best_pair_loss"] == pytest.approx((1.0 + 3.0) / 2)
    with pytest.raises(ValueError):
        pair_rerank_report(deployed, alternatives[:1], torch.ones(2))
    # The degenerate case the preflight actually runs: M = 2 candidates, so the
    # probe has exactly one pair, that pair is column 0, and the deployed rule
    # serves it on every sample.  The rate of a measurement that could only come
    # out one way, and it is neither 0.0 nor a claim about column 0.
    single = pair_rerank_report(
        deployed, alternatives[:, :1], torch.ones(2, dtype=torch.float32)
    )
    assert single["pairs_measured"] == 1
    assert single["best_pair_is_deployed_rate"] == pytest.approx(1.0)
    # A sample whose deployed pair was never measured cannot answer the
    # question.  It leaves the denominator instead of counting as a failure --
    # and where nothing is left the rate is NaN, not a claim.
    none = pair_rerank_report(
        deployed, alternatives, torch.zeros(2, dtype=torch.float32)
    )
    assert none["comparable_samples"] == 0
    assert math.isnan(none["best_pair_is_deployed_rate"])


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


def test_a_row_the_answer_cannot_rank_still_pays_the_sparse_penalty():
    """``contribution.valid`` masks the key supervision, and only that.

    Regression: the sparse and budget terms were once masked by the same
    all-False row mask.  They then read exactly ``0.0`` on precisely the early
    steps whose routing is undecided -- the steps whose routing they exist to
    shape -- and nothing in the log distinguished "no pressure applied" from
    "pressure applied and satisfied".  The key term, by contrast, *should* be
    zero here: the answer made no statement about which expert helps.
    """
    config = V9Config()
    probabilities = torch.tensor([[0.9, 0.6], [0.4, 0.3]], requires_grad=True)
    responsibility = torch.zeros_like(probabilities)
    valid = torch.tensor([False, False])
    mask = torch.ones_like(probabilities, dtype=torch.bool)
    terms = compose_total_loss(
        torch.tensor(1.0), probabilities, responsibility, valid, mask, config.loss
    )
    assert float(terms.key) == 0.0, "nothing to distil: the answer ranked no expert"
    assert torch.allclose(terms.sparse, probabilities.detach().sum(dim=1).mean())
    assert float(terms.sparse) > 0.0
    # And it is not merely reported: the pressure has to reach the gates.
    assert float(torch.autograd.grad(terms.total, probabilities)[0].sum()) > 0.0


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
# gradient-window boundary
# ----------------------------------------------------------------------
class _BoundaryStub:
    """Just enough of the trainer for ``_at_sync_boundary`` to answer."""

    def __init__(self, accumulation: int) -> None:
        self.accumulation = accumulation
        self.state = types.SimpleNamespace(global_step=0)
        self.accelerator = types.SimpleNamespace(sync_gradients=False)
        self._v9_last_micro_was_boundary = False
        self._v9_global_step_seen = None
        self.v9_unsynced_optimizer_steps = 0

    def micro_step(self) -> bool:
        """One micro-step, then whatever the optimizer did in between."""
        boundary = V9ComposeTrainer._at_sync_boundary(self)
        if boundary:
            self.state.global_step += 1
        return boundary


def test_the_gradient_window_closes_once_per_accumulation_window():
    """Regression: the V9-S v1 boundary was constant within a window.

    It read ``(global_step + 1) % accumulation == 0``.  ``global_step`` only
    advances *at* the boundary being detected, so the expression was constant
    across a window: it fired on all eight micro-steps of one window in every
    eight and on none of the other seven.  The ranks then stepped the optimizer
    on their own local gradients for 44 of 50 steps, and the only symptom was
    the task-end audit's differing LoRA checksums.
    """
    stub = _BoundaryStub(accumulation=8)
    stub.accelerator.sync_gradients = True
    # Eight windows, one boundary each, in phase -- the real accelerate
    # behaviour for ``num_steps = gradient_accumulation_steps``.
    fired = []
    for index in range(64):
        stub.accelerator.sync_gradients = (index + 1) % 8 == 0
        fired.append(stub.micro_step())
    assert sum(fired) == 8
    assert all(fired[index * 8 + 7] and not any(fired[index * 8:index * 8 + 7])
               for index in range(8))
    assert stub.v9_unsynced_optimizer_steps == 0


def test_a_missed_gradient_window_raises_instead_of_diverging():
    """A boundary the trainer does not notice must not be silent."""
    stub = _BoundaryStub(accumulation=8)
    stub.accelerator.sync_gradients = False
    for _ in range(4):
        assert stub.micro_step() is False
    # The optimizer steps here without the trainer having called the window: the
    # ranks are now accumulating locally and would drift apart.
    stub.state.global_step += 1
    with pytest.raises(V9TrainerError, match="did not treat as a gradient window boundary"):
        stub.micro_step()
    assert stub.v9_unsynced_optimizer_steps == 1


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
    # The straight-through form: identical value, identity derivative.  A bare
    # indicator would serve the right experts and supervise nothing.
    assert hard_route.forward_gates.requires_grad
    assert float(torch.autograd.grad(
        hard_route.forward_gates.sum(), hard_route.answer_gates, retain_graph=True
    )[0].abs().sub(1.0).abs().max()) < 1e-6
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

    contribution = torch.tensor([[0.5, -0.25, 0.75, -1.0]])
    responsibility = torch.tensor([[0.6, 0.0, 0.4, 0.0]])
    statistics = contribution_statistics(
        contribution, responsibility, torch.ones(1, 4, dtype=torch.bool)
    )
    assert set(statistics) >= {
        "mean",
        "mean_positive",
        "mean_negative",
        "positive_rate",
        "std",
        "responsibility_mean",
        "responsibility_max",
    }
    # Regression: ``responsibility_mean`` used to be ``clamp(contribution, 0)``
    # averaged -- a second name for ``mean_positive * positive_rate``, which
    # could never disagree with the number printed beside it.  It is the mean of
    # the *teacher* now, so the two differ by construction.
    assert statistics["responsibility_mean"] == pytest.approx(0.25)
    assert statistics["responsibility_mean"] != pytest.approx(statistics["mean"])
    assert statistics["responsibility_max"] == pytest.approx(0.6)

    config = V9Config()
    assert config.validation.contribution_calibration is True, (
        "§34's soft/hard validation scores are produced by the calibration pass; "
        "turning it off removes them"
    )


# ----------------------------------------------------------------------
# the bounded calibration (spec §30, §35)
#
# This path only runs after a task's training, on held-out samples, from a
# separate process -- so it is the one part of the method a CPU suite can check
# against a closed form, and the one part a compressed preflight reaches last.
# Everything below drives the real methods with a model whose answer NLL is a
# known function of the selection, which makes ``G_exact`` and ``G_grad``
# predictable rather than merely finite.
# ----------------------------------------------------------------------
class _SelectionRecorder:
    """The ``selection_context`` the composition reads, and a log of it."""

    def __init__(self) -> None:
        self.current = None
        self.selections = []

    def selection_context(self, selection):
        recorder = self

        class _Context:
            def __enter__(self):
                self.previous = recorder.current
                recorder.current = selection
                recorder.selections.append(selection)
                return selection

            def __exit__(self, *_exc):
                recorder.current = self.previous
                return False

        return _Context()


class _NllStubModel:
    """A model whose per-sample answer NLL is a function of the selection.

    ``gate_sum``: ``L_b = sum_slot(gate)``.  Differentiable in the gate, so
    ``dL_ans/da == 1`` on every active slot and ``G_grad`` has the closed form
    ``-a``.

    ``expert_weight``: ``L_b = 10 - sum_slot(gate * (id + 1))``.  Removing
    expert ``k`` costs exactly the term ``k`` was contributing, which is the
    quantity §30 calibrates the gate-gradient estimate against.
    """

    def __init__(self, recorder, mode: str) -> None:
        self.recorder = recorder
        self.mode = mode
        self.training = True
        self.v7_per_sample_answer_nll = None

    def __call__(self, **_prepared):
        selection = self.recorder.current
        if selection is None:
            raise AssertionError("the model was called outside a selection context")
        # The body runs inside an activation checkpoint, exactly as the real
        # backbone does under ``gradient_checkpointing``.  That is what lets
        # this stub fail the way the GPU fails: recomputation during
        # ``backward()`` re-enters the body, and a body that can no longer see
        # the selection rebuilds a *different* graph.  The gates are passed as
        # checkpoint *inputs* because that is what arms recomputation at all --
        # a checkpoint called with no tensor inputs runs the body once, eagerly,
        # and never recomputes it.
        per_sample = torch.utils.checkpoint.checkpoint(
            self._block, selection.gates, selection.expert_ids,
            use_reentrant=False,
        )
        self.v7_per_sample_answer_nll = per_sample
        return None

    def _block(self, gates, ids):
        if self.recorder.current is None:
            raise AssertionError(
                "the checkpointed body was recomputed outside the selection "
                "context, so it would rebuild a different graph than the "
                "forward did"
            )
        if self.mode == "gate_sum":
            per_sample = gates.sum(dim=1)
        elif self.mode == "expert_weight":
            active = ids.ne(-1).to(gates.dtype)
            weight = ids.to(gates.dtype).add(1.0)
            per_sample = 10.0 - (gates * weight * active).sum(dim=1)
        else:
            raise AssertionError("unknown stub mode {!r}".format(self.mode))
        return per_sample

    def train(self, mode: bool = True):
        self.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)


def _calibration_stub(config, router, recorder):
    stub = types.SimpleNamespace(
        v9_config=config,
        v9_router=router,
        expert_pool=types.SimpleNamespace(manager=recorder),
        v9_extra_forwards=0,
        _v9_pair_probe=None,
    )
    # The pair probe is a method on the trainer, not a value: bind it so the
    # stub answers ``_pair_probe`` the way the real attribute would.
    stub._pair_probe = types.MethodType(V9ComposeTrainer._pair_probe, stub)
    return stub


def test_exact_removal_measures_one_expert_removed_per_forward():
    """``G_exact`` must be a per-sample, per-slot quantity with no broadcast luck.

    Two shape facts are load-bearing and neither is visible from the numbers:
    the answer loss is per *sample* while the matrix is per *slot*, so the delta
    has to be oriented down the slot axis; and a removed expert is removed by
    padding its slot, which the composition only accepts with a zero gate.  The
    fixture is deliberately ``[3, 5]`` -- a width mismatch would be caught by
    broadcasting here rather than by a recorded value that looks plausible.
    """
    config = V9Config()
    pool, _ = _pool(config, historical=3, candidates=config.candidate_count)
    _with_task_keys(pool)
    router = _router(config, pool)
    queries = _query_matrix(3, seed=5)
    topc = _recall(pool, queries, config)
    route = router.route(queries, topc.expert_ids, 1.0, STAGE_SOFT)
    assert route.expert_ids.shape == (3, 5), route.expert_ids.shape

    recorder = _SelectionRecorder()
    model = _NllStubModel(recorder, mode="expert_weight")
    stub = _calibration_stub(config, router, recorder)
    _, _, base = V9ComposeTrainer._calibration_gradients(
        stub, model, queries, topc.expert_ids, {}
    )
    matrix = V9ComposeTrainer._exact_removal_matrix(stub, model, {}, route, base)

    ids = route.expert_ids
    gates = route.forward_gates
    mask = route.slot_mask
    expected = torch.zeros_like(matrix)
    for row in range(ids.shape[0]):
        for slot in range(ids.shape[1]):
            if not bool(mask[row, slot]):
                continue
            expert_id = int(ids[row, slot])
            expected[row, slot] = float(gates[row, slot].item()) * (expert_id + 1)
    assert torch.allclose(matrix, expected, atol=1e-5), (matrix, expected)
    # One forward per distinct active expert, on top of the base forward.
    assert stub.v9_extra_forwards == len(
        {int(value) for value in ids[mask].tolist()}
    )
    # The base selection is the first one recorded; every later selection is an
    # alternative, and none of them may leave a gate on a padded slot.
    assert len(recorder.selections) == stub.v9_extra_forwards + 1
    for selection in recorder.selections[1:]:
        padded = selection.expert_ids.eq(-1)
        assert not bool(selection.gates[padded].abs().gt(0).any().item())
    # Every alternative removes exactly one expert, from every row that had it.
    for selection in recorder.selections[1:]:
        padded = selection.expert_ids.eq(-1)
        removed = {int(value) for value in route.expert_ids[padded].tolist()}
        assert len(removed) == 1, "one alternative removes exactly one expert"
        expert_id = removed.pop()
        assert int(padded.sum().item()) == int(route.expert_ids.eq(expert_id).sum().item())


def test_the_exact_removal_accumulator_shares_the_device_of_its_inputs():
    """The removal matrix must be allocated where its inputs are.

    ``torch.zeros(shape, dtype=...)`` defaults to CPU, and on CPU this file
    cannot see the difference: every other tensor is there too, so the fixture
    above passes while the same call on a GPU raises ``Expected all tensors to
    be on the same device`` -- in the post-training calibration, after the
    training loop has already finished and written its checkpoints.  That is the
    one place a run cannot afford to be wrong, so the check is a real execution
    of the real method with the routing tensors on a second device, and it skips
    only where no second device exists.
    """
    if not torch.cuda.is_available():
        pytest.skip("this failure needs a second device to exist")
    # ``cuda`` and ``cuda:0`` are different ``torch.device`` values even on a
    # one-GPU machine, so the pinned form is the one to compare against.
    device = torch.device("cuda", torch.cuda.current_device())
    config = V9Config()
    pool, _ = _pool(config, historical=3, candidates=config.candidate_count)
    _with_task_keys(pool)
    router = _router(config, pool)
    # The recall is built on CPU because that is where the task builds it: its
    # own scoring is device-checked, and the router moves the row to the query
    # device itself (``route`` -> ``historical_rows.to(queries.device)``).
    topc = _recall(pool, _query_matrix(3, seed=5), config)
    pool = pool.to(device)
    queries = _query_matrix(3, seed=5).to(device)
    route = router.route(queries, topc.expert_ids, 1.0, STAGE_SOFT)
    assert route.expert_ids.device == device

    recorder = _SelectionRecorder()
    model = _NllStubModel(recorder, mode="expert_weight")
    stub = _calibration_stub(config, router, recorder)
    _, _, base = V9ComposeTrainer._calibration_gradients(
        stub, model, queries, topc.expert_ids, {}
    )
    assert base.device == device
    matrix = V9ComposeTrainer._exact_removal_matrix(stub, model, {}, route, base)

    # Device agreement is the point; the values are the fixture above, so a fix
    # that moved the accumulator by *accident* still has to produce them.
    assert matrix.device == route.expert_ids.device
    ids = route.expert_ids
    gates = route.forward_gates
    mask = route.slot_mask
    expected = torch.zeros_like(matrix)
    for row in range(ids.shape[0]):
        for slot in range(ids.shape[1]):
            if not bool(mask[row, slot]):
                continue
            expected[row, slot] = float(gates[row, slot].item()) * (
                int(ids[row, slot]) + 1
            )
    assert torch.allclose(matrix, expected, atol=1e-5), (matrix, expected)


def test_the_calibration_gradient_is_taken_against_the_gate_that_deployed():
    """``G_grad`` on held-out data is ``-a * dL_ans/da``, on the deployed gate.

    The calibration reuses the trained model, so it is the one place the
    detach contract can silently break a second time: if the gate it
    differentiates were not the tensor the composition consumed, §30 would be
    comparing the exact removal effect against a zero.

    The other half of the assertion is the scale.  The per-sample loss is
    differentiated, not the batch mean: a ``1/batch`` factor changes no
    correlation, but it would divide the reported ``grad_mean`` and so make the
    proxy look weaker than the ``exact_mean`` beside it by exactly the width of
    the batch -- the one comparison §30 exists to make.
    """
    config = V9Config()
    pool, _ = _pool(config, historical=3, candidates=config.candidate_count)
    _with_task_keys(pool)
    router = _router(config, pool)
    queries = _query_matrix(4, seed=7)
    topc = _recall(pool, queries, config)
    recorder = _SelectionRecorder()
    model = _NllStubModel(recorder, mode="gate_sum")
    stub = _calibration_stub(config, router, recorder)
    _, local, base = V9ComposeTrainer._calibration_gradients(
        stub, model, queries, topc.expert_ids, {}
    )
    route = router.last_route
    # L_b = sum_slot(gate) -> dL/da = 1 -> G_ik = -a_ik, with a_ik the forward
    # gate the composition consumed.
    assert torch.allclose(local, -route.answer_gates.detach(), atol=1e-6)
    assert torch.allclose(base, route.answer_gates.detach().sum(dim=1), atol=1e-5)
    assert not local.requires_grad


def test_the_validation_gap_is_the_price_of_serving_the_hard_rule():
    """``gap`` is hard minus soft: what deployment costs, not what it saves."""
    config = V9Config()
    pool, _ = _pool(config, historical=3, candidates=config.candidate_count)
    _with_task_keys(pool)
    router = _router(config, pool)
    queries = _query_matrix(4, seed=11)
    topc = _recall(pool, queries, config)
    route = router.route(queries, topc.expert_ids, 1.0, STAGE_SOFT)
    recorder = _SelectionRecorder()
    model = _NllStubModel(recorder, mode="gate_sum")
    stub = _calibration_stub(config, router, recorder)
    scores = V9ComposeTrainer._validation_score_gap(stub, model, route, {}, None)
    # ``sum(gate)`` is exactly 2.0 under the two-hot deployed rule and the sum of
    # the mixture's sigmoids under the soft one: both scores are known, so the
    # sign convention is checked rather than assumed.
    soft = float(
        (route.probabilities.detach() * route.slot_mask).sum(dim=1).mean().item()
    )
    assert scores["hard_top2"] == pytest.approx(2.0)
    assert scores["soft"] == pytest.approx(soft, abs=1e-5)
    assert scores["gap"] == pytest.approx(scores["hard_top2"] - scores["soft"])
    assert abs(scores["gap"]) > 1e-3, "the fixture must separate the two rules"
    assert scores["batches"] == 1.0
    assert model.training is True, "the calibration restores the training mode"
    assert stub.v9_extra_forwards == 2


def test_pair_rerank_flags_a_pair_only_where_the_gate_deploys_it():
    """The §35 report must say which samples the deployed pair even applies to.

    The probe is a global, candidate-only pair set; the deployed rule picks two
    slots per row.  A sample whose deployed pair contains a recalled historical
    expert is not served the probed pair at all, and counting it as such would
    turn ``best_pair_is_deployed_rate`` into a statement about column 0.
    """
    raw = V9Config().to_dict()
    raw["expert"] = {**raw["expert"], "num_current_candidates": 3}
    config = V9Config.from_dict(raw)
    pool, candidate_ids = _pool(config, historical=1, candidates=3)
    _with_task_keys(pool)
    router = _router(config, pool)
    recorder = _SelectionRecorder()
    model = _NllStubModel(recorder, mode="expert_weight")
    stub = _calibration_stub(config, router, recorder)

    historical_id = pool.historical_ids[0]
    row_ids = torch.tensor(
        [[historical_id] + list(candidate_ids), [historical_id] + list(candidate_ids)]
    )
    gates = torch.tensor([[0.10, 0.90, 0.80, 0.10], [0.95, 0.50, 0.30, 0.10]])
    route = V9RouteOutput(
        expert_ids=row_ids,
        probabilities=gates,
        forward_gates=gates,
        slot_mask=row_ids.ne(-1),
        hard_gates=None,
        temperature=1.0,
        stage=STAGE_SOFT,
    )
    deployed, alternatives, flag = V9ComposeTrainer._exact_pair_losses(
        stub, model, {}, route
    )
    # Column 0 is the best candidate pair by summed gate mass: (c0, c1).
    assert alternatives.shape == (2, 3)
    assert torch.equal(deployed, alternatives[:, 0])
    # Row 0 deploys exactly that pair; row 1 deploys a historical expert plus c0.
    assert flag.tolist() == [1.0, 0.0]
    assert stub.v9_extra_forwards == 3
    for selection in recorder.selections:
        padded = selection.expert_ids.eq(-1)
        assert not bool(selection.gates[padded].abs().gt(0).any().item())
    # The probe is computed once and reused: ``p`` means one pair for every
    # batch, which is what makes the columns comparable across the sample.
    assert stub._v9_pair_probe == [(candidate_ids[0], candidate_ids[1]),
                                   (candidate_ids[0], candidate_ids[2]),
                                   (candidate_ids[1], candidate_ids[2])]
    # The probe and the report are two halves of one interface, and they were
    # tested apart -- the probe against its own flag, the report against a
    # hand-written index -- so both passed while the composition of the two was
    # wrong.  Composed here, on the probe's own output, the excluded row leaves
    # the denominator and the rate describes only the row it applies to.
    best = alternatives.argmin(dim=1)
    report = pair_rerank_report(deployed, alternatives, flag)
    assert report["comparable_samples"] == 1
    assert report["best_pair_is_deployed_rate"] == pytest.approx(float(best[0].eq(0)))


def test_committing_a_candidate_leaves_a_pool_the_next_task_can_load():
    """The audit has to *apply* the commit, not just decide it.

    Recording ``commit`` and leaving the lifecycle alone means the next task
    finds no historical expert, hands this task's candidates back to ``L_ans``,
    and refuses to start -- so the failure lands one task later, in a different
    process, with nothing in the log about the audit.
    """
    from compose.v9.audit import apply_candidate_commit, audit_candidates

    config = V9Config()
    pool, candidate_ids = _pool(config, historical=2, candidates=2)
    _with_task_keys(pool)
    good = {
        "usage": 100,
        "usage_rate": 1.0,
        "selected": 100,
        "selected_rate": 1.0,
        "effective_support": 1.0,
        "mean_contribution": 0.5,
        "mean_positive_contribution": 0.5,
        "positive_contribution_rate": 1.0,
    }
    dead = dict(good, usage=0, usage_rate=0.0, mean_positive_contribution=0.0,
                positive_contribution_rate=0.0)
    kept, dropped = candidate_ids[0], candidate_ids[1]
    decisions = audit_candidates(
        pool,
        {str(kept): good, str(dropped): dead},
        config.audit,
        ROUTING_TASK,
    )
    assert decisions["commit"] == [kept]
    assert decisions["delete"] == [dropped]
    applied = apply_candidate_commit(pool, None, decisions, ROUTING_TASK)
    assert applied["committed"] == [kept]
    assert pool.current_ids == []
    assert kept in pool.historical_ids
    assert dropped not in pool.historical_ids
    # And the committed pool is exactly what the next task loads: no leftover
    # candidate, on any of the three paths a candidate can take.
    restored = V9KeyPool.from_state(pool.export_state(), current_task=ROUTING_TASK + 1)
    assert restored.current_ids == []
    assert set(restored.historical_ids) == set(pool.historical_ids)
    assert restored.validate() is not None
