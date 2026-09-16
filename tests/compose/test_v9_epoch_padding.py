"""The full-data tail contract: every declared sample must reach an optimizer step.

The failure this file pins down is quiet in both directions.  A rank's
per-epoch micro-batch count ``ceil(ceil(N / micro) / world)`` can be odd; the
trailing micro-batch then opens an accumulation window that nothing closes,
because transformers 4.33.3 derives
``num_update_steps_per_epoch = len(loader) // gradient_accumulation_steps`` and
its partial-window escape hatch requires the whole epoch to be shorter than one
window.  The samples are forwarded, backwarded and dropped -- and counted as
covered, because the coverage audit counted forwards.

Two consequences the tests insist on:

* ``optimizer_steps`` must equal ``ceil(declared / GLOBAL_BATCH)``, recomputed
  from the declared split and the contract.  Comparing HF's number to V9's own
  number would not catch this: both are the same floor division, so they agree
  while both are short.
* coverage is reported twice -- forwards and optimizer-applied -- because the
  second is the claim the formal pipeline actually makes.

Run with ``python -m pytest tests/compose/test_v9_epoch_padding.py``.
"""

from __future__ import annotations

import math

import pytest

from compose.v7.training import full_data_coverage_audit
from compose.v9.epoch import (
    GlobalBatchPaddedSampler,
    V9EpochError,
    global_batch,
    plan_epoch,
    verify_epoch_plan,
)
from compose.v9.schedule import V9StageScheduler
from compose.v9.config import V9ScheduleConfig, V9RoutingConfig

WORLD, MICRO, GA = 4, 4, 2
GLOBAL = global_batch(WORLD, MICRO, GA)


def _shard(sampler, micro=MICRO, world=WORLD, ga=GA):
    """Per-rank batches, replicating accelerate's BatchSamplerShard.

    ``even_batches=True`` is accelerate's default and the reason the trailing
    batch is full rather than partial: it pads by repeating whole batches.
    """
    from torch.utils.data import BatchSampler
    from accelerate.data_loader import BatchSamplerShard

    batch_sampler = BatchSampler(sampler, batch_size=micro, drop_last=False)
    return [
        list(
            BatchSamplerShard(
                batch_sampler, num_processes=world, process_index=rank,
                split_batches=False, even_batches=True,
            )
        )
        for rank in range(world)
    ]


def _coverage(sampler, declared):
    """(forward ids, optimizer-applied ids, unclosed windows) over all ranks."""
    forward, applied, unclosed = set(), set(), 0
    for batches in _shard(sampler):
        closed = len(batches) - (len(batches) % GA)
        unclosed += len(batches) % GA
        for index, batch in enumerate(batches):
            forward.update(batch)
            if index < closed:
                applied.update(batch)
    declared_ids = set(range(declared))
    return forward & declared_ids, applied & declared_ids, unclosed


# --------------------------------------------------------------------------
# §6 TEST 1 -- divisible: no padding, and the count is what it always was
# --------------------------------------------------------------------------
def test_divisible_length_needs_no_padding():
    plan = plan_epoch(64, WORLD, MICRO, GA)
    assert plan.padded_epoch_samples == 64
    assert plan.padding_duplicate_count == 0
    assert plan.expected_optimizer_steps == 2
    assert plan.expected_optimizer_steps == math.ceil(64 / GLOBAL)


# --------------------------------------------------------------------------
# §6 TEST 2 -- ArxivQA: the case that made the old run invalid
# --------------------------------------------------------------------------
def test_indivisible_length_pads_to_the_next_global_batch():
    plan = plan_epoch(39720, WORLD, MICRO, GA)
    assert plan.padded_epoch_samples == 39744
    assert plan.padding_duplicate_count == 24
    assert plan.expected_optimizer_steps == 1242
    assert plan.expected_optimizer_steps == math.ceil(39720 / GLOBAL)
    # The bug this replaced: floor(len(loader) / ga) gave 1241.
    assert plan.per_rank_micro_batches // GA == 1242
    assert plan.per_rank_micro_batches % GA == 0


# --------------------------------------------------------------------------
# §6 TEST 3 -- every rank gets the same number of micro-batches, windows closed
# --------------------------------------------------------------------------
@pytest.mark.parametrize("declared", [64, 65, 23742, 39720, 39520, 29603, 39743, 38916])
def test_every_rank_agrees_and_every_window_closes(declared):
    sampler = GlobalBatchPaddedSampler(declared, WORLD, MICRO, GA, seed=42)
    shards = _shard(sampler)
    lengths = {len(batches) for batches in shards}
    assert len(lengths) == 1, "ranks disagree on the epoch length: {}".format(lengths)
    (length,) = lengths
    assert length % GA == 0, "{} micro-batches leaves a window open".format(length)
    assert length // GA == math.ceil(declared / GLOBAL)
    for batches in shards:
        assert all(len(batch) == MICRO for batch in batches)


# --------------------------------------------------------------------------
# §6 TEST 4 -- every declared id is applied by an optimizer step
# --------------------------------------------------------------------------
@pytest.mark.parametrize("declared", [65, 39720, 29603, 38916])
def test_every_declared_sample_reaches_an_optimizer_step(declared):
    sampler = GlobalBatchPaddedSampler(declared, WORLD, MICRO, GA, seed=42)
    forward, applied, unclosed = _coverage(sampler, declared)
    assert unclosed == 0
    assert forward == set(range(declared))
    assert applied == set(range(declared)), "{} samples were forwarded but never applied".format(
        len(set(range(declared)) - applied)
    )


# --------------------------------------------------------------------------
# §6 TEST 5 -- determinism: same seed, same padded order
# --------------------------------------------------------------------------
def test_same_seed_gives_the_same_padded_order():
    first = list(GlobalBatchPaddedSampler(39720, WORLD, MICRO, GA, seed=42))
    again = list(GlobalBatchPaddedSampler(39720, WORLD, MICRO, GA, seed=42))
    other = list(GlobalBatchPaddedSampler(39720, WORLD, MICRO, GA, seed=43))
    assert first == again
    assert first != other
    assert len(first) == 39744


def test_the_padding_repeats_the_epochs_own_opening_order():
    # world 1, micro 4, accumulation 2 -> unit 8, so 65 pads to 72 with 7 repeats.
    sampler = GlobalBatchPaddedSampler(65, 1, 4, 2, seed=7)
    order = list(sampler)
    assert len(order) == 72
    assert sorted(order[:65]) == list(range(65)), "the epoch is a permutation"
    # the tail is the head of that same permutation, not a fresh draw
    assert order[65:] == order[:7]


def test_set_epoch_reshuffles_and_repads_deterministically():
    sampler = GlobalBatchPaddedSampler(39720, WORLD, MICRO, GA, seed=42)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    sampler.set_epoch(0)
    assert list(sampler) == first
    assert second != first
    assert len(second) == len(first) == 39744


# --------------------------------------------------------------------------
# §6 TEST 6 -- the V9 stage schedule is defined over the real step count
# --------------------------------------------------------------------------
def test_scheduler_total_matches_the_contract_step_count():
    schedule = V9ScheduleConfig()
    routing = V9RoutingConfig()
    for declared in (23742, 39720, 39520, 29603, 39743, 38916):
        plan = plan_epoch(declared, WORLD, MICRO, GA)
        scheduler = V9StageScheduler(schedule, routing, plan.expected_optimizer_steps)
        assert scheduler.total_steps == math.ceil(declared / GLOBAL)
        # The last step of the run is the last hard step; a total one short
        # drops it.
        last = scheduler.state(plan.expected_optimizer_steps - 1)
        assert last.global_step == plan.expected_optimizer_steps - 1


# --------------------------------------------------------------------------
# §6 TESTS 7-9 -- the three shapes the task sizes actually take
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "declared,expected_steps,padded,padding",
    [
        (23742, 742, 23744, 2),     # Task 0 ImageNet-R -- already even
        (29603, 926, 29632, 29),    # Task 3 IconQA -- odd, was 925
        (38916, 1217, 38944, 28),   # Task 5 Flickr30k -- odd, was 1216
    ],
)
def test_the_three_task_shapes(declared, expected_steps, padded, padding):
    plan = plan_epoch(declared, WORLD, MICRO, GA)
    assert plan.expected_optimizer_steps == expected_steps
    assert plan.padded_epoch_samples == padded
    assert plan.padding_duplicate_count == padding
    sampler = GlobalBatchPaddedSampler(declared, WORLD, MICRO, GA, seed=42)
    forward, applied, unclosed = _coverage(sampler, declared)
    assert unclosed == 0 and applied == set(range(declared)) and forward == set(range(declared))


def test_the_whole_formal_task_table():
    """The six declared splits against the contract, recomputed not copied."""
    declared = {0: 23742, 1: 39720, 2: 39520, 3: 29603, 4: 39743, 5: 38916}
    affected_before = {1: 1241, 3: 925, 5: 1216}   # the old floor-division counts
    for task, count in declared.items():
        plan = plan_epoch(count, WORLD, MICRO, GA)
        assert plan.expected_optimizer_steps == math.ceil(count / GLOBAL)
        if task in affected_before:
            assert affected_before[task] != plan.expected_optimizer_steps
            assert plan.expected_optimizer_steps - affected_before[task] == 1


# --------------------------------------------------------------------------
# verify_epoch_plan is the startup gate: it must refuse the old behaviour
# --------------------------------------------------------------------------
def test_verify_rejects_a_dataloader_that_is_not_padded():
    plan = plan_epoch(39720, WORLD, MICRO, GA)
    verify_epoch_plan(plan, per_rank_dataloader_length=plan.per_rank_micro_batches)
    with pytest.raises(V9EpochError, match="dataloader yields"):
        verify_epoch_plan(plan, per_rank_dataloader_length=2483)


def test_verify_rejects_a_short_step_count():
    plan = plan_epoch(39720, WORLD, MICRO, GA)
    with pytest.raises(V9EpochError, match="optimizer steps"):
        verify_epoch_plan(plan, observed_optimizer_steps=1241)
    verify_epoch_plan(plan, observed_optimizer_steps=1242)


# --------------------------------------------------------------------------
# the coverage audit: optimizer side is a separate, stricter claim
# --------------------------------------------------------------------------
def _coverage_kwargs(declared, applied_count, unclosed=0):
    return {
        "num_train_samples": declared,
        "unique_sample_ids": [str(i) for i in range(declared)],
        "optimizer_micro_steps": 1,
        "optimizer_steps": math.ceil(declared / GLOBAL),
        "observed_sample_count": declared,
        "require_full": True,
        "optimizer_sample_ids": [str(i) for i in range(applied_count)],
        "unclosed_window_sample_ids": [str(declared - 1 - i) for i in range(unclosed)],
    }


def test_coverage_accepts_a_complete_epoch():
    result = full_data_coverage_audit(**_coverage_kwargs(2001, 2001))
    assert result["train_sample_coverage"] == 1.0
    assert result["optimizer_coverage"] == 1.0
    assert result["unclosed_window_sample_count"] == 0


def test_forward_coverage_alone_hides_the_dropped_tail():
    """The old audit's blind spot, stated as a test.

    Every sample was forwarded, so ``train_sample_coverage`` is 1.0 -- and the
    audit must still refuse, because four of them never reached a step.
    """
    with pytest.raises(RuntimeError, match="never stepped|did not apply"):
        full_data_coverage_audit(**_coverage_kwargs(2001, 1997, unclosed=4))


def test_coverage_artefact_records_both_sides():
    result = full_data_coverage_audit(**_coverage_kwargs(2001, 2001))
    assert result["unique_optimizer_applied_sample_ids"] == 2001
    assert result["unique_train_sample_ids_seen"] == 2001


def test_v7_callers_keep_the_forward_only_contract():
    """No optimizer ids passed -> the old shape, unchanged."""
    result = full_data_coverage_audit(2001, map(str, range(2001)), 2001, 32, 2001, True)
    assert result["train_sample_coverage"] == 1.0
    assert "optimizer_coverage" not in result


def test_a_plan_with_no_steps_is_not_a_plan():
    with pytest.raises(V9EpochError):
        plan_epoch(0, WORLD, MICRO, GA)
    with pytest.raises(V9EpochError):
        global_batch(0, MICRO, GA)
