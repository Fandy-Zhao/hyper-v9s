"""Tests for the V8 per-sample generation harness.

``compose/v8/generate.py`` supplies the 0/1/2-expert routing that V7's evaluator
cannot express, so it sits directly under every number the V8-A experiment
reports.  These tests pin the parts that fail *silently*: the route cache must
actually reach disk, it must never rewrite a cached answer, and each sample must
be generated under its own selection rather than one batch-wide route.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Mapping, Sequence

import pytest

from compose.v8 import generate
from compose.v8.config import STATE_BASE_ONLY, STATE_REUSE1, STATE_REUSE2
from compose.v8.generate import (
    GenerationEngine,
    GenerationError,
    RouteGenerationCache,
    experts_from_route,
    route_key,
)


class StubEngine(GenerationEngine):
    """A ``GenerationEngine`` whose model call returns a deterministic string.

    Only ``generate_route`` is under test here -- caching, grouping and
    per-sample selection -- so the 7B forward pass is stubbed out.  The stub
    records every call, which is what lets a test assert *which* experts a
    sample was generated under.
    """

    def __init__(self, cache_path=None) -> None:
        self.cache = RouteGenerationCache(cache_path) if cache_path else None
        self.generated_count = 0
        self.cache_hits = 0
        self.calls: List = []

    def _generate_one(self, record, experts):
        route = route_key(experts)
        sample_id = generate._record_id(record)
        self.calls.append((sample_id, route))
        return "answer|{}|{}".format(sample_id, route)


class FakeConfig:
    def __init__(self, mm_use_im_start_end: bool) -> None:
        self.mm_use_im_start_end = mm_use_im_start_end


def records(*ids):
    return {
        str(value): {
            "id": str(value),
            "image": "{}.jpg".format(value),
            "conversations": [{"from": "human", "value": "What is this?"}],
        }
        for value in ids
    }


def read_rows(path: Path) -> List[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ----------------------------------------------------------------------
# route identity
# ----------------------------------------------------------------------
def test_route_key_is_canonical_and_order_invariant():
    assert route_key([]) == "base"
    assert route_key([3]) == "e03"
    assert route_key([3, 12]) == route_key([12, 3]) == "e03_12"
    assert route_key([12, 3, 7]) == route_key([7, 12, 3]) == "e03_07_12"


def test_state_for_maps_cardinality_to_the_three_v8_states():
    assert generate._state_for([]) == STATE_BASE_ONLY
    assert generate._state_for([3]) == STATE_REUSE1
    assert generate._state_for([3, 12]) == STATE_REUSE2


def test_record_id_falls_back_to_question_id():
    assert generate._record_id({"id": "a"}) == "a"
    assert generate._record_id({"question_id": 7}) == "7"


# ----------------------------------------------------------------------
# the append-only cache
# ----------------------------------------------------------------------
def test_empty_route_cache_is_truthy():
    """Regression: defining ``__len__`` made a fresh cache falsy.

    A V8-A smoke run reported ``generated=31`` and left no
    ``generation_cache.jsonl`` behind, because the ``if self.cache:`` guard
    evaluated False on the empty cache and skipped every ``put``.  The run's
    answers existed only in memory and in the metrics derived from them.
    """
    assert bool(RouteGenerationCache(Path("never-written.jsonl"))) is True


def test_route_cache_writes_on_the_first_put_and_round_trips(tmp_path):
    path = tmp_path / "generation_cache.jsonl"
    cache = RouteGenerationCache(path)
    cache.put("7", [3, 12], "hello")
    assert path.is_file()

    reopened = RouteGenerationCache(path)
    assert reopened.get("7", [12, 3]) == "hello"
    assert len(reopened) == 1


def test_route_cache_refuses_to_rewrite_a_cached_answer(tmp_path):
    path = tmp_path / "generation_cache.jsonl"
    cache = RouteGenerationCache(path)
    cache.put("7", [3], "first")
    with pytest.raises(GenerationError):
        cache.put("7", [3], "second")
    # Identical text is a no-op, not an error: a resumed run re-derives it.
    cache.put("7", [3], "first")
    assert len(cache) == 1


# ----------------------------------------------------------------------
# generate_route: caching, grouping, per-sample selection
# ----------------------------------------------------------------------
def test_generate_route_writes_the_cache_even_when_it_starts_empty(tmp_path):
    """The end-to-end shape of the truthiness bug."""
    path = tmp_path / "generation_cache.jsonl"
    engine = StubEngine(path)
    engine.generate_route(
        {"1": [], "2": [3], "3": [3, 12]},
        records(1, 2, 3),
    )
    assert engine.generated_count == 3
    assert path.is_file()
    assert {(row["route"], row["sample_id"]) for row in read_rows(path)} == {
        ("base", "1"),
        ("e03", "2"),
        ("e03_12", "3"),
    }


def test_generate_route_resumes_from_the_cache_without_regenerating(tmp_path):
    path = tmp_path / "generation_cache.jsonl"
    selection = {"1": [], "2": [3], "3": [3, 12]}
    first = StubEngine(path)
    answers = first.generate_route(selection, records(1, 2, 3))

    second = StubEngine(path)
    resumed = second.generate_route(selection, records(1, 2, 3))
    assert second.generated_count == 0
    assert second.cache_hits == 3
    assert second.calls == []
    assert resumed == answers


def test_generate_route_uses_one_selection_per_sample(tmp_path):
    path = tmp_path / "generation_cache.jsonl"
    engine = StubEngine(path)
    engine.generate_route({"1": [3], "2": [12], "3": []}, records(1, 2, 3))
    assert sorted(engine.calls) == [("1", "e03"), ("2", "e12"), ("3", "base")]


def test_generate_route_groups_reversed_pairs_under_one_route(tmp_path):
    path = tmp_path / "generation_cache.jsonl"
    engine = StubEngine(path)
    engine.generate_route({"1": [3, 12], "2": [12, 3]}, records(1, 2))
    assert sorted(engine.calls) == [("1", "e03_12"), ("2", "e03_12")]
    assert len(read_rows(path)) == 2  # keyed by sample, not merely by route


def test_generate_route_rejects_an_unknown_sample(tmp_path):
    path = tmp_path / "generation_cache.jsonl"
    engine = StubEngine(path)
    with pytest.raises(GenerationError):
        engine.generate_route({"99": [3]}, records(1))
    assert not path.is_file()


# ----------------------------------------------------------------------
# the prompt must not drift away from the V7 evaluator
# ----------------------------------------------------------------------
def test_v8_prompt_is_byte_identical_to_the_v7_evaluator():
    """V8 numbers are compared against V7's; the prompt cannot differ."""
    from compose.eval.eval_task import _prompt as v7_prompt

    record_with_token = {
        "conversations": [{"from": "human", "value": "<image>\nWhat animal is this?"}]
    }
    record_without_token = {
        "conversations": [{"from": "human", "value": "How many cubes are left?"}]
    }
    for mm_use_im_start_end in (False, True):
        config = FakeConfig(mm_use_im_start_end)
        for record in (record_with_token, record_without_token):
            assert generate._prompt(record, config, "vicuna_v1") == v7_prompt(
                record, config, "vicuna_v1"
            )


# ----------------------------------------------------------------------
# the runner-side cache shares the contract
# ----------------------------------------------------------------------
def test_append_only_jsonl_is_truthy_when_empty_and_dedupes(tmp_path):
    from compose.experiments.v8_task_run import AppendOnlyJsonl

    path = tmp_path / "nll_cache.jsonl"
    cache = AppendOnlyJsonl(path, key_fields=("sample_id",))
    assert bool(cache) is True
    cache.put({"sample_id": "1", "nll": 1.5})
    cache.put({"sample_id": "1", "nll": 9.9})  # append-only: first write wins
    assert path.is_file()
    assert read_rows(path) == [{"sample_id": "1", "nll": 1.5}]


# ----------------------------------------------------------------------
# route_key <-> experts_from_route
# ----------------------------------------------------------------------
def test_experts_from_route_is_the_inverse_of_route_key():
    """Regression: ``"e10_13".split("_")[1:]`` is ``["13"]`` -- a single.

    A seed spot check parsed routes that way, so every seeded *pair* was
    re-measured as a single expert, and a run whose seeds were in fact correct
    aborted on the resulting mismatch.
    """
    assert experts_from_route("base") == []
    assert experts_from_route("e03") == [3]
    assert experts_from_route("e10_13") == [10, 13]
    assert experts_from_route("e00_13") == [0, 13]
    assert experts_from_route("e03_07_12") == [3, 7, 12]

    for route in ("base", "e03", "e10_13", "e00_13", "e03_07_12", "e12_15"):
        assert route_key(experts_from_route(route)) == route


def test_experts_from_route_rejects_non_canonical_routes():
    for bad in ("", "e", "x03", "e1_3", "e03_", "e_03", "e03_13_03", "pair_03_13"):
        with pytest.raises(GenerationError):
            experts_from_route(bad)


# ----------------------------------------------------------------------
# the routing scope
# ----------------------------------------------------------------------
def test_history_only_excludes_the_task_own_and_later_experts():
    """V8-A must not credit the router with experts the task has not trained.

    At task ``t`` the pool holds the experts of tasks ``< t``.  The committed
    V7 pool, however, is the pool *after* task 5, so it contains every task's
    experts -- including the ones the current task would have produced itself.
    ``--history-only`` removes them from the routing candidates.
    """
    from compose.experiments.v8_task_run import V8TaskRun

    run = V8TaskRun.__new__(V8TaskRun)
    run.task = 3
    run.expert_ids = [0, 1, 12, 13, 16, 20]

    class Pool:
        expert_records = {
            0: {"origin_task": 0}, 1: {"origin_task": 1},
            12: {"origin_task": 3}, 13: {"origin_task": 3},
            16: {"origin_task": 4}, 20: {"origin_task": 5},
        }

    run.pool = Pool()
    run.args = argparse.Namespace(history_only=False)
    assert run._excluded_experts() == []
    run.args = argparse.Namespace(history_only=True)
    assert run._excluded_experts() == [12, 13, 16, 20]
