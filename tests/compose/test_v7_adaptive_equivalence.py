"""Adaptive GPU execution equivalence (spec 0903, CPU-only tests).

The GPU-count-adaptive execution must be a pure *scheduling* change:
parallel worker execution must not alter any V7 method output.  These tests
pin that down where it can be verified without GPUs:

- the parallel pruning scorer (Deferred results over a job pool, spec S5)
  produces exactly the serial trajectory / metrics / audit of the legacy
  synchronous scorer;
- query shard payloads merge into exactly the single-GPU payload;
- evaluation chunk sizes reproduce eval_task's contiguous chunking.
"""

import concurrent.futures
import json

import pytest
import torch

from compose.eval.query_features import (
    assemble_query_payload,
    merge_query_shard_payloads,
    shard_expected_ids,
)
from compose.eval.sharding import partial_path
from compose.v7.config import V7PruningConfig
from compose.v7.pool import V7ExpertKeyPool
from compose.v7.pruning import CandidatePruner
from compose.v7.workers import deep_resolve, deferred_from


def basis(index):
    value = torch.zeros(1536)
    value[index] = 1.0
    return value


def pool_with(hist=(), current=()):
    pool = V7ExpertKeyPool()
    for expert_id, axis in hist:
        pool.add(expert_id, basis(axis), axis, "historical", False)
    for expert_id, axis in current:
        pool.add(expert_id, basis(axis), 9, "current", True)
    return pool


def score_rows(rows):
    """Deterministic score function over the rerouted ``[N,2]`` id rows."""
    routed = torch.tensor(rows)
    useful = routed.eq(7).any(dim=1) | routed.eq(8).any(dim=1)
    metric = float(useful.float().mean())
    return {
        "metric": metric,
        "loss": 1.0 - metric,
        "official_metric": {"name": "unit", "value": metric},
        "answer_nll": 1.0 - metric,
        "metric_fallback": False,
    }


class AsyncScorerPool:
    """Model of the adaptive scorer: one capped worker pool, Deferred results.

    Submission is non-blocking exactly like v7_task_run's adaptive scorer;
    resolution happens only when the pruner consumes a float.  The pool is
    capped (``max_workers``) to model the GPU cap, so a whole remove-and-
    reroute iteration runs concurrently only when the candidate count fits.
    """

    def __init__(self, max_workers):
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def make_scorer(self, call_box):
        def scorer(routes):
            call_box["value"] += 1
            future = self.pool.submit(score_rows, routes.detach().cpu().tolist())
            return {
                "metric": deferred_from(future, "metric"),
                "loss": deferred_from(future, "loss"),
                "official_metric": deferred_from(future, "official_metric"),
                "answer_nll": deferred_from(future, "answer_nll"),
                "metric_fallback": False,
            }

        return scorer

    def shutdown(self):
        self.pool.shutdown(wait=True)


def test_parallel_pruning_scorer_matches_serial_trajectory():
    queries = torch.nn.functional.normalize((basis(0) + basis(1)).unsqueeze(0), dim=-1)
    config = V7PruningConfig()

    serial_call_box = {"value": 0}

    def serial_scorer(routes):
        serial_call_box["value"] += 1
        return score_rows(routes.detach().cpu().tolist())

    serial_pool = pool_with(
        hist=((1, 4), (2, 5)), current=((7, 0), (8, 1), (9, 2), (10, 3))
    )
    retained, metrics, audit = CandidatePruner(serial_pool, config).evaluate(
        queries, queries, basis(0), serial_scorer
    )

    async_runner = AsyncScorerPool(max_workers=2)
    parallel_call_box = {"value": 0}
    parallel_pool = pool_with(
        hist=((1, 4), (2, 5)), current=((7, 0), (8, 1), (9, 2), (10, 3))
    )
    pretained, pmetrics, paudit = CandidatePruner(parallel_pool, config).evaluate(
        queries, queries, basis(0), async_runner.make_scorer(parallel_call_box)
    )
    async_runner.shutdown()
    pmetrics = deep_resolve(pmetrics)
    paudit = deep_resolve(paudit)

    assert tuple(pretained) == tuple(retained)
    assert pmetrics == metrics
    assert paudit == audit
    # Identical scorer call count: every hypothesis of an iteration is
    # submitted before any is consumed, so the number and order of scored
    # evaluations (job index contract) cannot drift.
    assert parallel_call_box["value"] == serial_call_box["value"]


def test_deferred_deep_resolve_matches_plain_values():
    future = concurrent.futures.Future()
    future.set_result({
        "x": {"a": {"b": [1.0, {"c": 2}]}},
        "y": {"a": {"b": [1.0, {"c": 2}]}},
        "metric": 0.25,
    })
    resolved = deep_resolve({
        "x": deferred_from(future, "x"),
        "plain": 3,
        "nested": {"y": [deferred_from(future, "y")]},
        "metric": deferred_from(future, "metric"),
    })
    assert resolved == {
        "x": {"a": {"b": [1.0, {"c": 2}]}},
        "plain": 3,
        "nested": {"y": [{"a": {"b": [1.0, {"c": 2}]}}]},
        "metric": 0.25,
    }


def test_query_shard_merge_matches_single_gpu_payload(tmp_path):
    records = []
    for index in range(7):
        records.append({
            "id": "s{}".format(index),
            "visual_feature": [float(index)] * 3,
            "text_feature": [float(index + 1)] * 3,
            "query": [float(index + 2)] * 4,
        })
    backbone = {"backbone_hash": "abc"}
    provenance = {"module": "fixed", "params": 0}
    encoder_hash = "enc1"
    full = assemble_query_payload(
        {str(value["id"]): value for value in records},
        backbone, provenance, encoder_hash, "v7_fixed",
    )
    # Shard payloads: emulate per-worker partial writes (each worker keeps
    # only its own contiguous record slice).
    shards = 3
    for shard_index in range(shards):
        ids = shard_expected_ids(records, shards, shard_index)
        shard_records = {
            value: next(r for r in records if str(r["id"]) == value)
            for value in ids
        }
        payload = assemble_query_payload(
            shard_records, backbone, provenance, encoder_hash, "v7_fixed"
        )
        payload_path = tmp_path / partial_path("shard.json", shard_index)
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "merged.json"
    count = merge_query_shard_payloads(
        [str(tmp_path / partial_path("shard.json", i)) for i in range(shards)],
        str(output),
        expected_ids=[str(value["id"]) for value in records],
    )
    assert count == len(records)
    assert json.loads(output.read_text(encoding="utf-8")) == full
    # Header disagreement must fail closed.
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(assemble_query_payload(
        {"s0": records[0]}, {"backbone_hash": "other"}, provenance,
        encoder_hash, "v7_fixed",
    )), encoding="utf-8")
    with pytest.raises(ValueError, match="headers disagree"):
        merge_query_shard_payloads([str(bad), str(output)], str(tmp_path / "x.json"))


def test_eval_chunk_sizes_match_eval_task_contiguous_chunking():
    from compose.experiments.v7_task_run import _shard_slice_lengths

    sizes = _shard_slice_lengths(7, 3)
    assert sizes == [3, 3, 1]
    assert sum(sizes) == 7
    # eval_task._chunk uses ceil(total/count) with contiguous slices; the
    # per-worker expected line counts must cover the whole file exactly.
    import math

    total, count = 10, 3
    chunk = math.ceil(total / count)
    observed = [min(chunk, total - i * chunk) for i in range(count)]
    assert _shard_slice_lengths(total, count) == observed
    assert sum(_shard_slice_lengths(total, count)) == total
