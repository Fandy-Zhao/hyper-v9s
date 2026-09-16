"""The per-task scheduling change, and the supervisor that watches it.

Two claims are tested here, and they are the two that would be expensive to get
wrong.

**A task is finished when its own row cell exists, not when its whole row
does.**  The cross-task cells are deferred to one sweep at the end, which is a
change to *when* they are measured and must not be a change to what "task t is
done" means -- a gate that waited for the full row would hold every task behind
a sweep that only runs after the last one.

**The sweep reuses, it does not regenerate.**  ``A[t][t]`` was measured beside
the training that produced it, with the pool that training committed; a sweep
that re-measured it would spend GPU hours to re-derive a number the run already
holds.  Reuse is only safe while the artefacts it was measured from are
unchanged, so the record of those hashes is load-bearing, and is tested.

The watchdog's taxonomy is tested as a pure function of one set of readings,
which is why it was written as one: the verdicts that matter most are the ones a
running fault would be needed to reproduce otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from compose.experiments.v9_chain import (
    V9ChainError,
    final_sweep,
    matrix_cells,
    self_eval,
    self_eval_intact,
    self_eval_record,
    task_lock,
    train_phase,
)
from compose.experiments.v9_watchdog import collapse_clones, evaluate_status
from compose.v9.closure import task_completion
from compose.v9.formal_eval import V9FormalEvaluationError, plan_cells
from compose.v9.heartbeat import age as heartbeat_age
from compose.v9.heartbeat import beat, read


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class _Args:
    """The slice of the chain's namespace these functions actually read."""

    def __init__(self, root: Path, **overrides: Any) -> None:
        self.run_root = str(root)
        self.repo = str(root)
        self.python = "python"
        self.from_task = 0
        self.to_task = 5
        self.sanity = False
        self.force = False
        self.__dict__.update(overrides)


def _task(root: Path, task: int, cells: list) -> Path:
    """A task whose training side is complete and whose row holds ``cells``."""
    training = root / "task{}".format(task) / "training" / "task{}".format(task)
    training.mkdir(parents=True, exist_ok=True)
    _write(training / "v9_task_statistics.json",
           {"micro_steps": 10, "observed_sample_count": 160})
    (training / "compose_experts.bin").write_bytes(b"expert-weights")
    _write(training / "v9_full_data_coverage.json", {
        "train_sample_coverage": 1.0, "optimizer_coverage": 1.0,
        "unique_optimizer_applied_sample_ids": 160, "unclosed_window_sample_count": 0,
        "optimizer_steps": 5, "num_train_samples": 160, "global_batch": 32,
        "expected_optimizer_steps": 5,
    })
    _write(training / "v9_freeze_audit.json", {"llm": True, "vision": True})
    _write(training / "v9_answer_key_isolation.json", {"answer_reaches_key": False})
    _write(training / "v9_distributed_audit.json", {"ranks": 4})
    _write(training / "v9_trainable_parameter_audit.json", {"illegal": []})
    _write(training / "v9_contribution_calibration.json", {"samples": 256, "pearson": 0.91})
    _write(training / "v9_candidate_validation_gain.json", {"0": 0.1})

    task_root = root / "task{}".format(task)
    (task_root / "state").mkdir(parents=True, exist_ok=True)
    (task_root / "state" / "key_pool_task{}.pt".format(task)).write_bytes(b"committed-pool")
    _write(task_root / "data" / "audit_task{}.json".format(task), {
        "candidate_commit_applied": {"committed": [0, 1], "historical_ids": []},
        "task_key_applied": {"reset": []},
    })

    matrix = root / "evaluation_matrix" / "evaluation" / "continual_matrix.json"
    rows = json.loads(matrix.read_text(encoding="utf-8")).get("rows") if matrix.is_file() else {}
    rows = rows or {}
    row = rows.setdefault(str(task), {})
    for cell in cells:
        row[str(cell)] = {"task_id": cell, "value": 50.0 + cell, "metric": "Accuracy",
                          "scorer": "llava.eval.eval_deepseek_r1"}
    _write(matrix, {"schema_version": 1, "rows": rows})
    return task_root


def _selection(root: Path, stage: int, cell: int, query_hash: str = "q" * 64) -> None:
    _write(
        root / "evaluation_matrix" / "evaluation" / "selections"
        / "t{}".format(stage) / "task{}".format(cell) / "selections.json",
        {
            "method": "v9s", "top_k": 2, "answer_features_used": False,
            "task_id_used": False, "training_retrieval_cache_used": False,
            "query_encoder_calls": 0,
            "query_source": {"kind": "v7_precomputed_query_tensor", "query_value_hash": query_hash},
        },
    )


def _diagonal(root: Path, task: int) -> Dict[str, Any]:
    """Write exactly what the chain leaves behind for one finished task."""
    _selection(root, task, task)
    args = _Args(root)
    cell = matrix_cells(args, task).get(task)
    assert cell is not None, "the fixture's matrix is missing A[{}][{}]".format(task, task)
    record = self_eval_record(args, task, cell)
    _write(root / "task{}".format(task) / "evaluation" / "self_eval.json", record)
    return record


def _finished_tasks(root: Path, count: int) -> None:
    for task in range(count):
        _task(root, task, cells=[task])
        _diagonal(root, task)


# ----------------------------------------------------------------------
# the gate: the diagonal, not the row
# ----------------------------------------------------------------------
def test_the_diagonal_alone_finishes_a_task(tmp_path):
    """Task 2 with only ``A[2][2]`` is done; the rest of the row is the sweep's."""
    root = _task(tmp_path, 2, cells=[2])
    narrowed = task_completion(root, 2, eval_cells=[2],
                               eval_root=tmp_path / "evaluation_matrix")
    assert narrowed["complete"], narrowed["failed"]

    whole_row = task_completion(root, 2, eval_root=tmp_path / "evaluation_matrix")
    assert not whole_row["complete"]
    assert "eval_row" in whole_row["failed"]
    detail = next(row["detail"] for row in whole_row["requirements"] if row["name"] == "eval_row")
    assert "required cells [0, 1, 2]" in detail


def test_a_missing_diagonal_still_fails_the_task(tmp_path):
    """Narrowing the row must not turn the gate into no gate at all."""
    root = _task(tmp_path, 2, cells=[0, 1])
    narrowed = task_completion(root, 2, eval_cells=[2],
                               eval_root=tmp_path / "evaluation_matrix")
    assert not narrowed["complete"]
    assert "eval_row" in narrowed["failed"]


def test_train_phase_admits_the_next_task_on_the_diagonal(tmp_path):
    root = _task(tmp_path, 2, cells=[2])
    _diagonal(tmp_path, 2)
    args = _Args(tmp_path)
    # ``self_eval`` finds the cell already scored and records what it was
    # measured from, rather than spending the GPU hours to measure it again.
    outcome = self_eval(args, 2)
    assert outcome["action"] == "diagonal-already-scored"
    payload = train_phase(args, 2, require_diagonal=True)
    assert (root / "data" / "train_phase_complete.json").is_file()
    assert payload["diagonal_required"] is True
    assert payload["completion"]["complete"] is True


def test_train_phase_refuses_a_diagonal_that_is_not_there(tmp_path):
    root = _task(tmp_path, 3, cells=[0, 1, 2])
    with pytest.raises(V9ChainError) as error:
        train_phase(_Args(tmp_path), 3, require_diagonal=True)
    assert "not ready to hand off" in str(error.value)
    assert not (root / "data" / "train_phase_complete.json").is_file()


def test_train_phase_refuses_a_cell_with_no_record_of_what_it_measured(tmp_path):
    """The cell exists, but nothing says what produced it -- so it is not evidence."""
    _task(tmp_path, 1, cells=[1])
    with pytest.raises(V9ChainError) as error:
        train_phase(_Args(tmp_path), 1, require_diagonal=True)
    assert "records what it was measured from" in str(error.value)


# ----------------------------------------------------------------------
# the diagonal's provenance
# ----------------------------------------------------------------------
def test_the_record_names_what_the_cell_was_measured_from(tmp_path):
    root = _task(tmp_path, 1, cells=[1])
    _selection(tmp_path, 1, 1)
    record = self_eval_record(_Args(tmp_path), 1, {"task_id": 1, "value": 51.0,
                                                   "metric": "Accuracy"})
    assert record["task_index"] == 1
    assert record["score"] == 51.0
    assert record["query_hash"] == "q" * 64
    assert record["task_id_used"] is False
    assert record["answer_features_used"] is False
    assert record["training_retrieval_cache_used"] is False
    assert record["query_encoder_calls"] == 0
    assert record["cardinality_scale"] == "none"
    assert record["expert_pool_hash"] is not None
    assert record["committed_key_state_sha256"] is not None


def test_the_record_refuses_a_cell_with_no_query_provenance(tmp_path):
    _task(tmp_path, 1, cells=[1])
    with pytest.raises(V9ChainError) as error:
        self_eval_record(_Args(tmp_path), 1, {"task_id": 1, "value": 51.0})
    assert "formal number" in str(error.value)


def test_the_record_notices_a_pool_that_changed_underneath_it(tmp_path):
    root = _task(tmp_path, 1, cells=[1])
    args = _Args(tmp_path)
    record = _diagonal(tmp_path, 1)
    assert self_eval_intact(args, 1, record)["ok"]

    (root / "state" / "key_pool_task1.pt").write_bytes(b"a different pool")
    intact = self_eval_intact(args, 1, record)
    assert not intact["ok"]
    assert intact["checks"]["committed_key_state"] is False
    assert intact["checks"]["expert_pool"] is True


def test_a_replaced_pool_is_never_quietly_re_measured(tmp_path):
    """Regenerating the cell would hide whatever replaced the pool."""
    _task(tmp_path, 1, cells=[1])
    args = _Args(tmp_path)
    record = _diagonal(tmp_path, 1)
    assert record["committed_key_state_sha256"]
    (tmp_path / "task1" / "state" / "key_pool_task1.pt").write_bytes(b"a different pool")

    with pytest.raises(V9ChainError) as error:
        self_eval(args, 1)
    assert "must not be silently" in str(error.value)


def test_a_legacy_cell_is_recorded_rather_than_regenerated(tmp_path):
    """A cell whose pass predates the record is kept; only its provenance is added."""
    _task(tmp_path, 1, cells=[1])
    _selection(tmp_path, 1, 1)
    args = _Args(tmp_path)
    outcome = self_eval(args, 1)
    assert outcome["action"] == "diagonal-recorded-from-existing"
    assert (tmp_path / "task1" / "evaluation" / "self_eval.json").is_file()


# ----------------------------------------------------------------------
# the sweep
# ----------------------------------------------------------------------
def test_the_sweep_asks_only_for_the_cells_that_are_missing(tmp_path, monkeypatch):
    """The diagonal is reused; only the cross-task cells are generated."""
    _finished_tasks(tmp_path, 6)
    asked = []

    def fake_evaluate(args, stage, cells):
        asked.append((stage, list(cells)))
        matrix = tmp_path / "evaluation_matrix" / "evaluation" / "continual_matrix.json"
        rows = json.loads(matrix.read_text(encoding="utf-8"))["rows"]
        for cell in cells:
            rows[str(stage)][str(cell)] = {"task_id": cell, "value": 1.0}
        _write(matrix, {"schema_version": 1, "rows": rows})
        return {"matrix": str(matrix), "log": "log", "stage": stage, "cells": cells}

    monkeypatch.setattr("compose.experiments.v9_chain.evaluate_task", fake_evaluate)
    sweep = final_sweep(_Args(tmp_path))

    # 21 cells in the lower triangle; 6 were measured beside their own training.
    assert sum(len(cells) for _, cells in asked) == 15
    assert [stage for stage, _ in asked] == [1, 2, 3, 4, 5]
    assert all(stage not in cells for stage, cells in asked), "a diagonal was regenerated"
    assert len(sweep["reused_diagonal"]) == 6
    assert {entry["stage"] for entry in sweep["filled"]} == {1, 2, 3, 4, 5}


def test_a_re_run_sweep_fills_nothing(tmp_path, monkeypatch):
    """The sweep is idempotent: a second pass over a full matrix does no work."""
    _finished_tasks(tmp_path, 3)

    def fake_evaluate(args, stage, cells):
        matrix = tmp_path / "evaluation_matrix" / "evaluation" / "continual_matrix.json"
        rows = json.loads(matrix.read_text(encoding="utf-8"))["rows"]
        for cell in cells:
            rows[str(stage)][str(cell)] = {"task_id": cell, "value": 1.0}
        _write(matrix, {"schema_version": 1, "rows": rows})
        return {"matrix": str(matrix), "log": "log", "stage": stage, "cells": cells}

    monkeypatch.setattr("compose.experiments.v9_chain.evaluate_task", fake_evaluate)
    final_sweep(_Args(tmp_path, to_task=2))
    second = final_sweep(_Args(tmp_path, to_task=2))
    assert second["filled"] == []
    assert len(second["reused_diagonal"]) == 3


def test_the_sweep_refuses_to_build_on_a_diagonal_that_moved(tmp_path, monkeypatch):
    _finished_tasks(tmp_path, 3)
    (tmp_path / "task1" / "state" / "key_pool_task1.pt").write_bytes(b"replaced after the fact")

    def explode(*args, **kwargs):
        raise AssertionError("the sweep must not evaluate over a stale diagonal")

    monkeypatch.setattr("compose.experiments.v9_chain.evaluate_task", explode)
    with pytest.raises(V9ChainError) as error:
        final_sweep(_Args(tmp_path, to_task=2))
    assert "since changed" in str(error.value)


def test_the_sweep_refuses_to_certify_an_unrecorded_diagonal(tmp_path, monkeypatch):
    _task(tmp_path, 0, cells=[0])
    monkeypatch.setattr("compose.experiments.v9_chain.evaluate_task",
                        lambda *a, **k: pytest.fail("nothing may be evaluated here"))
    with pytest.raises(V9ChainError) as error:
        final_sweep(_Args(tmp_path, to_task=0))
    assert "cannot be certified as reused" in str(error.value)


def test_the_sweep_refuses_to_run_before_the_last_task(tmp_path):
    """A stage with no committed pool has no row to complete."""
    _task(tmp_path, 1, cells=[1])
    with pytest.raises(V9ChainError) as error:
        final_sweep(_Args(tmp_path, to_task=1))
    assert "has not committed a pool" in str(error.value)


def test_a_cell_still_missing_after_a_successful_evaluation_is_an_error(tmp_path, monkeypatch):
    """An evaluation that exits zero and writes nothing must not read as filled."""
    _finished_tasks(tmp_path, 2)
    monkeypatch.setattr(
        "compose.experiments.v9_chain.evaluate_task",
        lambda args, stage, cells: {"matrix": "m", "log": "l", "stage": stage, "cells": cells},
    )
    with pytest.raises(V9ChainError) as error:
        final_sweep(_Args(tmp_path, to_task=1))
    assert "still missing" in str(error.value)


# ----------------------------------------------------------------------
# the evaluator's cell plan
# ----------------------------------------------------------------------
def _cells(count: int):
    return [{"task_index": index, "question_file": "/q.json", "query_cache": "/c.json"}
            for index in range(count)]


def test_a_plan_may_hold_a_subset_of_the_row(tmp_path):
    plan = plan_cells(tmp_path, 3, [_cells(4)[2]], "/keys.pt")
    assert [item["task_index"] for item in plan] == [2]


def test_a_plan_may_not_reach_past_the_stage(tmp_path):
    """``A[2][3]`` is not a measurement that exists: stage 2 never saw task 3."""
    with pytest.raises(V9FormalEvaluationError) as error:
        plan_cells(tmp_path, 2, [_cells(4)[3]], "/keys.pt")
    assert "has not been trained on tasks [3]" in str(error.value)


def test_a_plan_with_no_cells_is_refused(tmp_path):
    with pytest.raises(V9FormalEvaluationError):
        plan_cells(tmp_path, 2, [], "/keys.pt")


def test_a_plan_refuses_to_score_one_cell_twice(tmp_path):
    with pytest.raises(V9FormalEvaluationError) as error:
        plan_cells(tmp_path, 2, [_cells(4)[1], _cells(4)[1]], "/keys.pt")
    assert "repeats a cell" in str(error.value)


# ----------------------------------------------------------------------
# the watchdog's taxonomy
# ----------------------------------------------------------------------
def _state(**overrides: Any) -> Dict[str, Any]:
    base = {
        "non_finite": [], "backbone_invariant_broken": False,
        "backbone_ratios": {"backbone_forwards_per_micro_step": 1.0}, "query_encoder_calls": 0,
        "optimizer_coverage": None, "two_supervisors": False, "supervisors": 1,
        "git_sha_mismatch": False, "disk_free_gb": 500.0, "heartbeat_age": None,
        "metric_age": 5.0, "hard_stall": False, "idle_cycles": 0, "ranks": 4,
        "recent_errors": [], "chain_dead": False, "run_finished": False,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("field,value,needle", [
    ("non_finite", ["loss_total"], "non-finite"),
    ("backbone_invariant_broken", True, "per micro-step"),
    ("query_encoder_calls", 3, "encodes queries live"),
    ("optimizer_coverage", 0.97, "optimizer coverage"),
    ("two_supervisors", True, "v9_chain processes"),
    ("git_sha_mismatch", True, "the checkout is at"),
])
def test_a_correctness_fault_is_fatal_and_never_recoverable(field, value, needle):
    verdict = evaluate_status(_state(**{field: value}))
    assert verdict["verdict"] == "FATAL_CORRECTNESS"
    assert any(needle in item for item in verdict["fatal"])
    assert verdict["recoverable"] == []


def test_a_dead_supervisor_with_live_training_is_not_restarted():
    """The one recovery the watchdog must refuse: starting a second training."""
    verdict = evaluate_status(_state(chain_dead=True, ranks=4))
    assert verdict["verdict"] == "WARNING"
    assert verdict["recoverable"] == []


def test_a_dead_supervisor_with_nothing_running_is_recoverable():
    verdict = evaluate_status(_state(chain_dead=True, ranks=0))
    assert verdict["verdict"] == "RECOVERABLE"
    assert verdict["recoverable"] == ["restart_chain"]


def test_a_finished_run_is_left_alone():
    verdict = evaluate_status(_state(chain_dead=True, ranks=0, run_finished=True))
    assert verdict["recoverable"] == []


def test_a_stale_metric_with_a_fresh_heartbeat_is_not_a_stall():
    """Calibration writes no step metrics; the beat is what says it is alive."""
    verdict = evaluate_status(_state(metric_age=1500.0, heartbeat_age=60.0))
    assert not any(finding["kind"] == "metrics" for finding in verdict["findings"])


def test_a_stale_metric_with_a_stale_heartbeat_is_a_suspected_stall():
    verdict = evaluate_status(_state(metric_age=1000.0, heartbeat_age=None))
    assert any(finding["level"] == "SUSPECTED_STALL" for finding in verdict["findings"])
    assert verdict["verdict"] == "SUSPECTED_STALL"


def test_a_long_stall_escalates_to_a_diagnosis_and_still_not_a_kill():
    verdict = evaluate_status(_state(metric_age=1500.0, heartbeat_age=None))
    assert any(finding["level"] == "DIAGNOSE" for finding in verdict["findings"])
    assert verdict["recoverable"] == []


def test_five_idle_cycles_are_a_hard_stall():
    verdict = evaluate_status(_state(hard_stall=True, idle_cycles=5, metric_age=5.0))
    assert verdict["verdict"] == "HARD_STALL"


def test_a_historical_traceback_is_not_a_verdict():
    """A traceback a later restart fixed is still in the log; only its age matters."""
    verdict = evaluate_status(_state(recent_errors=[{"pattern": "RuntimeError",
                                                     "age_seconds": 900.0}]))
    assert verdict["verdict"] == "WARNING"
    assert verdict["fatal"] == []


def test_a_recent_traceback_still_does_not_stop_the_run():
    """Only the correctness taxonomy is fatal; a traceback is a report."""
    verdict = evaluate_status(_state(recent_errors=[{"pattern": "CUDA out of memory",
                                                     "age_seconds": 5.0}]))
    assert verdict["fatal"] == []
    assert verdict["recoverable"] == []


def test_disk_crosses_two_lines():
    assert evaluate_status(_state(disk_free_gb=40.0))["verdict"] == "WARNING"
    pause = evaluate_status(_state(disk_free_gb=10.0))
    assert pause["verdict"] == "PAUSE"
    assert evaluate_status(_state(disk_free_gb=100.0))["verdict"] == "HEALTHY"


def test_a_fatal_fault_outranks_a_recoverable_one():
    verdict = evaluate_status(_state(chain_dead=True, ranks=0, query_encoder_calls=1))
    assert verdict["verdict"] == "FATAL_CORRECTNESS"


# ----------------------------------------------------------------------
# process matching: the rule that was wrong once
# ----------------------------------------------------------------------
def test_fork_clones_collapse_into_the_process_that_was_exec_d():
    """Torchrun's ranks differ from their parent's argv; their workers do not.

    The first version of this rule dropped every candidate whose parent was also
    a candidate, which discarded torchrun's four ranks -- their parent's command
    line names the module it launched, so they looked like children of
    themselves.  Only an identical argv marks a clone, and a DataLoader worker
    is the only thing here that has one.
    """

    def rank(index: int):
        return ["python", "-m", "compose.train.train_compose", "--local-rank={}".format(index)]

    torchrun = ["python", "-m", "torch.distributed.run", "--nproc_per_node=4",
                "-m", "compose.train.train_compose"]
    candidates = {
        100: (1, torchrun),      # torchrun: exec'd from a shell that is not a candidate
        101: (100, rank(0)),     # rank 0: exec'd by torchrun, argv of its own
        102: (100, rank(1)),     # rank 1, likewise
        103: (101, rank(0)),     # a DataLoader worker forked by rank 0
        104: (101, rank(0)),     # ... and its siblings, each a byte-for-byte copy
        105: (101, rank(0)),
        999: (1, ["bash", "-c", "v9_chain --run-root /r"]),
    }
    assert collapse_clones(candidates) == [100, 101, 102, 999]


def test_one_rank_is_not_a_clone_of_another():
    """Siblings share an argv and are not copies: only a *parent* match is."""
    shared = ["python", "-m", "train_compose"]
    assert collapse_clones({1: (0, shared), 2: (0, shared)}) == [1, 2]


# ----------------------------------------------------------------------
# the locks and the heartbeat
# ----------------------------------------------------------------------
def test_a_second_holder_of_one_task_lock_is_refused(tmp_path):
    with task_lock(tmp_path, 3):
        with pytest.raises(V9ChainError) as error:
            with task_lock(tmp_path, 3):
                pass
        assert "another process holds" in str(error.value)
    # Released by the context manager, so the next task takes it cleanly.
    with task_lock(tmp_path, 3):
        assert (tmp_path / "locks" / "task3.lock").is_file()


def test_two_tasks_do_not_block_each_other(tmp_path):
    with task_lock(tmp_path, 0):
        with task_lock(tmp_path, 1):
            assert True


def test_no_beat_and_a_beat_just_now_are_different_answers(tmp_path):
    """``None`` is "never", not "zero seconds ago"."""
    assert heartbeat_age(tmp_path) is None
    assert read(tmp_path) is None

    beat(tmp_path, "calibration", 2, samples=256)
    age = heartbeat_age(tmp_path)
    assert age is not None and age < 5.0
    payload = read(tmp_path)
    assert payload["stage"] == "calibration"
    assert payload["task"] == 2
    assert payload["samples"] == 256


def test_a_torn_heartbeat_reads_as_never_beaten(tmp_path):
    """A half-written beat must not read as an infinitely old timestamp."""
    path = tmp_path / "status" / "heartbeat.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"stage": "calib', encoding="utf-8")
    assert read(tmp_path) is None
    assert heartbeat_age(tmp_path) is None


def test_the_beat_is_atomic(tmp_path):
    """The reader sees the whole payload or none of it, never a partial one."""
    beat(tmp_path, "self_eval", 1)
    leftovers = list((tmp_path / "status").glob("*.tmp"))
    assert leftovers == []
    assert set(read(tmp_path)) >= {"stage", "task", "timestamp", "iso", "pid"}
