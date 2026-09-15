"""Acceptance checks for the formal chain's gates (spec §27, §30, §31, §34, §35).

These run on CPU in seconds and need no checkpoint.  They cover the decisions the
unattended run makes about itself, which is the class of code where a mistake is
most expensive: nobody is watching when it is taken, and the failure mode is not
a crash but a run that reports success over work it did not do.

* :func:`task_completion` must read an artefact's content, not its existence --
  the predicate it replaced tested two file paths, so a truncated checkpoint and
  a never-evaluated row both passed.
* :func:`merge_shards` must re-emit answers in record order, because the caption
  scorer maps answer line *i* to COCO ``image_id = i+1``; a merge that reorders
  scores the right captions against the wrong images without failing.
* :func:`performance_report` must warn without stopping, and must catch the
  one-backbone-forward invariant being violated -- the method's central cost
  claim, and the one a per-expert enumeration would break silently.
* :func:`go_no_go` must refuse a launch whose factors do not multiply to 32.

Run with ``python -m pytest tests/compose/test_v9_formal_closure.py``.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from compose.experiments.v9_chain import (
    V9ChainError,
    _clear_markers,
    _git_dirty,
    performance_report,
)
from compose.v9.closure import go_no_go, task_completion
from compose.v9.data_wait import DataWaitError, discover, measure, report, verdict
from compose.v9.formal_eval import (
    TASK_NAMES,
    V9FormalEvaluationError,
    build_cells,
    main as formal_eval_main,
    merge_shards,
    plan_cells,
)


# ----------------------------------------------------------------------
# fixtures: a task root holding every artefact the predicate reads
# ----------------------------------------------------------------------
def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _complete_task(root: Path, task: int = 0, evaluation_root: Path | None = None) -> Path:
    """A task root that satisfies every requirement, so tests can break one."""
    training = root / "training" / "task{}".format(task)
    training.mkdir(parents=True, exist_ok=True)
    _write(training / "v9_task_statistics.json",
           {"micro_steps": 10, "observed_sample_count": 160})
    (training / "compose_experts.bin").write_bytes(b"\x00" * 16)
    _write(training / "v9_full_data_coverage.json", {"train_sample_coverage": 1.0})
    _write(training / "v9_freeze_audit.json", {"llm": True, "vision": True, "historical_lora": True})
    _write(training / "v9_answer_key_isolation.json", {"answer_reaches_key": False})
    _write(training / "v9_distributed_audit.json", {"ranks": 4})
    _write(training / "v9_trainable_parameter_audit.json", {"illegal": []})
    _write(training / "v9_contribution_calibration.json", {"samples": 256, "pearson": 0.91})
    _write(training / "v9_candidate_validation_gain.json", {"0": 0.1, "1": 0.2})
    (root / "state").mkdir(parents=True, exist_ok=True)
    (root / "state" / "key_pool_task{}.pt".format(task)).write_bytes(b"\x00" * 8)
    _write(root / "data" / "audit_task{}.json".format(task), {
        "candidate_commit_applied": {"committed": [0, 1], "historical_ids": []},
        "task_key_applied": {"reset": []},
    })
    evaluation_root = evaluation_root or root
    _write(evaluation_root / "evaluation" / "continual_matrix.json", {
        "rows": {str(task): {str(cell): {"value": 50.0} for cell in range(task + 1)}}
    })
    return root


# ----------------------------------------------------------------------
# the completion predicate
# ----------------------------------------------------------------------
def test_a_complete_task_passes_every_requirement(tmp_path):
    result = task_completion(_complete_task(tmp_path), 0)
    assert result["complete"], result["failed"]
    assert {row["name"] for row in result["requirements"]} >= {
        "training", "full_coverage", "key_commit", "freeze_audit", "calibration", "eval_row",
    }


def test_a_truncated_checkpoint_does_not_pass_as_training(tmp_path):
    """The failure the old two-file test could not see.

    ``state/key_pool_taskN.pt`` existed and ``v9_task_statistics.json`` existed,
    so the task was skipped forever -- including when the statistics said the
    training had run zero micro-steps.
    """
    root = _complete_task(tmp_path)
    _write(root / "training" / "task0" / "v9_task_statistics.json",
           {"micro_steps": 0, "observed_sample_count": 0})
    result = task_completion(root, 0)
    assert not result["complete"]
    assert "training" in result["failed"]


def test_a_missing_evaluation_row_fails_the_task(tmp_path):
    """Training and the commit can be finished while the row was never produced.

    The row lives in a shared evaluation root, not in the task's own, so this
    also checks the two roots are read as two: a predicate that looked only
    under ``root`` would pass every task of a chain that never evaluated one.
    """
    matrix = tmp_path / "matrix"
    root = _complete_task(tmp_path, evaluation_root=matrix)
    (matrix / "evaluation" / "continual_matrix.json").unlink()
    assert not task_completion(root, 0, eval_root=matrix)["complete"]
    assert "eval_row" in task_completion(root, 0, eval_root=matrix)["failed"]


def test_a_row_missing_a_cell_fails_the_task(tmp_path):
    """Row ``A[t][0..t]`` must be the whole row, not however much was scored."""
    root = _complete_task(tmp_path)
    _write(root / "evaluation" / "continual_matrix.json",
           {"rows": {"2": {"0": {"value": 1.0}, "1": {"value": 2.0}}}})
    result = task_completion(root, 2)
    assert not result["complete"]
    assert "eval_row" in result["failed"]


def test_an_unscored_cell_fails_the_task(tmp_path):
    root = _complete_task(tmp_path)
    _write(root / "evaluation" / "continual_matrix.json",
           {"rows": {"0": {"0": {"value": None}}}})
    result = task_completion(root, 0)
    assert not result["complete"]
    assert "eval_row" in result["failed"]


def test_a_commit_that_leaves_nothing_deployable_fails_the_task(tmp_path):
    root = _complete_task(tmp_path)
    _write(root / "data" / "audit_task0.json", {
        "candidate_commit_applied": {"committed": [], "historical_ids": []},
        "task_key_applied": {"reset": []},
    })
    result = task_completion(root, 0)
    assert not result["complete"]
    assert "key_commit" in result["failed"]


def test_a_frozen_parameter_that_moved_fails_the_task(tmp_path):
    root = _complete_task(tmp_path)
    _write(root / "training" / "task0" / "v9_freeze_audit.json",
           {"llm": True, "historical_lora": False})
    result = task_completion(root, 0)
    assert not result["complete"]
    assert "freeze_audit" in result["failed"]


def test_incomplete_coverage_fails_the_task_only_when_required(tmp_path):
    root = _complete_task(tmp_path)
    _write(root / "training" / "task0" / "v9_full_data_coverage.json",
           {"train_sample_coverage": 0.5})
    assert not task_completion(root, 0, require_full_coverage=True)["complete"]
    # A capped sanity run streams less than one epoch, so the same root is
    # consistent when the gate is not asked for.
    assert task_completion(root, 0, require_full_coverage=False)["complete"]


def test_a_missing_artefact_names_itself(tmp_path):
    root = _complete_task(tmp_path)
    (root / "training" / "task0" / "v9_contribution_calibration.json").unlink()
    result = task_completion(root, 0)
    assert not result["complete"]
    assert "calibration" in result["failed"]
    detail = next(row for row in result["requirements"] if row["name"] == "calibration")
    assert "missing" in detail["detail"]


# ----------------------------------------------------------------------
# the launch gate
# ----------------------------------------------------------------------
def test_go_no_go_recomputes_the_global_batch():
    assert go_no_go(None, world_size=4, per_device_batch=4, grad_accum=2)["decision"] == "GO"
    no_go = go_no_go(None, world_size=4, per_device_batch=2, grad_accum=2)
    assert no_go["decision"] == "NO-GO"
    assert "global_batch" in no_go["blocked_by"]


def test_go_no_go_refuses_a_run_that_would_encode_queries():
    assert go_no_go(None, query_encoder_calls=1)["decision"] == "NO-GO"
    assert "query_source" in go_no_go(None, query_encoder_calls=1)["blocked_by"]


def test_go_no_go_reads_the_preflights_backbone_ratio(tmp_path):
    """The invariant is read from the last row, not summed.

    Each row carries a cumulative ratio, so summing them would multiply the
    ratio by the number of rows and a perfectly healthy run would read as a
    violation.
    """
    root = _complete_task(tmp_path)
    metrics = tmp_path / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    rows = [
        {"step": step, "model_forwards_per_micro_step": 1.0,
         "backbone_forwards_per_micro_step": 1.0, "wide_model_forwards_per_micro_step": 1.0}
        for step in range(20)
    ]
    (metrics / "task0_train_steps.rank0.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    assert go_no_go(root)["decision"] == "GO"

    rows[-1]["model_forwards_per_micro_step"] = 7.0
    (metrics / "task0_train_steps.rank0.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    decision = go_no_go(root)
    assert decision["decision"] == "NO-GO"
    assert "backbone_invariant_task0" in decision["blocked_by"]


def test_go_no_go_takes_the_worst_rank(tmp_path):
    """One rank enumerating experts is a violation even if the others are clean."""
    root = _complete_task(tmp_path)
    metrics = tmp_path / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    for rank, ratio in ((0, 1.0), (1, 6.0)):
        (metrics / "task0_train_steps.rank{}.jsonl".format(rank)).write_text(
            json.dumps({"step": 0, "model_forwards_per_micro_step": ratio}) + "\n",
            encoding="utf-8",
        )
    assert go_no_go(root)["decision"] == "NO-GO"


# ----------------------------------------------------------------------
# the shard merge
# ----------------------------------------------------------------------
# §33: what may train, grouped and named
# ----------------------------------------------------------------------
class _Param:
    def __init__(self, requires_grad: bool, numel: int) -> None:
        self.requires_grad = requires_grad
        self._numel = numel

    def numel(self) -> int:
        return self._numel


class _Pool:
    """The slice of the key pool the grouping reads."""

    def __init__(self, current_ids, historical_ids, records) -> None:
        self.current_ids = list(current_ids)
        self.historical_ids = list(historical_ids)
        self.key_records = records
        self.keys = {key_id: _Param(True, 1536) for key_id in records}

    def origin_key_id(self, expert_id):
        return "e{}_t0_origin".format(int(expert_id))

    def trainable_key_ids(self):
        return sorted(
            key_id
            for key_id, record in self.key_records.items()
            if record.get("trainable") and self.keys[key_id].requires_grad
        )


class _GroupingHarness:
    """The trainer's classifier, over a model and pool small enough to reason about."""

    from compose.v9.trainer import V9ComposeTrainer as _Trainer

    _TRAINABLE_GROUP_LABELS = _Trainer._TRAINABLE_GROUP_LABELS
    # ``staticmethod`` because reading one off the class yields the plain
    # function, which would rebind as an ordinary method here.
    _expert_id_from_name = staticmethod(_Trainer._expert_id_from_name)
    _forbidden_home = staticmethod(_Trainer._forbidden_home)
    _candidate_key_ids = _Trainer._candidate_key_ids
    trainable_parameter_groups = _Trainer.trainable_parameter_groups

    def __init__(self, params, current_ids=(0, 1), records=None, task_index=1):
        self.model = types.SimpleNamespace(named_parameters=lambda: list(params.items()))
        self.v9_key_pool = _Pool(current_ids, [9], records or {})
        self.v9_task_index = task_index


def _pool_records(**overrides):
    records = {
        "e0_t0_origin": {"key_type": "origin", "task_id": 0, "trainable": True},
        "e1_t0_origin": {"key_type": "origin", "task_id": 0, "trainable": True},
    }
    records.update(overrides)
    return records


def test_the_four_legal_groups_are_the_only_ones_that_classify(tmp_path):
    params = {
        "model.layers.0.self_attn.q_proj.experts.0.lora_A.weight": _Param(True, 10),
        "model.layers.0.self_attn.q_proj.experts.1.lora_B.weight": _Param(True, 20),
        "model.layers.0.self_attn.q_proj.base_layer.weight": _Param(False, 999),
        "model.embed_tokens.weight": _Param(False, 999),
        "model.v9_router.gate.weight": _Param(True, 5),
    }
    harness = _GroupingHarness(params, records=_pool_records())
    groups, illegal = harness.trainable_parameter_groups()
    assert illegal == []
    assert len(groups["current_candidate_lora"]) == 2
    assert groups["router"] == [("model.v9_router.gate.weight", 5)]
    assert len(groups["current_candidate_key"]) == 2
    assert groups["current_task_historical_key"] == []


def test_a_historical_experts_lora_is_illegal_even_though_it_is_an_expert():
    """The flat ``.experts.`` name test cannot tell the two apart; this can."""
    params = {"model.layers.3.self_attn.v_proj.experts.9.lora_A.weight": _Param(True, 7)}
    groups, illegal = _GroupingHarness(params).trainable_parameter_groups()
    assert groups["current_candidate_lora"] == []
    assert illegal == [("model.layers.3.self_attn.v_proj.experts.9.lora_A.weight", "historical LoRA")]


@pytest.mark.parametrize(
    "name,home",
    [
        ("model.embed_tokens.weight", "token embedding"),
        ("vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight", "vision tower"),
        ("model.mm_projector.0.weight", "projector"),
        ("query_encoder.visual_projection.weight", "query encoder"),
        ("model.layers.7.self_attn.q_proj.weight", "LLM backbone"),
    ],
)
def test_every_frozen_home_is_named_when_it_leaks_into_training(name, home):
    groups, illegal = _GroupingHarness({name: _Param(True, 3)}).trainable_parameter_groups()
    assert illegal == [(name, home)]


def test_a_retained_key_of_an_earlier_task_is_illegal():
    """An origin key of a committed expert must not be trainable again."""
    records = _pool_records(
        **{"e9_t0_origin": {"key_type": "origin", "task_id": 0, "trainable": True}}
    )
    harness = _GroupingHarness({}, records=records)
    harness.v9_key_pool.keys["e9_t0_origin"] = _Param(True, 1536)
    groups, illegal = harness.trainable_parameter_groups()
    assert illegal == [("e9_t0_origin", "retained origin key of task 0")]


def test_an_earlier_tasks_alias_key_is_illegal_but_this_tasks_is_not():
    records = _pool_records(**{
        "e9_t0_alias": {"key_type": "task_alias", "task_id": 0, "trainable": True},
        "e9_t1_alias": {"key_type": "task_alias", "task_id": 1, "trainable": True},
    })
    groups, illegal = _GroupingHarness({}, records=records, task_index=1).trainable_parameter_groups()
    assert illegal == [("e9_t0_alias", "retained task_alias key of task 0")]
    assert groups["current_task_historical_key"] == [("e9_t1_alias", 1536)]


def test_a_frozen_parameter_is_not_reported_at_all():
    """The grouping classifies trainables; frozen state is not evidence of anything."""
    params = {"model.embed_tokens.weight": _Param(False, 999), "vision_tower.x": _Param(False, 1)}
    groups, illegal = _GroupingHarness(params).trainable_parameter_groups()
    assert illegal == []
    assert all(not members for _, members in
               ((key, members) for key, members in groups.items()))


# ----------------------------------------------------------------------
# a row holds its own cells and no others
# ----------------------------------------------------------------------
def _fake_instructions(root: Path) -> None:
    for name in TASK_NAMES:
        target = root / name / "test_3000.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps([{"question_id": "q0", "text": "x"}]), encoding="utf-8")


def test_build_cells_emits_the_row_the_stage_asked_for(tmp_path, monkeypatch):
    """``A[t][0:t]`` is the row; a builder that emits all six is wrong for t<5.

    The chain builds a row's cells immediately before evaluating it, and
    ``plan_cells`` requires the file to hold exactly ``range(stage+1)``.  Emitting
    every task here made the first row of every run fail its own check, which is
    a failure that would otherwise surface hours into a formal chain.
    """
    instructions = tmp_path / "instructions"
    _fake_instructions(instructions)
    cells_path = tmp_path / "cells_t2.json"
    monkeypatch.setattr("sys.argv", [
        "formal_eval", "--build-cells", "--root", str(tmp_path),
        "--stage", "2", "--key-state", str(tmp_path / "keys.pt"),
        "--cells-json", str(cells_path), "--instructions-root", str(instructions),
        "--query-cache-manifest", str(tmp_path / "manifest.json"),
        "--query-cache-root", str(tmp_path / "query_cache"),
    ])
    formal_eval_main()
    cells = json.loads(cells_path.read_text(encoding="utf-8"))
    assert [cell["task_index"] for cell in cells] == [0, 1, 2]
    plan = plan_cells(tmp_path, 2, cells, tmp_path / "keys.pt")
    assert [item["task_index"] for item in plan] == [0, 1, 2]


def test_the_last_row_still_holds_every_cell(tmp_path):
    """The cap is the stage, not a truncation: stage 5 is the whole triangle."""
    instructions = tmp_path / "instructions"
    _fake_instructions(instructions)
    cells = build_cells(instructions, "manifest.json", "root", tasks=range(6))
    assert [cell["task_index"] for cell in cells] == [0, 1, 2, 3, 4, 5]
    plan = plan_cells(tmp_path, 5, cells, tmp_path / "keys.pt")
    assert len(plan) == 6


def test_a_full_cell_file_is_rejected_as_a_row(tmp_path):
    """The guard that caught this is kept: extra cells are not a row."""
    instructions = tmp_path / "instructions"
    _fake_instructions(instructions)
    all_cells = build_cells(instructions, "manifest.json", "root")
    with pytest.raises(V9FormalEvaluationError, match="requires cells"):
        plan_cells(tmp_path, 0, all_cells, tmp_path / "keys.pt")


def test_a_missing_cell_is_not_papered_over(tmp_path):
    """Rows are checked for content, so a gap cannot pass as a shorter row."""
    instructions = tmp_path / "instructions"
    _fake_instructions(instructions)
    cells = [cell for cell in build_cells(instructions, "manifest.json", "root")
             if cell["task_index"] != 1]
    with pytest.raises(V9FormalEvaluationError, match="requires cells"):
        plan_cells(tmp_path, 2, cells, tmp_path / "keys.pt")


# ----------------------------------------------------------------------
def _records(count: int):
    return [{"question_id": "q{}".format(index)} for index in range(count)]


def test_merge_shards_reemits_in_record_order(tmp_path):
    """The order is the result, not an incidental property of the merge."""
    records = _records(6)
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    # Written deliberately out of order within each shard, and shard 1 holds
    # records that precede shard 0's.
    first.write_text("\n".join(json.dumps({"question_id": "q3", "text": "d"}) for _ in [0])
                     + "\n" + json.dumps({"question_id": "q1", "text": "b"}) + "\n",
                     encoding="utf-8")
    second.write_text("\n".join(json.dumps({"question_id": key, "text": key})
                                for key in ("q5", "q0", "q4", "q2")) + "\n", encoding="utf-8")
    output = tmp_path / "answers.jsonl"
    merge_shards([first, second], records, output)
    merged = [json.loads(line)["question_id"] for line in output.read_text().splitlines()]
    assert merged == ["q{}".format(index) for index in range(6)]


def test_merge_shards_rejects_a_dropped_row(tmp_path):
    records = _records(3)
    part = tmp_path / "a.jsonl"
    part.write_text(json.dumps({"question_id": "q0"}) + "\n", encoding="utf-8")
    with pytest.raises(V9FormalEvaluationError, match="incomplete shard merge"):
        merge_shards([part], records, tmp_path / "out.jsonl")


def test_merge_shards_rejects_a_duplicate_row(tmp_path):
    records = _records(2)
    parts = []
    for index in range(2):
        part = tmp_path / "p{}.jsonl".format(index)
        part.write_text(
            "\n".join(json.dumps({"question_id": key}) for key in ("q0", "q1")) + "\n",
            encoding="utf-8",
        )
        parts.append(part)
    with pytest.raises(V9FormalEvaluationError, match="duplicate question_id"):
        merge_shards(parts, records, tmp_path / "out.jsonl")


def test_merge_shards_rejects_an_unknown_row(tmp_path):
    records = _records(1)
    part = tmp_path / "a.jsonl"
    part.write_text(json.dumps({"question_id": "intruder"}) + "\n", encoding="utf-8")
    with pytest.raises(V9FormalEvaluationError, match="unexpected question_id"):
        merge_shards([part], records, tmp_path / "out.jsonl")


def test_merge_shards_refuses_ambiguous_record_ids(tmp_path):
    records = [{"question_id": "q0"}, {"question_id": "q0"}]
    part = tmp_path / "a.jsonl"
    part.write_text(json.dumps({"question_id": "q0"}) + "\n", encoding="utf-8")
    with pytest.raises(V9FormalEvaluationError, match="duplicate ids"):
        merge_shards([part], records, tmp_path / "out.jsonl")


def test_merge_shards_reports_a_missing_shard(tmp_path):
    with pytest.raises(V9FormalEvaluationError, match="missing shard output"):
        merge_shards([tmp_path / "absent.jsonl"], _records(1), tmp_path / "out.jsonl")


# ----------------------------------------------------------------------
# the DataLoader verdict
# ----------------------------------------------------------------------
def test_a_wait_below_one_percent_leaves_the_loader_alone():
    result = verdict(0.0038)
    assert result["below_threshold"]
    assert not result["dataloader_change_warranted"]


def test_a_wait_above_one_percent_warrants_profiling_it():
    result = verdict(0.25)
    assert not result["below_threshold"]
    assert result["dataloader_change_warranted"]


def test_measure_does_not_count_the_first_micro_steps_missing_wait():
    """A ``None`` wait is unmeasured, not zero.

    Counting it as zero would bias the fraction downward by exactly one step,
    which on a short run is the difference between above and below the
    threshold.
    """
    rows = [
        {"inter_step_wait_sec": None, "training_step_sec": 10.0},
        {"inter_step_wait_sec": 1.0, "training_step_sec": 9.0},
    ]
    measured = measure(rows)
    assert measured["rows"] == 1
    assert measured["fraction"] == pytest.approx(0.1)


def test_measure_prefers_the_profiler_rows_over_the_trainer_rows():
    """The two sources measure the same seconds; summing them double-counts."""
    rows = [
        {"data_wait_time": 0.5, "step_body_time": 9.5},
        {"inter_step_wait_sec": 99.0, "training_step_sec": 1.0},
    ]
    measured = measure(rows)
    assert measured["source"] == "profiler"
    assert measured["fraction"] == pytest.approx(0.05)


def test_measure_refuses_to_decide_without_a_measurement():
    with pytest.raises(DataWaitError):
        measure([{"training_step_sec": 1.0}])


def test_discover_finds_the_profiler_file_a_real_run_writes(tmp_path):
    """The profiler names its file ``task{N}_profile_steps.jsonl``.

    A ``profile_steps*`` glob matches nothing there, so the verdict silently
    fell back to the trainer rows -- the source the module documents as the
    less informative one, and on this run the one that disagreed.
    """
    metrics = tmp_path / "task0" / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "task0_profile_steps.rank0.jsonl").write_text("{}\n", encoding="utf-8")
    (metrics / "task0_train_steps.rank0.jsonl").write_text("{}\n", encoding="utf-8")
    found = {path.name for path in discover(tmp_path)}
    assert found == {"task0_profile_steps.rank0.jsonl", "task0_train_steps.rank0.jsonl"}


def test_the_verdict_uses_the_profiler_when_a_run_has_one(tmp_path):
    """Both files present, and the profiler is the one that decides."""
    metrics = tmp_path / "task0" / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "task0_profile_steps.rank0.jsonl").write_text(
        json.dumps({"data_wait_time": 0.4, "window_wall_time": 10.0}) + "\n", encoding="utf-8")
    (metrics / "task0_train_steps.rank0.jsonl").write_text(
        json.dumps({"inter_step_wait_sec": 0.001, "training_step_sec": 10.0}) + "\n",
        encoding="utf-8")
    payload = report(tmp_path)
    assert payload["source"] == "profiler"
    assert payload["fraction"] == pytest.approx(0.04)
    assert payload["dataloader_change_warranted"] is True


# ----------------------------------------------------------------------
# performance warnings (never fatal)
# ----------------------------------------------------------------------
def _step_row(**overrides):
    row = {
        "step": 0, "training_step_sec": 10.0, "inter_step_wait_sec": 0.1,
        "peak_memory_bytes": 8 * (1024 ** 3),
        "backbone_forwards_per_micro_step": 1.0,
        "model_forwards_per_micro_step": 1.0,
        "wide_model_forwards_per_micro_step": 0.0,
    }
    row.update(overrides)
    return row


def _write_metrics(root: Path, rows) -> None:
    metrics = root / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    (metrics / "task0_train_steps.rank0.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def test_a_healthy_window_produces_no_warning(tmp_path):
    _write_metrics(tmp_path, [_step_row(step=index) for index in range(20)])
    report = performance_report(tmp_path)
    assert not report["warning"], report["warnings"]
    assert report["peak_allocated_GiB"] == pytest.approx(8.0)


def test_the_hook_catching_two_traversals_is_a_warning_not_a_stop(tmp_path):
    """Performance is reported, never fatal: a slow step is not a wrong one."""
    _write_metrics(tmp_path, [_step_row(step=index, model_forwards_per_micro_step=2.0)
                              for index in range(20)])
    report = performance_report(tmp_path)
    assert report["warning"]
    assert any("forward hook" in message for message in report["warnings"])


def test_wide_retrieval_that_costs_a_traversal_is_flagged(tmp_path):
    _write_metrics(tmp_path, [_step_row(step=index, wide_model_forwards_per_micro_step=1.5)
                              for index in range(20)])
    report = performance_report(tmp_path)
    assert any("wide retrieval" in message for message in report["warnings"])


def test_a_stalled_loader_is_flagged_above_thirty_percent(tmp_path):
    _write_metrics(tmp_path, [_step_row(step=index, training_step_sec=1.0,
                                        inter_step_wait_sec=1.0) for index in range(20)])
    report = performance_report(tmp_path)
    assert any("data wait" in message for message in report["warnings"])


def test_an_empty_window_is_reported_rather_than_crashing(tmp_path):
    report = performance_report(tmp_path)
    assert report["warning"]
    assert report["steps_observed"] == 0


# ----------------------------------------------------------------------
# git binding
# ----------------------------------------------------------------------
def test_a_dirty_tree_is_detected(tmp_path):
    """Spec §34: the run refuses to start on a tree that can change under it."""
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "one"], cwd=repo, check=True)
    assert not _git_dirty(repo)
    (repo / "tracked.txt").write_text("two\n", encoding="utf-8")
    assert _git_dirty(repo)
    subprocess.run(["git", "checkout", "--", "tracked.txt"], cwd=repo, check=True)
    assert not _git_dirty(repo)
    (repo / "staged.txt").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "add", "staged.txt"], cwd=repo, check=True)
    assert _git_dirty(repo)


def test_markers_do_not_survive_a_restart(tmp_path):
    """A run root may not hold COMPLETE beside FAILED for a reader to choose from."""
    for name in ("FORMAL_RUNNING", "FORMAL_COMPLETE", "SANITY_FAILED"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    _clear_markers(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == []


def test_the_sanity_and_formal_markers_are_distinct(tmp_path):
    """Nothing a sanity run writes may be readable as a formal verdict."""
    from compose.experiments import v9_chain

    assert v9_chain.MARKER_COMPLETE == "FORMAL_COMPLETE"
    assert v9_chain.MARKER_FAILED == "FORMAL_FAILED"
    assert "SANITY_COMPLETE" != v9_chain.MARKER_COMPLETE
