"""CPU tests for the Phase B twin-run equivalence comparator
(``compose/eval/v7_twin_run_compare.py``): identical synthetic run roots
must PASS every stage gate; each targeted perturbation must FAIL exactly
its own gate."""

import json
import math

import pytest

from compose.eval import v7_twin_run_compare as twin


def _query_row(seed: int) -> list:
    # deterministic unit-norm 1536-D row
    values = [(math.sin(seed * 1000.0 + i) + 1.0) / 2.0 for i in range(1536)]
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


def _features_payload(sample_ids, perturb: float = 0.0) -> dict:
    # alternating-sign perturbation: perpendicular-ish to the row so the
    # cosine drop is ~ 1536*perturb**2/2 (deterministic, no common-mode
    # norm shift)
    return {
        "schema_version": 1,
        "feature_source": "frozen_clip_l14_336",
        "query_mode": "v7_fixed",
        "records": {
            sid: {"query": [v + (perturb if i % 2 == 0 else -perturb)
                            for i, v in enumerate(_query_row(seed))]}
            for seed, sid in enumerate(sample_ids)
        },
    }


def _nll_payload(sample_ids, perturb: float = 0.0) -> dict:
    return {
        sid: {"global_top2": {"eos_policy": "same_as_training_labels",
                              "mean_answer_nll": 1.0 + perturb + seed * 0.01,
                              "tokens": 12 + seed % 3}}
        for seed, sid in enumerate(sample_ids)
    }


def _selections_payload(sample_ids, perturb: bool = False) -> dict:
    ids = {"a": [0, 3], "b": [2, 1], "c": [3, 0]}
    return {sid: {"global_top2": ids[sid[0]] if not perturb else [0, 0]}
            for sid in sample_ids}


def _rms_payload(perturb: float = 0.0) -> dict:
    return {
        "rms_mode": "commit_frozen",
        "samples": 256,
        "expert_ids": [0, 1, 2, 3],
        "layers": [
            {"layer_index": i, "kappa": 0.01 * i + 1.0 + perturb,
             "mean_activation": 2.0 + i * 0.001}
            for i in range(4)
        ],
    }


def _official_metric_payload(perturb: float = 0.0) -> dict:
    return {"dataset": "ImageNet-R", "metric": "Accuracy",
            "accuracy": 0.25 + perturb}


def _train_step_row(step: int, perturb: float = 0.0) -> dict:
    return {
        "step": step,
        "answer_loss": 0.35 + perturb + step * 0.01,
        "key_loss": 0.11 + perturb,
        "total_loss": 0.36 + perturb,
        "local_batch_size": 1,
        "route_types": ["historical_current"],
        "sample_ids": ["v7_t0_train_{}".format(step)],
        "selected_expert_ids": [0],
        "selected_current_ids": [0, 3],
        "selected_current_key_grad_norm": 5.0e-4 + perturb,
        "selected_current_lora_grad_norm": 1.5e-2 + perturb,
        "per_sample_key_loss": [0.11 + perturb],
        "old_old_noop": False,
        "gradient_window_synced": True,
        "wall_time": 1788420167.0 + step,
        "training_step_sec": 2.15,
        "inter_step_wait_sec": None,
    }


def _write_root(root, *, perturb_query: float = 0.0,
                perturb_steps: float = 0.0, perturb_rms: float = 0.0,
                perturb_nll: float = 0.0, perturb_metric: float = 0.0,
                perturb_selections: bool = False, drop_commit: bool = False,
                commit_perturb: bool = False, drop_job1: bool = False):
    features = root / "features"
    features.mkdir(parents=True, exist_ok=True)
    sample_ids = ["a1", "b2", "c3"]
    (features / "train.json").write_text(json.dumps(
        _features_payload(sample_ids, perturb_query)))
    (features / "val.json").write_text(json.dumps(
        _features_payload(sample_ids, perturb_query)))
    metrics = root / "metrics"
    metrics.mkdir(exist_ok=True)
    (metrics / "train_steps.jsonl").write_text("\n".join(
        json.dumps(_train_step_row(step, perturb_steps)) for step in range(2)))
    rms = root / "rms"
    rms.mkdir(exist_ok=True)
    for filename, payload in (
            ("rms_summary.json", _rms_payload(perturb_rms)),
            ("rms_calibration.json", _rms_payload(perturb_rms)),
            ("rms_statistics.json", _rms_payload(perturb_rms))):
        (rms / filename).write_text(json.dumps(payload))
    pruning = root / "pruning"
    pruning.mkdir(exist_ok=True)
    if drop_job1:  # remove leftovers of a previous full write
        for filename in ("selections_1.json", "nll_1.json",
                         "official_metric_1.json"):
            (pruning / filename).unlink(missing_ok=True)
    for job in (0, 1):
        if drop_job1 and job == 1:
            continue
        (pruning / "selections_{}.json".format(job)).write_text(json.dumps(
            _selections_payload(sample_ids, perturb_selections)))
        (pruning / "nll_{}.json".format(job)).write_text(json.dumps(
            _nll_payload(sample_ids, perturb_nll)))
        (pruning / "official_metric_{}.json".format(job)).write_text(
            json.dumps(_official_metric_payload(perturb_metric)))
    committed = root / "committed"
    if drop_commit:
        import shutil
        shutil.rmtree(committed, ignore_errors=True)
    else:
        committed.mkdir(exist_ok=True)
        keys = b"frozen-keys-bytes"
        if commit_perturb:
            keys = b"frozen-keys-bytes-CHANGED"
        (committed / "v7_keys.pt").write_bytes(keys)
        (committed / "manifest.json").write_text(json.dumps(
            {"candidate_ids": [0, 3]}))


def _identical_roots(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _write_root(root_a)
    _write_root(root_b)
    return root_a, root_b


def test_identical_roots_pass_every_gate(tmp_path):
    root_a, root_b = _identical_roots(tmp_path)
    audit = twin.run_compare(root_a, root_b, gate_mode="cache")
    assert audit["verdict"] == "PASS"
    for name in ("S1_QUERY_ROWS_EQUIVALENCE", "S3_TRAIN_STEPS_EQUIVALENCE",
                 "RMS_CACHE_EQUIVALENCE", "PRUNING_TRAJECTORY_EQUIVALENCE",
                 "COMMIT_STATE_EQUIVALENCE"):
        assert audit[name]["verdict"] == "PASS", name


def test_gate_mode_distributed_renames_rms_verdict(tmp_path):
    root_a, root_b = _identical_roots(tmp_path)
    audit = twin.run_compare(root_a, root_b, gate_mode="distributed")
    assert audit["verdict"] == "PASS"
    assert "DISTRIBUTED_RMS_EQUIVALENCE" in audit
    assert "RMS_CACHE_EQUIVALENCE" not in audit


def test_query_perturbation_fails_s1_only(tmp_path):
    root_a, root_b = _identical_roots(tmp_path)
    _write_root(root_b, perturb_query=1e-4)  # cosine drop ~7.7e-6 > 1e-6
    audit = twin.run_compare(root_a, root_b)
    assert audit["S1_QUERY_ROWS_EQUIVALENCE"]["verdict"] == "FAIL"
    assert audit["verdict"] == "FAIL"
    # downstream artifacts untouched -> still PASS
    assert audit["S3_TRAIN_STEPS_EQUIVALENCE"]["verdict"] == "PASS"
    assert audit["RMS_CACHE_EQUIVALENCE"]["verdict"] == "PASS"


def test_sequence_mismatch_fails_s1(tmp_path):
    root_a, root_b = _identical_roots(tmp_path)
    (root_b / "features" / "train.json").unlink()
    payload = _features_payload(["c3", "a1", "b2"])  # reordered
    (root_b / "features" / "train.json").write_text(json.dumps(payload))
    audit = twin.run_compare(root_a, root_b)
    assert audit["S1_QUERY_ROWS_EQUIVALENCE"]["verdict"] == "FAIL"


def test_step_perturbation_fails_s3(tmp_path):
    root_a, root_b = _identical_roots(tmp_path)
    _write_root(root_b, perturb_steps=1e-3)  # over LOSS_ATOL
    audit = twin.run_compare(root_a, root_b)
    assert audit["S3_TRAIN_STEPS_EQUIVALENCE"]["verdict"] == "FAIL"
    assert audit["S1_QUERY_ROWS_EQUIVALENCE"]["verdict"] == "PASS"
    assert audit["RMS_CACHE_EQUIVALENCE"]["verdict"] == "PASS"
    assert audit["PRUNING_TRAJECTORY_EQUIVALENCE"]["verdict"] == "PASS"


def test_step_id_field_mismatch_fails_s3(tmp_path):
    root_a, root_b = _identical_roots(tmp_path)
    row = _train_step_row(0)
    row["selected_expert_ids"] = [1]  # different route
    (root_b / "metrics" / "train_steps.jsonl").write_text("\n".join(
        json.dumps(row if step == 0 else _train_step_row(step))
        for step in range(2)))
    audit = twin.run_compare(root_a, root_b)
    assert audit["S3_TRAIN_STEPS_EQUIVALENCE"]["verdict"] == "FAIL"


def test_rms_perturbation_fails_rms_gate(tmp_path):
    root_a, root_b = _identical_roots(tmp_path)
    _write_root(root_b, perturb_rms=1.0)  # far over tolerance
    audit = twin.run_compare(root_a, root_b)
    assert audit["RMS_CACHE_EQUIVALENCE"]["verdict"] == "FAIL"
    assert audit["PRUNING_TRAJECTORY_EQUIVALENCE"]["verdict"] == "PASS"


def test_pruning_job_set_mismatch_fails_trajectory(tmp_path):
    root_a, root_b = _identical_roots(tmp_path)
    _write_root(root_b, drop_job1=True)
    audit = twin.run_compare(root_a, root_b)
    assert audit["PRUNING_TRAJECTORY_EQUIVALENCE"]["verdict"] == "FAIL"
    assert audit["S1_QUERY_ROWS_EQUIVALENCE"]["verdict"] == "PASS"


def test_pruning_selection_flip_fails_trajectory(tmp_path):
    root_a, root_b = _identical_roots(tmp_path)
    _write_root(root_b, perturb_selections=True)
    audit = twin.run_compare(root_a, root_b)
    assert audit["PRUNING_TRAJECTORY_EQUIVALENCE"]["verdict"] == "FAIL"


def test_nll_perturbation_fails_trajectory(tmp_path):
    root_a, root_b = _identical_roots(tmp_path)
    _write_root(root_b, perturb_nll=0.05)
    audit = twin.run_compare(root_a, root_b)
    assert audit["PRUNING_TRAJECTORY_EQUIVALENCE"]["verdict"] == "FAIL"


def test_metric_perturbation_fails_trajectory(tmp_path):
    root_a, root_b = _identical_roots(tmp_path)
    _write_root(root_b, perturb_metric=1e-2)
    audit = twin.run_compare(root_a, root_b)
    assert audit["PRUNING_TRAJECTORY_EQUIVALENCE"]["verdict"] == "FAIL"


def test_commit_sha_mismatch_fails_commit_gate(tmp_path):
    root_a, root_b = _identical_roots(tmp_path)
    _write_root(root_b, commit_perturb=True)
    audit = twin.run_compare(root_a, root_b)
    assert audit["COMMIT_STATE_EQUIVALENCE"]["verdict"] == "FAIL"
    assert audit["PRUNING_TRAJECTORY_EQUIVALENCE"]["verdict"] == "PASS"


def test_missing_commit_fails_commit_gate(tmp_path):
    root_a, root_b = _identical_roots(tmp_path)
    _write_root(root_b, drop_commit=True)
    audit = twin.run_compare(root_a, root_b)
    assert audit["COMMIT_STATE_EQUIVALENCE"]["verdict"] == "FAIL"


def test_perturbed_query_within_tolerance_passes_s1(tmp_path):
    # sub-tolerance cosine noise (fp16 CLIP forward scale) must PASS
    root_a, root_b = _identical_roots(tmp_path)
    _write_root(root_b, perturb_query=1e-8)
    audit = twin.run_compare(root_a, root_b)
    assert audit["S1_QUERY_ROWS_EQUIVALENCE"]["verdict"] == "PASS"
    assert audit["verdict"] == "PASS"


@pytest.mark.parametrize("gate_mode", ["cache", "distributed"])
def test_main_exit_code(tmp_path, gate_mode, capsys):
    root_a, root_b = _identical_roots(tmp_path)
    output = tmp_path / "audit.json"
    code = twin.main(["--root-a", str(root_a), "--root-b", str(root_b),
                      "--gate-mode", gate_mode,
                      "--audit-output", str(output)])
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS"
    out = capsys.readouterr().out
    assert "OVERALL: PASS" in out
