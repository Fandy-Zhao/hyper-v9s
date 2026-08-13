import ast
import json
from pathlib import Path

import pytest

from compose.experiments.task_run import _load_config


REPO = Path(__file__).resolve().parents[2]


def test_v62_formal_config_contract():
    config = _load_config(str(REPO / "configs" / "compose_ucit.yaml"))
    assert config["data"]["seed"] == 42
    assert [value["name"] for value in config["task_sequence"]] == [
        "ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR", "Flickr30k"
    ]
    assert config["lora"] == {"rank": 8, "alpha": 16.0}
    assert config["clustering"]["max_clusters"] == 4
    assert config["clustering"]["n_init"] == 20
    assert config["router"]["top_m"] == 8
    assert config["router"]["max_active_experts"] == 2
    assert config["set_router"]["pair_threshold"] == 0.65


def test_runner_exposes_explicit_resume_and_hard_gpu_contract():
    source = (REPO / "compose" / "experiments" / "task_run.py").read_text()
    tree = ast.parse(source)
    assert 'parser.add_argument("--resume", action="store_true"' in source
    assert 'args.gpus != "4,5,6,7"' in source
    assert '"seed43_44_started": False' in source


def test_v62_smoke_is_non_destructive_and_checks_method_invariants():
    source = (REPO / "scripts" / "Compose" / "smoke_4gpu.sh").read_text()
    assert 'rm -rf "$SMOKE_ROOT"' not in source
    assert "compose_ucit_v62_smoke_4gpu_r5" in source
    assert "expected exactly one bootstrap expert" in source
    assert 'cluster1.get("n_init") != 20' in source
    assert 'summary.get("rms_mode") != "commit_frozen"' in source
    assert '--batch-size 1' in source
    assert '--question-file "$DATA/parity64.json"' in source
    phase6 = source[source.index("=== Phase 6:"):source.index("=== Phase 7:")]
    assert "$PY -m compose.eval.formal_ucit_eval" not in phase6
    assert 'value.get("rank") == 0 and "samples_per_second" in value' in source
    assert "grep '\"rank\": 0'" not in source


def test_formal_summary_writes_required_v62_artifact_names():
    source = (REPO / "compose" / "eval" / "formal_ucit_summary.py").read_text()
    for name in (
        "continual_metrics_wrapper.json", "expert_growth.csv",
        "routing_diagnostics.json", "cross_task_contribution.csv",
    ):
        assert name in source


def test_rms_binding_is_the_only_intentional_registry_replacement():
    source = (REPO / "compose" / "experiments" / "task_run.py").read_text()
    binding_start = source.index(
        'if cluster_expert_ids and not _stage_done(root, "s9_rms_frozen_binding"):'
    )
    snapshot_start = source.index("# ---- S10:", binding_start)
    binding_source = source[binding_start:snapshot_start]
    assert "registry.save_atomic(" in binding_source
    assert "allow_overwrite=True" in binding_source
    assert "registry.save_json(" not in binding_source


def test_sparse_pair_search_validates_only_samples_with_pair_candidates():
    source = (REPO / "compose" / "experiments" / "task_run.py").read_text()
    assert "pair_subset = [" in source
    assert "if _record_id(record) in pair_selections" in source
    assert "nll_r2_command,\n                        pair_subset," in source
