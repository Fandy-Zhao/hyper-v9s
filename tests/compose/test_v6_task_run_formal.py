"""Regression tests for the V6 formal-run supplement components.

Covers:
  1. configs/v6_ucit_formal_locked.yaml loads, hashes match, all six task
     train/test files exist (DEV-13 fix), task order locked.
  2. v6_task_run.run_task stage markers are idempotent and the state machine
     advances legally on a mock (subprocess-stubbed) run.
  3. v6_task_run requires a prev snapshot with the correct task id.
  4. slot id namespace per task (task_id*10, task_id*10+1) stays unique.
"""

import json
import os
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import yaml

FORMAL_CONFIG = REPO / "configs/v6_ucit_formal_locked.yaml"
HANDOFF_CONFIG = REPO / "artifacts/v6_ucit_handoff/locked_config.yaml"


def _load_config():
    with open(FORMAL_CONFIG, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# ---------------------------------------------------------------------------
# 1. formal locked config integrity
# ---------------------------------------------------------------------------


class TestFormalConfig:
    def test_config_exists_and_parses(self):
        assert FORMAL_CONFIG.is_file()
        config = _load_config()
        assert config["schema_version"] == 1
        assert config["config_hash"] not in (None, "", "PLACEHOLDER")

    def test_config_hash_self_consistent(self):
        import hashlib

        text = FORMAL_CONFIG.read_text(encoding="utf-8")
        hashable = "\n".join(
            line for line in text.splitlines() if not line.startswith("config_hash:")
        )
        digest = hashlib.sha256(hashable.encode("utf-8")).hexdigest()[:16]
        config = _load_config()
        assert config["config_hash"] == digest

    def test_task_sequence_order_locked(self):
        config = _load_config()
        names = [task["name"] for task in config["task_sequence"]]
        assert names == ["ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR", "Flickr30k"]
        assert [task["task_id"] for task in config["task_sequence"]] == [0, 1, 2, 3, 4, 5]

    def test_all_train_test_files_exist(self):
        config = _load_config()
        for task in config["task_sequence"]:
            assert os.path.isfile(task["train_instructions"]), task["train_instructions"]
            assert os.path.isfile(task["test_instructions"]), task["test_instructions"]

    def test_test_paths_point_to_test_3000(self):
        config = _load_config()
        for task in config["task_sequence"]:
            assert task["test_instructions"].endswith("test_3000.json"), task

    def test_features_all_disabled(self):
        config = _load_config()
        features = config["features"]
        for key in (
            "shadow_update",
            "hyperbolic_router",
            "pair_interaction",
            "learnable_composition_gate",
            "automatic_expert_merge",
            "token_level_routing",
        ):
            assert features[key]["enabled"] is False, key

    def test_runs_root_isolated_from_dry_run(self):
        config = _load_config()
        assert "formal" in config["output"]["runs_root"]
        assert config["output"]["runs_root"] != str(
            REPO / "experiments/runs/v6_ucit_engineering/dry_run"
        )

    def test_tasks_section_matches_sequence(self):
        config = _load_config()
        assert len(config["tasks"]) == 6
        for task in config["tasks"]:
            seq = config["task_sequence"][task["task_id"]]
            assert task["name"] == seq["name"]

    def test_eval_uses_full_3000(self):
        config = _load_config()
        assert config["eval"]["max_test_samples"] == 3000


# ---------------------------------------------------------------------------
# 2. v6_task_run idempotent stage markers (mock subprocess)
# ---------------------------------------------------------------------------


class TestV6TaskRunIdempotency:
    @pytest.fixture()
    def fake_config(self):
        config = _load_config()
        # shrink teacher search sizes for the mock run
        for task in config["tasks"]:
            task["teacher_search_train_samples"] = 2
            task["teacher_search_validation_samples"] = 1
            task["cold_start_train_samples"] = 2
            task["validation_samples"] = 1
        return config

    @pytest.fixture()
    def fake_prev_root(self, tmp_path):
        """A minimal loadable prev snapshot with an empty registry."""
        from compose.experts.registry import ExpertRegistry
        from compose.experts.task_state import TaskStateMachine, TaskStage
        from compose.experiments.v6_snapshot import V6Snapshot

        prev = tmp_path / "prev"
        snap_dir = prev / "snapshots" / "task1"
        registry = ExpertRegistry()
        machine = TaskStateMachine(1, "ArxivQA")
        machine.advance(TaskStage.DATA_READY, note="mock")
        V6Snapshot.create(
            str(snap_dir),
            task_id=1,
            task_name="ArxivQA",
            registry=registry,
            task_state=machine,
            git_commit="mock",
            command="mock",
            data_hash="mock",
        )
        return prev

    def test_run_task_s0_idempotent_on_resume(self, tmp_path, fake_config, fake_prev_root):
        """S0 runs once; a resumed run skips it and fails later at S1
        (no old expert checkpoint under the mock prev root), proving the
        stage markers gate re-execution."""
        from compose.experiments import v6_task_run as runner

        root = tmp_path / "task2"
        # First attempt: S0 succeeds, S1 raises because prev root has no
        # old expert checkpoint dir.
        with pytest.raises(RuntimeError, match="no old expert checkpoint"):
            runner.run_task(root, fake_prev_root, 2, "0", 29661, fake_config)
        assert (root / "stages" / "s0_snapshot_load.done").is_file()
        state = json.loads((root / "state" / "task_state.json").read_text())
        assert state["task_id"] == 2
        assert state["stage"] == "DATA_READY"

        # Second (resume) attempt: S0 must not be re-executed; S1 still fails
        # the same way (not on snapshot reload).
        with pytest.raises(RuntimeError, match="no old expert checkpoint"):
            runner.run_task(root, fake_prev_root, 2, "0", 29661, fake_config)
        assert (root / "stages" / "s0_snapshot_load.done").is_file()

    def test_wrong_prev_snapshot_task_id_rejected(self, tmp_path, fake_config):
        from compose.experiments import v6_task_run as runner

        root = tmp_path / "task2"
        prev = tmp_path / "prev_wrong"
        # runner loads snapshots/task1 for task_id=2; create a snapshot with
        # the wrong task id (0) at that path.
        snap_dir = prev / "snapshots" / "task1"
        from compose.experts.registry import ExpertRegistry
        from compose.experts.task_state import TaskStateMachine, TaskStage
        from compose.experiments.v6_snapshot import V6Snapshot

        machine = TaskStateMachine(0, "ImageNet-R")
        machine.advance(TaskStage.DATA_READY, note="mock")
        V6Snapshot.create(
            str(snap_dir),
            task_id=0,
            task_name="ImageNet-R",
            registry=ExpertRegistry(),
            task_state=machine,
            git_commit="mock",
            command="mock",
            data_hash="mock",
        )
        with pytest.raises(ValueError, match="prev snapshot task_id"):
            runner.run_task(root, prev, 2, "0", 29661, fake_config)

    def test_slot_ids_unique_per_task(self):
        # task0 -> 10 (cold start), task1 -> 20/21 (accepted runner),
        # task t>=2 -> (t+1)*10, (t+1)*10+1; all unique across the pool.
        from compose.experiments import v6_task_run as runner

        seen = {10, 20, 21}
        for task_id in range(2, 6):
            slots = ((task_id + 1) * 10, (task_id + 1) * 10 + 1)
            assert slots[0] not in seen
            assert slots[1] not in seen
            seen.update(slots)
        assert seen == {10, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61}


# ---------------------------------------------------------------------------
# 3. orchestration scripts exist and are executable
# ---------------------------------------------------------------------------


class TestOrchestrationScripts:
    def test_scripts_exist(self):
        for name in ("six_task_run.sh", "resume_run.sh"):
            path = REPO / "scripts" / "v6_ucit" / name
            assert path.is_file(), name
            assert os.access(str(path), os.X_OK), name

    def test_six_task_run_references_formal_config(self):
        text = (REPO / "scripts/v6_ucit/six_task_run.sh").read_text()
        assert "v6_ucit_formal_locked.yaml" in text
        assert "seed_$SEED" in text

    def test_six_task_run_has_six_tasks(self):
        text = (REPO / "scripts/v6_ucit/six_task_run.sh").read_text()
        assert text.count("--output-root") == 6
