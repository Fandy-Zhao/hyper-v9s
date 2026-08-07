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
import torch

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

    def test_run_task_empty_registry_s0_idempotent_and_rebootstrap(
        self, tmp_path, fake_config, fake_prev_root, monkeypatch
    ):
        """Empty-registry fix: S0 runs once; an empty active pool scores the
        base-only pool (real base NLL, never the old synthetic None losses),
        the residual split produces candidate material, and candidate
        training re-bootstraps WITHOUT --old-expert-checkpoint. The mock
        validation gains stay below tau_support (locked config), so both
        candidates are rejected and archived; eval then scores the base-only
        pool. A resumed run skips every stage (markers gate re-execution)."""
        from compose.experiments import v6_task_run as runner

        captured = {}

        def fake_subprocess_run(command, env=None, capture_output=False, text=False,
                                **kwargs):
            if command[0] == "git":
                return types.SimpleNamespace(
                    stdout="mockcommit", stderr="", returncode=0
                )
            if command[0] == "nvidia-smi":
                return types.SimpleNamespace(
                    stdout="", stderr="", returncode=0
                )
            module = command[2]
            if module == "compose.eval.v6_nll_eval":
                output = next(
                    command[index + 1]
                    for index, arg in enumerate(command)
                    if arg == "--output"
                )
                selections_path = next(
                    command[index + 1]
                    for index, arg in enumerate(command)
                    if arg == "--selections"
                )
                selections = json.loads(Path(selections_path).read_text())
                rows = {}
                for sample_id, keys in selections.items():
                    row = {}
                    for key in keys:
                        if key == "empty" or key == "old":
                            row[key] = 5.0
                        else:
                            row[key] = 3.9
                    rows[sample_id] = row
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                Path(output).write_text(json.dumps(rows), encoding="utf-8")
            elif module == "compose.eval.v6_query_features":
                output = next(
                    command[index + 1]
                    for index, arg in enumerate(command)
                    if arg == "--output"
                )
                questions = json.loads(
                    Path(
                        next(
                            command[index + 1]
                            for index, arg in enumerate(command)
                            if arg == "--questions"
                        )
                    ).read_text()
                )
                records = {}
                for index, record in enumerate(questions):
                    sample_id = record.get("id", record.get("question_id"))
                    # Distinct rows: k-means++ needs at least as many
                    # distinct query rows as candidate slots.
                    records[str(sample_id)] = {
                        "image": [0.01 + 0.001 * index] * 768
                    }
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                Path(output).write_text(
                    json.dumps({"records": records}), encoding="utf-8"
                )
            elif module == "compose.train.train_v6_candidate":
                captured["candidate_command"] = command
                output_dir = Path(
                    next(
                        command[index + 1]
                        for index, arg in enumerate(command)
                        if arg == "--output-dir"
                    )
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "compose_experts.json").write_text(
                    json.dumps({"format_version": 1, "experts": []}),
                    encoding="utf-8",
                )
                for slot in next(
                    command[index + 1]
                    for index, arg in enumerate(command)
                    if arg == "--candidate-ids"
                ).split(","):
                    torch.save({}, str(output_dir / "candidate_{}.pt".format(slot)))
            elif module == "compose.eval.v6_assemble_expert":
                output_dir = Path(
                    next(
                        command[index + 1]
                        for index, arg in enumerate(command)
                        if arg == "--output-dir"
                    )
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "compose_experts.bin").write_bytes(b"")
                (output_dir / "compose_experts.json").write_text(
                    json.dumps({"format_version": 1, "experts": []}),
                    encoding="utf-8",
                )
                (output_dir / "assembly.json").write_text(
                    json.dumps(
                        {
                            "compose_experts_bin_sha256": "a" * 64,
                            "compose_experts_json_sha256": "b" * 64,
                        }
                    ),
                    encoding="utf-8",
                )
            elif module == "compose.eval.eval_task":
                answers = next(
                    command[index + 1]
                    for index, arg in enumerate(command)
                    if arg == "--answers-file"
                )
                summary = next(
                    command[index + 1]
                    for index, arg in enumerate(command)
                    if arg == "--run-summary-file"
                )
                Path(answers).parent.mkdir(parents=True, exist_ok=True)
                Path(answers).write_text("", encoding="utf-8")
                Path(summary).write_text(
                    json.dumps({"samples": 0}), encoding="utf-8"
                )
            else:
                raise AssertionError("unexpected module: {}".format(module))
            return types.SimpleNamespace(stdout="", stderr="", returncode=0)

        monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)

        root = tmp_path / "task2"
        runner.run_task(root, fake_prev_root, 2, "0", 29661, fake_config)
        assert (root / "stages" / "s0_snapshot_load.done").is_file()
        state = json.loads((root / "state" / "task_state.json").read_text())
        assert state["stage"] == "COMPLETED"

        # S1: base teacher over the base-only pool -> real losses (R4).
        summary = json.loads((root / "teacher" / "summary.json").read_text())
        assert summary["base_only_mode"] is True
        assert summary["active_expert_ids"] == []
        assert summary["pool_checkpoint_dir"].endswith(
            str(Path("prev") / "candidate" / "base_only")
        )
        records = json.loads(
            (root / "teacher" / "teacher_records_train.json").read_text()
        )
        assert records
        assert all(record["teacher_set"] == [] for record in records)
        assert all(
            record["empty_loss"] is not None for record in records
        ), "base-only teacher must score real NLL, never None"

        # S3: empty registry still produces residual material (R5).
        residual = json.loads((root / "residual" / "residual.json").read_text())
        assert len(residual) == 2
        assert all(
            record["residual_reason"] == "base_only_insufficient"
            for record in residual
        )
        assert all(record["old_gain"] == 0.0 for record in residual)

        # S5: re-bootstrap mode — same trainer, no old-expert flag (R6).
        mode = json.loads((root / "candidate" / "mode.json").read_text())
        assert mode["mode"] == "rebootstrap"
        assert mode["old_expert_checkpoint"] is None
        assert "--old-expert-checkpoint" not in captured["candidate_command"]

        # S7: validation gains below the locked tau_support -> rejected
        # candidates archived as diagnostics (R2); never active.
        commit = json.loads((root / "committed" / "commit_record.json").read_text())
        assert commit["committed_expert_ids"] == []
        assert commit["reason"] == "all_candidates_below_tau"
        registry = json.loads(
            (root / "state" / "expert_registry.json").read_text()
        )
        assert registry["active_lifecycle_ids"] == []
        assert sorted(registry["rejected_candidate_ids"]) == [30, 31]
        assert (root / "rejected_candidates" / "task_02" / "candidate_30").is_dir()

        # S11: empty selection over the base-only pool (R3): no
        # inherited-checkpoint fallback, no last_known_checkpoint.
        assert not (root / "candidate" / "last_known_checkpoint.json").exists()

        # Resume: every stage marker gates re-execution; S0 runs once.
        runner.run_task(root, fake_prev_root, 2, "0", 29661, fake_config)
        history = json.loads((root / "state" / "task_state.json").read_text())["history"]
        assert sum(1 for item in history if item["note"] == "prev snapshot loaded") == 1
        assert (root / "stages" / "s11_eval.done").is_file()

    def test_draw_train_val_splits_raises_on_short_dataset(self):
        """The task-0 split must never silently return a short/empty
        validation slice: a 0-sample validation silently yields an
        all-below_tau commit decision (mean_gain 0.0), which is a bug
        artifact, not a data-driven outcome."""
        from compose.experiments import v6_task1_dry_run as runner

        records = [{"id": i} for i in range(5)]
        # cold start consumes everything -> validation slice empty -> loud error
        with pytest.raises(RuntimeError, match="train split is short"):
            runner._draw_train_val_splits(records, 5, 2)
        with pytest.raises(RuntimeError, match="train split is short"):
            runner._draw_train_val_splits(records, 4, 2)
        # a fitting split draws disjoint seeded-shuffle slices covering
        # the dataset exactly (fix 9: no longer the positional tail)
        train_ids, val_ids = runner._draw_train_val_splits(records, 3, 2)
        assert sorted(train_ids + val_ids) == ["0", "1", "2", "3", "4"]
        assert len(train_ids) == 3
        assert len(val_ids) == 2
        assert not set(train_ids) & set(val_ids)

    def test_task0_cold_start_split_fits_real_dataset(self):
        """Config v6 regression: the locked cold_start_train_samples +
        validation_samples must fit the real ImageNet-R train.json, and the
        validation slice must be non-empty. v5 set cold_start = the full
        23998 records, making records[23998:24254] empty; task0 then
        committed 0 experts via below_tau on a 0-sample validation."""
        import yaml

        from compose.experiments import v6_task1_dry_run as runner

        with open(FORMAL_CONFIG, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        task0 = config["tasks"][0]
        seq0 = config["task_sequence"][0]
        with open(seq0["train_instructions"], "r", encoding="utf-8") as handle:
            records = json.load(handle)
        cold_start = task0["cold_start_train_samples"]
        val_count = task0["validation_samples"]
        train_ids, val_ids = runner._draw_train_val_splits(
            records, cold_start, val_count
        )
        assert len(train_ids) == cold_start
        assert len(val_ids) == val_count
        # fix 9 (split bias): the old positional tail slice held 2 of 200
        # classes and 61% of its samples came from a class never seen in
        # training, manufacturing below_tau. The seeded shuffle must give
        # a representative slice: many classes, no validation class absent
        # from training, no class dominating the slice.
        def _cls(record_id):
            return record_id.split("/")[0]

        val_classes = {_cls(value) for value in val_ids}
        train_classes = {_cls(value) for value in train_ids}
        assert len(val_classes) >= 120  # measured 137/200 with seed 42
        assert val_classes - train_classes == set()
        val_counts = {}
        for value in val_ids:
            val_counts[_cls(value)] = val_counts.get(_cls(value), 0) + 1
        assert max(val_counts.values()) <= 0.10 * val_count  # measured 7/256
        # determinism: the split is a pure function of the seed
        again, _ = runner._draw_train_val_splits(records, cold_start, val_count)
        assert again == train_ids
        other, _ = runner._draw_train_val_splits(records, cold_start, val_count, seed=43)
        assert other != train_ids

    def test_task2_teacher_search_selections_empty_registry(self):
        """The task-1 runner's S1 selections must not crash on an empty
        registry (seed-42 task0 committed 0 experts, below_tau): only the
        empty baseline is evaluated."""
        from compose.experiments import v6_task2_dry_run as runner

        records = [{"id": 1}, {"id": 2}]
        assert runner._teacher_search_selections(records, []) == {
            "1": {"empty": []},
            "2": {"empty": []},
        }
        assert runner._teacher_search_selections(records, [10]) == {
            "1": {"empty": [], "single_10": [10]},
            "2": {"empty": [], "single_10": [10]},
        }

    def test_pool_checkpoint_resolution_no_fallback_to_rejected(self, tmp_path):
        """R3/R9: the scoring pool is the snapshot's pool_checkpoint_dir or
        the base-only pool — never a last-known pointer or a cold-start
        candidate. An empty active registry ALWAYS scores the base-only
        pool even when a rejected candidate's pool exists on disk."""
        from compose.experiments import v6_task_run as runner
        from compose.experiments.v6_snapshot import V6Snapshot
        from compose.experts.metadata import ExpertLifecycleStatus, ExpertMetadata
        from compose.experts.registry import ExpertRegistry
        from compose.experts.task_state import TaskStage, TaskStateMachine

        # Rejected candidate pool exists on disk (the old fallback target).
        rejected_pool = tmp_path / "prev" / "candidate" / "cold_start"
        rejected_pool.mkdir(parents=True)
        (rejected_pool / "compose_experts.json").write_text("{}", encoding="utf-8")

        registry = ExpertRegistry()
        machine = TaskStateMachine(1, "ArxivQA")
        machine.advance(TaskStage.DATA_READY, note="mock")
        snap_dir = tmp_path / "prev" / "snapshots" / "task1"
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
        snapshot = V6Snapshot.load(str(snap_dir))

        # Empty active registry -> base-only pool, rejected weights ignored.
        pool = runner._resolve_pool_checkpoint_dir(snapshot, tmp_path / "prev", registry)
        assert pool == tmp_path / "prev" / "candidate" / "base_only"
        assert (pool / "compose_experts.json").is_file()

        # A declared pool dir is honored only with a non-empty active
        # registry; an empty active registry always means base-only (the
        # pointer could be a pre-fix chain into rejected weights).
        declared = tmp_path / "prev" / "candidate" / "train"
        declared.mkdir(parents=True)
        (declared / "compose_experts.json").write_text("{}", encoding="utf-8")
        active_registry = ExpertRegistry()
        for expert_id in (20, 21):
            metadata = ExpertMetadata(
                expert_id=expert_id,
                adapter_name="e{:04d}".format(expert_id),
                rank=8,
                alpha=16.0,
                creation_task=1,
                creation_task_name="ArxivQA",
                created_seed=42,
                checkpoint_path=str(
                    tmp_path / "prev" / "committed"
                    / "expert_{:04d}".format(expert_id) / "compose_experts.bin"
                ),
                checkpoint_sha256="c" * 64,
                lifecycle_status=ExpertLifecycleStatus.CANDIDATE,
            )
            active_registry.register(metadata)
            active_registry.mark_provisional(
                expert_id, {"task_id": 1, "support_count": 20,
                            "mean_conditional_gain": 0.2}
            )
        V6Snapshot.create(
            str(tmp_path / "snap2"),
            task_id=1,
            task_name="ArxivQA",
            registry=active_registry,
            task_state=machine,
            git_commit="mock",
            command="mock",
            data_hash="mock",
            pool_checkpoint_dir=str(declared),
        )
        snapshot2 = V6Snapshot.load(str(tmp_path / "snap2"))
        assert (
            runner._resolve_pool_checkpoint_dir(
                snapshot2, tmp_path / "prev", active_registry
            )
            == declared
        )

        # Pre-fix snapshot (no field) with an active expert: resolved from
        # the registry metadata (newest formal expert's checkpoint parent).
        newest_dir = tmp_path / "prev" / "committed" / "expert_0021"
        newest_dir.mkdir(parents=True)
        (newest_dir / "compose_experts.bin").write_bytes(b"")
        resolved = runner._resolve_pool_checkpoint_dir(
            snapshot, tmp_path / "prev", active_registry
        )
        assert resolved == newest_dir

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

    def test_candidate_training_launches_single_gpu(self):
        """The formal run trains on one GPU via direct python (not torchrun):
        measured single-GPU throughput through torchrun was ~4x slower
        (12.6 vs 3.3 s/step) on this stack, and multi-GPU would need DDP
        because DataParallel cannot split the custom ComposeSelection."""
        text = (REPO / "compose/experiments/v6_task1_dry_run.py").read_text()
        assert '"compose.train.train_v6_candidate"' in text
        assert '"torch.distributed.run"' not in text
        text2 = (REPO / "compose/experiments/v6_task2_dry_run.py").read_text()
        assert '"compose.train.train_v6_candidate"' in text2
        assert '"torch.distributed.run"' not in text2
        text3 = (REPO / "compose/experiments/v6_task_run.py").read_text()
        assert '"compose.train.train_v6_candidate"' in text3
        assert '"torch.distributed.run"' not in text3

    def test_dataloader_workers_passed_from_config(self):
        """Runner must forward dataloader_num_workers from the locked config
        to train_v6_candidate (full-data training bottleneck fix)."""
        for path in (
            "compose/experiments/v6_task1_dry_run.py",
            "compose/experiments/v6_task2_dry_run.py",
            "compose/experiments/v6_task_run.py",
        ):
            text = (REPO / path).read_text()
            assert "dataloader-num-workers" in text
            assert "dataloader_num_workers" in text
        config = _load_config()
        assert config["training"]["dataloader_num_workers"] >= 1
        assert (REPO / "compose/train/train_v6_candidate.py").read_text().count(
            "dataloader-num-workers"
        ) >= 1

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
