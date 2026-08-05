"""V6 Stage E10: task-boundary snapshots, independent load, resume nodes."""

import tempfile
import unittest

import torch
from pathlib import Path

from compose.experts.metadata import ExpertLifecycleStatus, ExpertMetadata
from compose.experts.registry import ExpertRegistry
from compose.experts.task_state import TaskStage, TaskStateMachine
from compose.experts.transaction import CommitTransaction
from compose.experiments.v6_snapshot import (
    RESUME_NODES,
    V6Snapshot,
    analyze_resume,
)


def _registry(with_expert=False, pool_version=1):
    registry = ExpertRegistry()
    if with_expert:
        metadata = ExpertMetadata(
            expert_id=0,
            adapter_name="expert_0000",
            rank=8,
            alpha=16.0,
            creation_task=0,
            creation_task_name="ImageNet-R",
            created_seed=42,
            checkpoint_path="/ckpt/compose_experts.bin",
            checkpoint_sha256="a" * 64,
            lifecycle_status=ExpertLifecycleStatus.PROVISIONAL,
        )
        registry.register(metadata)
    for _ in range(pool_version - 1):
        registry.increment_pool_version()
    return registry


def _task_state(stage=TaskStage.DATA_READY):
    machine = TaskStateMachine(1, "ArxivQA")
    path = [
        TaskStage.DATA_READY,
        TaskStage.OLD_TEACHER_RUNNING,
        TaskStage.OLD_TEACHER_READY,
        TaskStage.RESIDUAL_READY,
        TaskStage.CANDIDATE_TRAINING,
        TaskStage.CANDIDATE_TRAINED,
        TaskStage.CANDIDATE_VALIDATED,
        TaskStage.EXPERTS_COMMITTED,
        TaskStage.GLOBAL_TEACHER_READY,
        TaskStage.ROUTER_TRAINING,
        TaskStage.ROUTER_READY,
        TaskStage.RMS_READY,
        TaskStage.SNAPSHOT_READY,
        TaskStage.EVALUATION_COMPLETE,
        TaskStage.COMPLETED,
    ]
    if stage not in path:
        raise ValueError("unknown stage {}".format(stage))
    for target in path:
        machine.advance(target)
        if target is stage:
            break
    return machine


class SnapshotCreateLoadTest(unittest.TestCase):
    def test_create_and_independent_load(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = _registry(with_expert=True, pool_version=3)
            task_state = _task_state(TaskStage.OLD_TEACHER_READY)
            snapshot = V6Snapshot.create(
                str(Path(directory) / "snap"),
                task_id=1, task_name="ArxivQA",
                registry=registry, task_state=task_state,
                git_commit="abc123", command="run --seed 42",
                data_hash="data-hash-1",
                config_copy_path="/nonexistent/config.yaml",
                teacher_cache_manifest={"keys": ["k1"]},
                residual_manifest={"samples": ["s1"]},
                candidate_validation={"decisions": []},
                stdout_text="hello out", stderr_text="hello err",
            )
            self.assertEqual(snapshot.registry.pool_version, 3)
            self.assertEqual(snapshot.task_state.stage, TaskStage.OLD_TEACHER_READY)
            # Delete the registry dir used at save time? No: the snapshot
            # must load from its own directory alone.
            loaded = V6Snapshot.load(str(Path(directory) / "snap"))
            self.assertEqual(loaded.manifest["task_name"], "ArxivQA")
            self.assertEqual(loaded.manifest["git_commit"], "abc123")
            self.assertEqual(loaded.registry.pool_version, 3)
            self.assertTrue(loaded.registry.contains(0))
            self.assertEqual(
                loaded.registry.get(0).lifecycle_status,
                ExpertLifecycleStatus.PROVISIONAL,
            )
            self.assertEqual(loaded.task_state.stage, TaskStage.OLD_TEACHER_READY)
            self.assertEqual(loaded.teacher_cache_manifest, {"keys": ["k1"]})
            self.assertEqual(loaded.residual_manifest, {"samples": ["s1"]})
            self.assertEqual(loaded.candidate_validation, {"decisions": []})
            self.assertTrue((loaded.path / "stdout.log").is_file())
            self.assertTrue((loaded.path / "stderr.log").is_file())
            self.assertEqual(loaded.verify_complete(), [])
            # Random states are JSON-free torch payloads; restore works.
            from compose.experiments.v6_snapshot import _restore_random_states

            _restore_random_states(loaded.random_states)
            self.assertIsNotNone(torch.get_rng_state())

    def test_create_refuses_nonempty_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "snap"
            target.mkdir()
            (target / "junk").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                V6Snapshot.create(
                    str(target), 0, "ImageNet-R",
                    _registry(), _task_state(TaskStage.DATA_READY),
                    git_commit="c", command="cmd", data_hash="d",
                )

    def test_load_rejects_incomplete_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = _registry()
            task_state = _task_state(TaskStage.DATA_READY)
            snapshot = V6Snapshot.create(
                str(Path(directory) / "snap"), 0, "ImageNet-R",
                registry, task_state, git_commit="c", command="cmd", data_hash="d",
            )
            (snapshot.path / "task_state.json").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                V6Snapshot.load(str(Path(directory) / "snap"))

    def test_load_rejects_tampered_component(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = V6Snapshot.create(
                str(Path(directory) / "snap"), 0, "ImageNet-R",
                _registry(with_expert=True), _task_state(TaskStage.DATA_READY),
                git_commit="c", command="cmd", data_hash="d",
            )
            (snapshot.path / "expert_registry.json").write_text(
                '{"tampered": true}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                V6Snapshot.load(str(Path(directory) / "snap"))

    def test_unknown_schema_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = V6Snapshot.create(
                str(Path(directory) / "snap"), 0, "ImageNet-R",
                _registry(), _task_state(TaskStage.DATA_READY),
                git_commit="c", command="cmd", data_hash="d",
            )
            manifest_path = snapshot.path / "manifest.json"
            manifest = Path(manifest_path).read_text(encoding="utf-8").replace(
                '"schema_version": 1', '"schema_version": 99'
            )
            manifest_path.write_text(manifest, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version"):
                V6Snapshot.load(str(Path(directory) / "snap"))

    def test_expert_hashes_unchanged_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = V6Snapshot.create(
                str(Path(directory) / "snap"), 0, "ImageNet-R",
                _registry(with_expert=True), _task_state(TaskStage.DATA_READY),
                git_commit="c", command="cmd", data_hash="d",
            )
            mismatches = snapshot.verify_expert_hashes_unchanged({0: "a" * 64})
            self.assertEqual(mismatches, [])
            mismatches = snapshot.verify_expert_hashes_unchanged({0: "b" * 64})
            self.assertEqual(len(mismatches), 1)

    def test_resume_nodes_cover_task_book_list(self):
        self.assertEqual(
            RESUME_NODES,
            (
                "old_teacher_running",
                "candidate_epoch_running",
                "candidate_validated_not_committed",
                "candidate_committed",
                "router_calibrating",
                "snapshot_ready_eval_pending",
            ),
        )


class ResumeAnalysisTest(unittest.TestCase):
    def _snapshot(self, stage, directory):
        return V6Snapshot.create(
            str(Path(directory) / "snap"), 1, "ArxivQA",
            _registry(with_expert=True, pool_version=2),
            _task_state(stage), git_commit="c", command="cmd", data_hash="d",
        )

    def test_resume_nodes_mapped_per_stage(self):
        cases = [
            (TaskStage.OLD_TEACHER_RUNNING, "old_teacher_running"),
            (TaskStage.CANDIDATE_TRAINING, "candidate_epoch_running"),
            (TaskStage.CANDIDATE_VALIDATED, "candidate_validated_not_committed"),
            (TaskStage.EXPERTS_COMMITTED, "candidate_committed"),
            (TaskStage.ROUTER_TRAINING, "router_calibrating"),
            (TaskStage.SNAPSHOT_READY, "snapshot_ready_eval_pending"),
        ]
        for stage, expected in cases:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                self._snapshot(stage, directory)
                result = analyze_resume(str(Path(directory) / "snap"))
                self.assertEqual(result["resume_node"], expected)
                self.assertTrue(result["can_resume"])

    def test_completed_task_not_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            self._snapshot(TaskStage.COMPLETED, directory)
            result = analyze_resume(str(Path(directory) / "snap"))
            self.assertEqual(result["resume_node"], "completed")
            self.assertFalse(result["can_resume"])

    def test_pending_transactions_surface_in_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self._snapshot(TaskStage.CANDIDATE_VALIDATED, directory)
            transaction = CommitTransaction(str(snapshot.path), snapshot.registry)
            transaction.begin(77, {"task_id": 1})
            result = analyze_resume(str(Path(directory) / "snap"))
            self.assertIn("77", result["pending_transactions"])
            # Resume cleans nothing for an incomplete transaction (registry
            # has no entry 77); it is reported for rollback.
            self.assertEqual(result["completed_transactions_cleaned"], [])
            self.assertEqual(result["pool_version"], 2)


if __name__ == "__main__":
    unittest.main()
