"""Empty-registry re-bootstrap fix: 15 unit tests + 2 three-task chains.

Covers Stages R1..R10 of the empty-registry absorbing-state fix:

  R1  unified active expert view (lifecycle = provisional | formal)
  R2  rejected candidate artifacts + terminal REJECTED registration
  R3  no rejected/default adapter fallback (empty selection = backbone)
  R4  base-only pool checkpoint (empty manifest, loadable, real NLL)
  R5  residual split under an empty registry (base_only_mode, same
      gain-floor threshold, never a new threshold)
  R6  re-bootstrap = same candidate trainer without the old-expert flag
  R8  Router with pool=0 (0 keys, save/load, never rejected keys)
  R9  snapshot manifest lifecycle fields + pool_checkpoint_dir
  R10 state machine NO_EXPANSION_REQUIRED (reason insufficient_residual)

Integration chains simulate the runner decision flow at module level
(no GPU): Task0 FAIL -> Task1 PASS -> Task2 normal, and Task0 FAIL ->
Task1 FAIL (re-bootstrap rejects again) -> Task2 re-bootstrap PASS.
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from compose.adapters import ExpertManager, inject_compose_adapters
from compose.config import ComposeAdapterConfig
from compose.experiments.v6_snapshot import V6Snapshot
from compose.experts import ExpertPool
from compose.experts.checkpoint import load_expert_checkpoint
from compose.experts.metadata import (
    ACTIVE_LIFECYCLE_STATUSES,
    ExpertLifecycleStatus,
    ExpertMetadata,
)
from compose.experts.registry import ExpertRegistry
from compose.experts.task_state import TaskStage, TaskStateMachine
from compose.experts.transaction import CommitTransaction
from compose.expansion.v6_base_pool import (
    WEIGHTS_NAME,
    decoder_layer_names,
    write_base_only_checkpoint,
)
from compose.expansion.v6_rejected import (
    REJECTION_REASON_BELOW_TAU,
    register_rejected_candidate,
    rejected_candidate_dir,
    write_rejected_candidate,
)
from compose.expansion.v6_residual import (
    RESIDUAL_REASON_BASE_ONLY,
    RESIDUAL_REASON_EMPTY_TEACHER,
    RESIDUAL_REASON_GAIN_FLOOR,
    RESIDUAL_REASON_SUFFICIENT,
    V6ResidualRecord,
    build_residual_records,
    is_residual,
    should_create_candidates,
)
from compose.router.v6_router import (
    V6QueryEncoder,
    V6Router,
    load_v6_router_checkpoint,
    save_v6_router_checkpoint,
)
from compose.teacher.v6_teacher import V6TeacherRecord

from test_injection import TinyModel


def _teacher_record(
    sample_id: str,
    teacher_set=(),
    empty_loss: float = 5.0,
    teacher_loss: float = 5.0,
) -> V6TeacherRecord:
    return V6TeacherRecord(
        sample_id=sample_id,
        task_id=1,
        pool_version=1,
        router_version="v6_router_v1",
        candidate_experts=tuple(teacher_set),
        empty_loss=empty_loss,
        single_losses={int(e): teacher_loss for e in teacher_set},
        pair_losses={},
        best_single=tuple(teacher_set),
        best_pair=(),
        pair_gain=None,
        teacher_set=tuple(teacher_set),
        teacher_loss=teacher_loss,
        teacher_multi_hot={int(e): 1 for e in teacher_set},
        cache_key="k_" + sample_id,
    )


def _registry_with_active(expert_id: int) -> ExpertRegistry:
    registry = ExpertRegistry()
    metadata = ExpertMetadata(
        expert_id=expert_id,
        adapter_name="expert_{:04d}".format(expert_id),
        rank=8,
        alpha=16.0,
        creation_task=0,
        creation_task_name="ImageNet-R",
        created_seed=42,
        checkpoint_path="/tmp/expert_{:04d}/compose_experts.bin".format(expert_id),
        checkpoint_sha256="a" * 64,
        lifecycle_status=ExpertLifecycleStatus.CANDIDATE,
    )
    registry.register(metadata)
    registry.mark_provisional(
        expert_id,
        {"task_id": 0, "support_count": 100, "mean_conditional_gain": 0.3},
    )
    return registry


def _reject_in_registry(registry: ExpertRegistry, expert_id: int) -> None:
    metadata = ExpertMetadata(
        expert_id=expert_id,
        adapter_name="expert_{:04d}".format(expert_id),
        rank=8,
        alpha=16.0,
        creation_task=1,
        creation_task_name="ArxivQA",
        created_seed=42,
        checkpoint_path="/tmp/rejected_expert_{:04d}/compose_experts.bin".format(expert_id),
        checkpoint_sha256="b" * 64,
        lifecycle_status=ExpertLifecycleStatus.CANDIDATE,
    )
    registry.register(metadata)
    registry.mark_rejected(expert_id, {"task_id": 1, "reason": "below_tau"})


def _model_and_pool():
    torch.manual_seed(7)
    model = TinyModel(layer_count=32)
    inject_compose_adapters(model, ComposeAdapterConfig(rank=1, alpha=2))
    return model, ExpertPool(ExpertManager(model))


class RegistryActiveViewTest(unittest.TestCase):
    """R1: the formal pool is defined by lifecycle, never by disk state."""

    def test_active_view_is_exactly_provisional_and_formal(self):
        registry = ExpertRegistry()
        statuses = [ExpertLifecycleStatus.CANDIDATE,
                    ExpertLifecycleStatus.PROVISIONAL,
                    ExpertLifecycleStatus.FORMAL,
                    ExpertLifecycleStatus.ARCHIVED,
                    ExpertLifecycleStatus.REJECTED]
        for index, status in enumerate(statuses):
            metadata = ExpertMetadata(
                expert_id=index,
                adapter_name="e{:02d}".format(index),
                rank=8,
                alpha=16.0,
                creation_task=0,
                creation_task_name="t",
                checkpoint_path="/tmp/e{:02d}.bin".format(index),
                lifecycle_status=ExpertLifecycleStatus.CANDIDATE,
            )
            registry.register(metadata)
            if status is ExpertLifecycleStatus.PROVISIONAL:
                registry.mark_provisional(index, {"task_id": 0})
            elif status is ExpertLifecycleStatus.FORMAL:
                registry.mark_provisional(index, {"task_id": 0})
                registry.mark_formal(index, {"task_id": 0})
            elif status is ExpertLifecycleStatus.REJECTED:
                registry.mark_rejected(index, {"task_id": 0, "reason": "below_tau"})
            elif status is ExpertLifecycleStatus.ARCHIVED:
                registry.archive(index)
        self.assertEqual(
            sorted(registry.active_lifecycle_ids()),
            [1, 2],
            "only provisional/formal lifecycle experts are active",
        )
        self.assertEqual(
            {e.lifecycle_status for e in registry.get_active_experts()},
            set(ACTIVE_LIFECYCLE_STATUSES),
        )
        self.assertEqual(
            [e.expert_id for e in registry.get_rejected_candidates()],
            [4],
        )
        # A candidate with a checkpoint on disk must NOT become active.
        self.assertEqual(len(registry.get_active_experts()), 2)

    def test_registry_state_dict_round_trip_lifecycle_fields(self):
        registry = _registry_with_active(20)
        _reject_in_registry(registry, 30)
        state = registry.state_dict()
        self.assertEqual(sorted(state["active_lifecycle_ids"]), [20])
        self.assertEqual(sorted(state["rejected_candidate_ids"]), [30])
        reloaded = ExpertRegistry()
        reloaded.load_state_dict(state)
        self.assertEqual(sorted(reloaded.active_lifecycle_ids()), [20])
        self.assertEqual(
            [e.expert_id for e in reloaded.get_rejected_candidates()], [30]
        )

    def test_registry_state_dict_rejects_active_lifecycle_mismatch(self):
        registry = _registry_with_active(20)
        state = registry.state_dict()
        state["active_lifecycle_ids"] = [99]
        with self.assertRaises(ValueError):
            ExpertRegistry().load_state_dict(state)


class RejectedIsolationTest(unittest.TestCase):
    """R2: rejected candidates are diagnostics-only, never active."""

    def test_register_rejected_never_enters_active_and_preserves_pool_version(self):
        registry = ExpertRegistry()
        version_before = registry.pool_version
        register_rejected_candidate(
            registry,
            expert_id=10,
            task_id=0,
            task_name="ImageNet-R",
            seed=42,
            reason=REJECTION_REASON_BELOW_TAU,
            mean_gain=-0.249,
            support=7,
            checkpoint_path="/tmp/rejected_expert_0010/compose_experts.bin",
            checkpoint_sha256="c" * 64,
            commit_thresholds={"tau_support": 8, "tau_gain": 0.0},
        )
        self.assertEqual(registry.pool_version, version_before,
                         "rejection must not bump pool_version")
        self.assertEqual(registry.get_active_experts(), [])
        self.assertEqual(registry.get_rejected_candidates()[0].expert_id, 10)
        self.assertTrue(registry.contains(10))
        with self.assertRaises(ValueError):
            register_rejected_candidate(
                registry,
                expert_id=10,
                task_id=0,
                task_name="ImageNet-R",
                seed=42,
                reason=REJECTION_REASON_BELOW_TAU,
                mean_gain=-0.249,
                support=7,
                checkpoint_path="/tmp/x.bin",
                checkpoint_sha256="d" * 64,
                commit_thresholds={"tau_support": 8, "tau_gain": 0.0},
            )

    def test_write_rejected_candidate_artifact_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # A fake candidate training pool (manifest + weights).
            adapter = root / "candidate" / "train"
            adapter.mkdir(parents=True)
            (adapter / "compose_experts.json").write_text(
                json.dumps({"format_version": 1, "experts": []}), encoding="utf-8"
            )
            torch.save({"fake": torch.tensor([1.0])}, str(adapter / "compose_experts.bin"))
            rejection = write_rejected_candidate(
                root,
                task_id=0,
                task_name="ImageNet-R",
                candidate_id=10,
                reason=REJECTION_REASON_BELOW_TAU,
                mean_gain=-0.249,
                support=7,
                validation_stats={"samples": 256, "mean_gain": -0.249,
                                  "support_count": 7},
                commit_thresholds={"tau_support": 8, "tau_gain": 0.0},
                adapter_dir=str(adapter),
                seed=42,
                pool_version=1,
                config_hash="formal",
            )
            dest = rejected_candidate_dir(root, 0, 10)
            self.assertTrue((dest / "adapter" / "compose_experts.json").is_file())
            self.assertTrue((dest / "adapter" / "compose_experts.bin").is_file())
            self.assertTrue((dest / "validation.json").is_file())
            self.assertTrue((dest / "manifest.json").is_file())
            self.assertTrue(rejection["excluded_from_active_pool"])
            self.assertEqual(rejection["status"], "rejected")
            self.assertEqual(rejection["reason"], REJECTION_REASON_BELOW_TAU)
            self.assertTrue(len(rejection["checkpoint_sha256"]) == 64)
            # The rejection dir must live outside candidate/train.
            self.assertNotIn("candidate/train", str(dest))

    def test_rejected_lifecycle_is_terminal(self):
        registry = ExpertRegistry()
        metadata = ExpertMetadata(
            expert_id=10,
            adapter_name="e10",
            rank=8,
            alpha=16.0,
            creation_task=0,
            creation_task_name="t",
            lifecycle_status=ExpertLifecycleStatus.CANDIDATE,
        )
        registry.register(metadata)
        registry.mark_rejected(10, {"task_id": 0, "reason": "below_tau"})
        with self.assertRaises(ValueError):
            registry.mark_provisional(10, {"task_id": 0})
        with self.assertRaises(ValueError):
            registry.mark_rejected(10, {"task_id": 0, "reason": "again"})
        self.assertEqual(
            registry.get(10).lifecycle_status, ExpertLifecycleStatus.REJECTED
        )


class BaseOnlyPoolTest(unittest.TestCase):
    """R4: empty pool checkpoint = frozen backbone, loadable, zero experts."""

    def test_base_only_checkpoint_loads_on_empty_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            model, pool = _model_and_pool()
            output = write_base_only_checkpoint(
                directory, layer_names=sorted(pool.manager.layers)
            )
            manifest = load_expert_checkpoint(pool, str(output))
            self.assertEqual(manifest["kind"], "base_only_pool")
            self.assertEqual(manifest["experts"], [])
            self.assertEqual(manifest["metrics"]["adapter_tensor_count"], 0)
            self.assertEqual(manifest["load_summary"]["loaded_tensor_count"], 0)
            self.assertEqual(pool.expert_ids(), [])
            # Empty selection = backbone-only forward.
            pool.manager.clear_default_selection()
            inputs = torch.randn(2, 4, 3)
            model.model.layers[0].self_attn.o_proj(inputs)
            model.model.layers[0].mlp.down_proj(inputs)

    def test_base_only_manifest_has_224_decoder_layer_names(self):
        names = decoder_layer_names()
        self.assertEqual(len(names), 32 * 7)
        self.assertEqual(names, sorted(names))
        self.assertIn("model.layers.0.self_attn.q_proj", names)
        self.assertIn("model.layers.31.mlp.down_proj", names)
        with tempfile.TemporaryDirectory() as directory:
            output = write_base_only_checkpoint(directory)
            manifest = json.loads(
                (Path(directory) / "compose_experts.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["kind"], "base_only_pool")
            self.assertEqual(manifest["experts"], [])
            self.assertEqual(manifest["adapter"]["layers"], names)
            self.assertEqual(manifest["metrics"]["adapter_parameter_count"], 0)
            self.assertTrue((Path(directory) / WEIGHTS_NAME).is_file())


class ResidualBaseOnlyTest(unittest.TestCase):
    """R5: empty registry -> residual decision still runs with the SAME
    gain-floor threshold (never a new threshold, never ``residual = []``)."""

    def test_is_residual_base_only_mode(self):
        record = _teacher_record("s1", teacher_set=(), empty_loss=5.0,
                                 teacher_loss=5.0)
        self.assertEqual(
            is_residual(record, min_old_gain=0.02, recall_covered=True,
                        base_only_mode=True),
            (True, RESIDUAL_REASON_BASE_ONLY),
        )
        self.assertEqual(
            is_residual(record, min_old_gain=0.02, recall_covered=True,
                        base_only_mode=False),
            (False, RESIDUAL_REASON_EMPTY_TEACHER),
        )

    def test_is_residual_reuses_gain_floor_unchanged(self):
        # old_gain = empty - teacher = 5.0 - 4.8 = 0.2 >= min_old_gain
        sufficient = _teacher_record("s2", teacher_set=(20,),
                                     empty_loss=5.0, teacher_loss=4.8)
        # old_gain = 5.0 - 4.99 = 0.01 < min_old_gain=0.02
        below_floor = _teacher_record("s3", teacher_set=(20,),
                                      empty_loss=5.0, teacher_loss=4.99)
        for mode in (False, True):
            self.assertEqual(
                is_residual(sufficient, min_old_gain=0.02, recall_covered=True,
                            base_only_mode=mode),
                (False, RESIDUAL_REASON_SUFFICIENT),
            )
            self.assertEqual(
                is_residual(below_floor, min_old_gain=0.02, recall_covered=True,
                            base_only_mode=mode),
                (True, RESIDUAL_REASON_GAIN_FLOOR),
            )

    def test_build_residual_records_base_only_mode_keeps_empty_teacher(self):
        records = [
            _teacher_record("a", teacher_set=(), empty_loss=5.0, teacher_loss=5.0),
            _teacher_record("b", teacher_set=(), empty_loss=4.0, teacher_loss=4.0),
            _teacher_record("c", teacher_set=(20,), empty_loss=5.0, teacher_loss=4.8),
        ]
        reuse, residual = build_residual_records(
            records, "features.json", min_old_gain=0.02,
            top_m_covered=[True, True, True], split="train", base_only_mode=True,
        )
        self.assertEqual(len(residual), 2)
        self.assertEqual(
            {r.sample_id for r in residual}, {"a", "b"},
            "empty-teacher samples are residual candidate material "
            "under an empty registry",
        )
        self.assertTrue(
            all(r.residual_reason == RESIDUAL_REASON_BASE_ONLY for r in residual)
        )
        self.assertTrue(
            all(r.old_gain == 0.0 for r in residual),
            "old_gain = empty_loss - teacher_loss over the base itself",
        )
        self.assertEqual([r.sample_id for r in reuse], ["c"])

    def test_build_residual_records_without_base_only_mode_drops_empties(self):
        records = [
            _teacher_record("a", teacher_set=(), empty_loss=5.0, teacher_loss=5.0),
            _teacher_record("c", teacher_set=(20,), empty_loss=5.0, teacher_loss=4.8),
        ]
        reuse, residual = build_residual_records(
            records, "features.json", min_old_gain=0.02,
            top_m_covered=[True, True], split="train", base_only_mode=False,
        )
        self.assertEqual(residual, [])
        self.assertEqual([r.sample_id for r in reuse], ["c"])

    def test_should_create_candidates_threshold_never_lowered(self):
        self.assertFalse(should_create_candidates(7, min_residual_samples=8))
        self.assertTrue(should_create_candidates(8, min_residual_samples=8))
        self.assertTrue(should_create_candidates(256, min_residual_samples=8))

    def test_residual_record_accepts_base_only_reason(self):
        record = V6ResidualRecord(
            sample_id="x",
            task_id=1,
            old_teacher_set=(),
            empty_loss=5.0,
            old_teacher_loss=5.0,
            old_gain=0.0,
            residual_reason=RESIDUAL_REASON_BASE_ONLY,
            query_feature_path="f.json",
            split="train",
        )
        self.assertEqual(record.residual_reason, RESIDUAL_REASON_BASE_ONLY)


class StateMachineNoExpansionTest(unittest.TestCase):
    """R10: NO_EXPANSION_REQUIRED with reason insufficient_residual."""

    def test_residual_ready_to_no_expansion_to_global_teacher(self):
        machine = TaskStateMachine(2, "VizWiz")
        machine.advance(TaskStage.DATA_READY)
        machine.advance(TaskStage.OLD_TEACHER_RUNNING)
        machine.advance(TaskStage.OLD_TEACHER_READY)
        machine.advance(TaskStage.RESIDUAL_READY)
        machine.advance(TaskStage.NO_EXPANSION_REQUIRED,
                        note="insufficient_residual")
        self.assertEqual(machine.stage, TaskStage.NO_EXPANSION_REQUIRED)
        machine.advance(TaskStage.GLOBAL_TEACHER_READY)
        self.assertEqual(machine.stage, TaskStage.GLOBAL_TEACHER_READY)

    def test_residual_ready_skipping_straight_to_global_teacher_is_illegal(self):
        machine = TaskStateMachine(2, "VizWiz")
        machine.advance(TaskStage.DATA_READY)
        machine.advance(TaskStage.OLD_TEACHER_RUNNING)
        machine.advance(TaskStage.OLD_TEACHER_READY)
        machine.advance(TaskStage.RESIDUAL_READY)
        with self.assertRaises(ValueError):
            machine.advance(TaskStage.GLOBAL_TEACHER_READY)

    def test_no_expansion_is_not_reachable_from_candidate_training(self):
        machine = TaskStateMachine(2, "VizWiz")
        machine.advance(TaskStage.DATA_READY)
        machine.advance(TaskStage.OLD_TEACHER_RUNNING)
        machine.advance(TaskStage.OLD_TEACHER_READY)
        machine.advance(TaskStage.RESIDUAL_READY)
        machine.advance(TaskStage.CANDIDATE_TRAINING)
        with self.assertRaises(ValueError):
            machine.advance(TaskStage.NO_EXPANSION_REQUIRED)


class SnapshotLifecycleFieldsTest(unittest.TestCase):
    """R9: snapshots carry the formal-pool view for the next task."""

    def _create_snapshot(self, directory, registry, pool_dir):
        machine = TaskStateMachine(0, "ImageNet-R")
        machine.advance(TaskStage.DATA_READY)
        machine.advance(TaskStage.OLD_TEACHER_RUNNING)
        machine.advance(TaskStage.OLD_TEACHER_READY)
        machine.advance(TaskStage.RESIDUAL_READY)
        machine.advance(TaskStage.NO_EXPANSION_REQUIRED, note="insufficient_residual")
        return V6Snapshot.create(
            directory,
            task_id=0,
            task_name="ImageNet-R",
            registry=registry,
            task_state=machine,
            git_commit="abc123",
            command="test",
            data_hash="deadbeef",
            pool_checkpoint_dir=pool_dir,
        )

    def test_empty_registry_snapshot_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool_dir = str(root / "candidate" / "base_only")
            (Path(pool_dir)).mkdir(parents=True)
            registry = ExpertRegistry()
            _reject_in_registry(registry, 10)
            snapshot = self._create_snapshot(str(root / "snap"), registry, pool_dir)
            self.assertEqual(snapshot.manifest["active_expert_ids"], [])
            self.assertEqual(snapshot.manifest["rejected_candidate_ids"], [10])
            self.assertTrue(snapshot.manifest["rebootstrap_allowed"])
            self.assertEqual(snapshot.manifest["pool_checkpoint_dir"], pool_dir)

    def test_active_registry_snapshot_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool_dir = str(root / "candidate" / "train")
            (Path(pool_dir)).mkdir(parents=True)
            registry = _registry_with_active(20)
            snapshot = self._create_snapshot(str(root / "snap"), registry, pool_dir)
            self.assertEqual(snapshot.manifest["active_expert_ids"], [20])
            self.assertFalse(snapshot.manifest["rebootstrap_allowed"])
            reloaded = V6Snapshot.load(str(root / "snap"))
            self.assertEqual(
                reloaded.registry.active_lifecycle_ids(), (20,),
                "snapshot round trip preserves the lifecycle-derived view",
            )


class RouterEmptyPoolTest(unittest.TestCase):
    """R8: Router with pool=0 carries zero keys and round-trips."""

    def test_router_zero_experts_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "router.pt")
            router = V6Router(V6QueryEncoder(), seed=42)
            self.assertEqual(router.expert_ids, ())
            save_v6_router_checkpoint(
                path, router, pool_version=1, config_hash="formal",
                extra={"task_id": 1, "mode": "calibrated"},
            )
            reloaded = V6Router(V6QueryEncoder(), seed=42)
            extra = load_v6_router_checkpoint(path, reloaded)
            self.assertEqual(reloaded.expert_ids, ())
            self.assertEqual(extra["task_id"], 1)


class ThreeTaskChainIntegrationTest(unittest.TestCase):
    """3-task integration: state + registry + snapshot chaining (no GPU).

    Path A: Task0 FAIL -> Task1 PASS (re-bootstrap) -> Task2 normal.
    Path B: Task0 FAIL -> Task1 FAIL (re-bootstrap rejects again) ->
            Task2 re-bootstrap PASS.
    """

    @staticmethod
    def _task0_fail_snapshot(root: Path, registry: ExpertRegistry) -> str:
        base_dir = root / "task0" / "candidate" / "base_only"
        base_dir.mkdir(parents=True)
        write_base_only_checkpoint(str(base_dir))
        machine = TaskStateMachine(0, "ImageNet-R")
        machine.advance(TaskStage.DATA_READY)
        machine.advance(TaskStage.OLD_TEACHER_RUNNING)
        machine.advance(TaskStage.OLD_TEACHER_READY)
        machine.advance(TaskStage.RESIDUAL_READY)
        machine.advance(TaskStage.NO_EXPANSION_REQUIRED, note="insufficient_residual")
        snapshot_dir = str(root / "task0" / "snapshots" / "task0")
        V6Snapshot.create(
            snapshot_dir,
            task_id=0, task_name="ImageNet-R", registry=registry,
            task_state=machine, git_commit="t0", command="test",
            data_hash="h0", pool_checkpoint_dir=str(base_dir),
        )
        return snapshot_dir

    @staticmethod
    def _task1_view(
        root: Path, snapshot_dir: str
    ):
        """Simulates v6_task2_dry_run S0-S2 + commit for an empty registry."""
        snapshot = V6Snapshot.load(snapshot_dir)
        registry = ExpertRegistry()
        registry.load_state_dict(snapshot.registry.state_dict())
        active_ids = list(registry.active_lifecycle_ids())
        base_only_mode = not active_ids
        # S1: base-only teacher scoring -> empty teachers with real base NLL.
        records = [
            _teacher_record("t1a", teacher_set=(), empty_loss=5.0, teacher_loss=5.0),
            _teacher_record("t1b", teacher_set=(), empty_loss=4.0, teacher_loss=4.0),
            _teacher_record("t1c", teacher_set=(), empty_loss=6.0, teacher_loss=6.0),
        ]
        reuse, residual = build_residual_records(
            records, "features.json", min_old_gain=0.02,
            top_m_covered=[True] * len(records), split="train",
            base_only_mode=base_only_mode,
        )
        should_train = should_create_candidates(
            len(residual), min_residual_samples=8 if not base_only_mode else 2
        )
        return registry, active_ids, base_only_mode, residual, should_train

    def test_chain_task0_fail_task1_pass_task2_normal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry0 = ExpertRegistry()
            # Task0: cold-start candidate rejected (below_tau).
            register_rejected_candidate(
                registry0, expert_id=10, task_id=0, task_name="ImageNet-R",
                seed=42, reason=REJECTION_REASON_BELOW_TAU,
                mean_gain=-0.249, support=7,
                checkpoint_path="/tmp/r10/compose_experts.bin",
                checkpoint_sha256="e" * 64,
                commit_thresholds={"tau_support": 8, "tau_gain": 0.0},
            )
            snapshot0 = self._task0_fail_snapshot(root, registry0)

            # Task1: empty active registry -> base-only teacher -> residual
            # exists -> re-bootstrap -> commit PASSES.
            registry1, active1, base_only1, residual1, train1 = self._task1_view(
                root, snapshot0
            )
            self.assertEqual(active1, [])
            self.assertTrue(base_only1)
            self.assertTrue(residual1, "empty registry must still produce residual")
            self.assertTrue(train1)
            # Commit: both slots pass tau.
            for slot in (20, 21):
                transaction = CommitTransaction(
                    str(root / "task1" / "state"), registry1
                )
                transaction.begin(slot, {"task_id": 1, "stage": "task1_commit"})
                transaction.complete(
                    slot,
                    artifacts={},
                    condition_record={"task_id": 1, "support_count": 20,
                                     "mean_conditional_gain": 0.25},
                    metadata=ExpertMetadata(
                        expert_id=slot,
                        adapter_name="expert_{:04d}".format(slot),
                        rank=8, alpha=16.0,
                        creation_task=1, creation_task_name="ArxivQA",
                        created_seed=42,
                        checkpoint_path="/tmp/e{:04d}/compose_experts.bin".format(slot),
                        checkpoint_sha256="f" * 64,
                        lifecycle_status=ExpertLifecycleStatus.CANDIDATE,
                        support_count=20,
                        mean_conditional_gain=0.25,
                    ),
                )
            self.assertEqual(sorted(registry1.active_lifecycle_ids()), [20, 21])
            # Task1 snapshot: active pool present.
            machine1 = TaskStateMachine(1, "ArxivQA")
            machine1.advance(TaskStage.DATA_READY)
            machine1.advance(TaskStage.OLD_TEACHER_RUNNING)
            machine1.advance(TaskStage.OLD_TEACHER_READY)
            machine1.advance(TaskStage.RESIDUAL_READY)
            machine1.advance(TaskStage.CANDIDATE_TRAINING)
            machine1.advance(TaskStage.CANDIDATE_TRAINED)
            machine1.advance(TaskStage.CANDIDATE_VALIDATED)
            machine1.advance(TaskStage.EXPERTS_COMMITTED)
            pool1 = str(root / "task1" / "candidate" / "train")
            Path(pool1).mkdir(parents=True)
            (Path(pool1) / "compose_experts.json").write_text(
                json.dumps({"format_version": 1, "experts": []}), encoding="utf-8"
            )
            V6Snapshot.create(
                str(root / "task1" / "snapshots" / "task1"),
                task_id=1, task_name="ArxivQA", registry=registry1,
                task_state=machine1, git_commit="t1", command="test",
                data_hash="h1", pool_checkpoint_dir=pool1,
            )

            # Task2: active pool present -> normal (non-base-only) flow.
            snapshot1 = V6Snapshot.load(str(root / "task1" / "snapshots" / "task1"))
            registry2 = ExpertRegistry()
            registry2.load_state_dict(snapshot1.registry.state_dict())
            active2 = list(registry2.active_lifecycle_ids())
            self.assertEqual(active2, [20, 21])
            self.assertFalse(snapshot1.manifest["rebootstrap_allowed"])
            self.assertEqual(
                snapshot1.manifest["pool_checkpoint_dir"], pool1,
                "next task scores against the propagated pool, not a glob",
            )
            # The rejected task0 candidate never surfaces in the active view.
            all_ids = [e.expert_id for e in registry2.get_all_artifacts()]
            self.assertIn(10, all_ids)
            self.assertNotIn(10, active2)

    def test_chain_task0_fail_task1_fail_task2_rebootstrap_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry0 = ExpertRegistry()
            register_rejected_candidate(
                registry0, expert_id=10, task_id=0, task_name="ImageNet-R",
                seed=42, reason=REJECTION_REASON_BELOW_TAU,
                mean_gain=-0.249, support=7,
                checkpoint_path="/tmp/r10/compose_experts.bin",
                checkpoint_sha256="e" * 64,
                commit_thresholds={"tau_support": 8, "tau_gain": 0.0},
            )
            snapshot0 = self._task0_fail_snapshot(root, registry0)

            # Task1: re-bootstrap runs but both candidates are rejected.
            registry1, active1, base_only1, residual1, train1 = self._task1_view(
                root, snapshot0
            )
            self.assertTrue(base_only1)
            self.assertTrue(residual1)
            for slot in (20, 21):
                register_rejected_candidate(
                    registry1, expert_id=slot, task_id=1, task_name="ArxivQA",
                    seed=42, reason=REJECTION_REASON_BELOW_TAU,
                    mean_gain=-0.05, support=3,
                    checkpoint_path="/tmp/r{}/compose_experts.bin".format(slot),
                    checkpoint_sha256="g" * 64,
                    commit_thresholds={"tau_support": 8, "tau_gain": 0.0},
                )
            self.assertEqual(registry1.get_active_experts(), [])
            self.assertEqual(
                sorted(e.expert_id for e in registry1.get_rejected_candidates()),
                [10, 20, 21],
            )
            # Task1 snapshot: still rebootstrap_allowed, pool = base-only.
            machine1 = TaskStateMachine(1, "ArxivQA")
            machine1.advance(TaskStage.DATA_READY)
            machine1.advance(TaskStage.OLD_TEACHER_RUNNING)
            machine1.advance(TaskStage.OLD_TEACHER_READY)
            machine1.advance(TaskStage.RESIDUAL_READY)
            machine1.advance(TaskStage.NO_EXPANSION_REQUIRED,
                            note="insufficient_residual")
            base1 = root / "task1" / "candidate" / "base_only"
            base1.mkdir(parents=True)
            write_base_only_checkpoint(str(base1))
            V6Snapshot.create(
                str(root / "task1" / "snapshots" / "task1"),
                task_id=1, task_name="ArxivQA", registry=registry1,
                task_state=machine1, git_commit="t1", command="test",
                data_hash="h1", pool_checkpoint_dir=str(base1),
            )

            # Task2: empty registry again -> re-bootstrap allowed -> PASS.
            snapshot1 = V6Snapshot.load(str(root / "task1" / "snapshots" / "task1"))
            registry2 = ExpertRegistry()
            registry2.load_state_dict(snapshot1.registry.state_dict())
            self.assertEqual(list(registry2.active_lifecycle_ids()), [])
            self.assertTrue(snapshot1.manifest["rebootstrap_allowed"])
            self.assertEqual(
                snapshot1.manifest["pool_checkpoint_dir"], str(base1),
                "repeated re-bootstrap: still the base-only pool, no rejected "
                "weights propagated",
            )
            # Task2 re-bootstrap commits.
            for slot in (30, 31):
                transaction = CommitTransaction(
                    str(root / "task2" / "state"), registry2
                )
                transaction.begin(slot, {"task_id": 2, "stage": "task2_commit"})
                transaction.complete(
                    slot,
                    artifacts={},
                    condition_record={"task_id": 2, "support_count": 25,
                                     "mean_conditional_gain": 0.31},
                    metadata=ExpertMetadata(
                        expert_id=slot,
                        adapter_name="expert_{:04d}".format(slot),
                        rank=8, alpha=16.0,
                        creation_task=2, creation_task_name="VizWiz",
                        created_seed=42,
                        checkpoint_path="/tmp/e{:04d}/compose_experts.bin".format(slot),
                        checkpoint_sha256="h" * 64,
                        lifecycle_status=ExpertLifecycleStatus.CANDIDATE,
                        support_count=25,
                        mean_conditional_gain=0.31,
                    ),
                )
            self.assertEqual(sorted(registry2.active_lifecycle_ids()), [30, 31])
            # Rejected ids from task0/task1 remain excluded from the active view.
            self.assertNotIn(10, registry2.active_lifecycle_ids())
            self.assertNotIn(20, registry2.active_lifecycle_ids())
            self.assertNotIn(21, registry2.active_lifecycle_ids())


if __name__ == "__main__":
    unittest.main()
