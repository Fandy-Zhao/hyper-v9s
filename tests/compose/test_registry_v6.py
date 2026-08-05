"""V6 Stage E2: expert lifecycle, pool_version and commit transactions."""

import json
import tempfile
import unittest
from pathlib import Path

from compose.experts.metadata import ExpertLifecycleStatus, ExpertMetadata
from compose.experts.registry import POOL_VERSION_INITIAL, ExpertRegistry
from compose.experts.transaction import CommitTransaction, file_sha256


def _candidate(expert_id: int, **overrides) -> ExpertMetadata:
    fields = dict(
        expert_id=expert_id,
        adapter_name="expert-{}".format(expert_id),
        rank=8,
        alpha=16.0,
        creation_task=0,
        creation_task_name="ImageNet-R",
        created_seed=42,
    )
    fields.update(overrides)
    return ExpertMetadata(**fields)


class ExpertLifecycleV6Test(unittest.TestCase):
    def test_candidate_default_when_uncommitted(self):
        registry = ExpertRegistry()
        metadata = registry.register(_candidate(1))
        self.assertIs(metadata.lifecycle_status, ExpertLifecycleStatus.CANDIDATE)

    def test_provisional_inferred_when_checkpoint_bound(self):
        registry = ExpertRegistry()
        metadata = registry.register(
            _candidate(1, checkpoint_path="/tmp/expert_0001",
                       checkpoint_sha256="a" * 64)
        )
        self.assertIs(metadata.lifecycle_status, ExpertLifecycleStatus.PROVISIONAL)

    def test_candidate_to_provisional_records_conditions(self):
        registry = ExpertRegistry()
        registry.register(_candidate(1))
        registry.mark_provisional(
            1, {"support_count": 12, "mean_conditional_gain": 0.05,
                "key_accuracy": 0.8}
        )
        metadata = registry.get(1)
        self.assertIs(metadata.lifecycle_status, ExpertLifecycleStatus.PROVISIONAL)
        conditions = metadata.extra["lifecycle_conditions"]
        self.assertEqual(conditions["provisional"]["support_count"], 12)

    def test_provisional_to_formal_requires_conditions_record(self):
        registry = ExpertRegistry()
        registry.register(_candidate(1, checkpoint_path="/tmp/e1",
                                       checkpoint_sha256="b" * 64))
        registry.mark_formal(1, {"reason": "survived six-task eval"})
        self.assertIs(registry.get(1).lifecycle_status, ExpertLifecycleStatus.FORMAL)

    def test_illegal_transitions_rejected(self):
        registry = ExpertRegistry()
        registry.register(_candidate(1))
        # candidate -> formal skips provisional.
        with self.assertRaisesRegex(ValueError, "illegal lifecycle transition"):
            registry.mark_formal(1, {})
        # candidate -> rejected is legal (validation failure).
        registry.mark_rejected(1, {"reason": "below tau_support"})
        self.assertIs(registry.get(1).lifecycle_status, ExpertLifecycleStatus.REJECTED)
        # terminal states cannot move.
        with self.assertRaisesRegex(ValueError, "illegal lifecycle transition"):
            registry.mark_provisional(1, {})
        # formal -> archived is legal.
        registry2 = ExpertRegistry()
        registry2.register(_candidate(2, checkpoint_path="/tmp/e2",
                                     checkpoint_sha256="c" * 64))
        registry2.mark_formal(2, {})
        registry2._set_lifecycle(2, ExpertLifecycleStatus.ARCHIVED, {"reason": "archived"})
        self.assertIs(registry2.get(2).lifecycle_status, ExpertLifecycleStatus.ARCHIVED)

    def test_round_trip_preserves_lifecycle_and_pool_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            registry = ExpertRegistry()
            registry.register(_candidate(1, checkpoint_path="/tmp/e1",
                                         checkpoint_sha256="d" * 64))
            registry.mark_formal(1, {"reason": "ok"})
            registry.increment_pool_version()
            registry.save_json(str(path))

            loaded = ExpertRegistry.load_json(str(path))
            self.assertEqual(loaded.pool_version, POOL_VERSION_INITIAL + 1)
            self.assertIs(
                loaded.get(1).lifecycle_status, ExpertLifecycleStatus.FORMAL
            )
            self.assertEqual(
                loaded.get(1).extra["lifecycle_conditions"]["formal"]["reason"],
                "ok",
            )

    def test_load_does_not_bump_pool_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            registry = ExpertRegistry()
            registry.register(_candidate(1, checkpoint_path="/tmp/e1",
                                         checkpoint_sha256="e" * 64))
            registry.increment_pool_version()
            registry.save_json(str(path))
            for _ in range(3):
                loaded = ExpertRegistry.load_json(str(path))
                self.assertEqual(loaded.pool_version, POOL_VERSION_INITIAL + 1)

    def test_canonical_alias_fields(self):
        registry = ExpertRegistry()
        metadata = registry.register(_candidate(1))
        state = registry.state_dict()["experts"][0]
        self.assertEqual(state["created_task_id"], 0)
        self.assertEqual(state["creation_task_name"], "ImageNet-R")
        self.assertEqual(state["lora_alpha"], 16.0)
        self.assertEqual(state["created_seed"], 42)
        self.assertEqual(state["rank"], 8)


class CommitTransactionV6Test(unittest.TestCase):
    def _setup(self, directory):
        registry_dir = Path(directory)
        registry = ExpertRegistry()
        transaction = CommitTransaction(str(registry_dir), registry)
        artifact = registry_dir / "lora.bin"
        artifact.write_bytes(b"weights")
        return registry_dir, registry, transaction, artifact

    def _complete(self, transaction, artifact):
        return transaction.complete(
            1,
            artifacts={str(artifact): file_sha256(str(artifact))},
            condition_record={"support_count": 10, "mean_conditional_gain": 0.04},
            metadata=_candidate(
                1,
                checkpoint_path=str(artifact),
                checkpoint_sha256=file_sha256(str(artifact)),
                lifecycle_status=ExpertLifecycleStatus.CANDIDATE,
            ),
        )

    def test_commit_lands_provisional_and_bumps_pool_version(self):
        with tempfile.TemporaryDirectory() as directory:
            registry_dir, registry, transaction, artifact = self._setup(directory)
            transaction.begin(1, {"task_id": 0})
            result = self._complete(transaction, artifact)
            self.assertEqual(result["status"], "committed")
            self.assertEqual(registry.pool_version, POOL_VERSION_INITIAL + 1)
            self.assertIs(
                registry.get(1).lifecycle_status, ExpertLifecycleStatus.PROVISIONAL
            )
            self.assertFalse(transaction.pending_path(1).exists())
            # Registry persisted atomically.
            loaded = ExpertRegistry.load_json(str(registry_dir / "expert_registry.json"))
            self.assertEqual(loaded.pool_version, POOL_VERSION_INITIAL + 1)

    def test_hash_mismatch_aborts_before_registry_update(self):
        with tempfile.TemporaryDirectory() as directory:
            registry_dir, registry, transaction, artifact = self._setup(directory)
            transaction.begin(1, {})
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                transaction.complete(
                    1,
                    artifacts={str(artifact): "0" * 64},
                    condition_record={},
                    metadata=_candidate(1),
                )
            self.assertEqual(registry.pool_version, POOL_VERSION_INITIAL)
            self.assertFalse(registry.contains(1))
            self.assertTrue(transaction.pending_path(1).exists())

    def test_recovery_incomplete_when_registry_missing_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            registry_dir, registry, transaction, artifact = self._setup(directory)
            transaction.begin(1, {"task_id": 0})
            # Crash before registry update.
            recovered, incomplete, completed = CommitTransaction.resume(
                str(registry_dir)
            )
            self.assertIn(1, incomplete)
            self.assertEqual(completed, {})
            self.assertFalse(recovered.contains(1))
            self.assertEqual(recovered.pool_version, POOL_VERSION_INITIAL)

    def test_recovery_completed_is_idempotent_without_version_bump(self):
        with tempfile.TemporaryDirectory() as directory:
            registry_dir, registry, transaction, artifact = self._setup(directory)
            transaction.begin(1, {"task_id": 0})
            self._complete(transaction, artifact)
            # Simulate crash after registry write: re-create the marker.
            transaction.pending_path(1).write_text(
                json.dumps({"transaction_version": 1, "expert_id": 1, "phase": "pending"}),
                encoding="utf-8",
            )
            recovered, incomplete, completed = CommitTransaction.resume(
                str(registry_dir)
            )
            self.assertEqual(incomplete, {})
            self.assertIn(1, completed)
            self.assertEqual(recovered.pool_version, POOL_VERSION_INITIAL + 1)
            self.assertFalse(transaction.pending_path(1).exists())

    def test_expert_id_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            registry_dir, registry, transaction, artifact = self._setup(directory)
            transaction.begin(1, {})
            self._complete(transaction, artifact)
            # A second transaction for the same id must refuse.
            with self.assertRaisesRegex(ValueError, "already exists"):
                transaction.begin(1, {})

    def test_abort_removes_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            registry_dir, registry, transaction, artifact = self._setup(directory)
            transaction.begin(1, {})
            transaction.abort(1)
            self.assertFalse(transaction.pending_path(1).exists())
            # The expert never entered the registry (commit never landed).
            self.assertFalse(registry.contains(1))


if __name__ == "__main__":
    unittest.main()
