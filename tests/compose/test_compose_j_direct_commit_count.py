"""Test J (spec §27): direct commit -- each committed expert bumps
``pool_version`` exactly once; reruns are idempotent; artifact hashes are
verified; rejected candidates never enter the active pool."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from compose.experts import ExpertLifecycleStatus, ExpertMetadata, ExpertRegistry
from compose.experts.transaction import CommitTransaction


def _write(path: Path, content: bytes) -> str:
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _metadata(expert_id, checkpoint_path, checkpoint_sha256):
    return ExpertMetadata(
        expert_id=expert_id,
        adapter_name="expert_{}".format(expert_id),
        rank=8,
        alpha=16.0,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        active=True,
        trainable=False,
        lifecycle_status=ExpertLifecycleStatus.FORMAL,
        extra={"commit_rule": "direct_cluster_commit", "key_mode": "learnable"},
    )


class DirectCommitCountTest(unittest.TestCase):
    def test_each_commit_bumps_pool_version_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ExpertRegistry()
            transaction = CommitTransaction(root, registry)
            bin_path = root / "bin" / "pool_0000.bin"
            bin_path.parent.mkdir(parents=True)
            content_hash = _write(bin_path, b"expert-0-weights")
            transaction.begin(0, {"intent": "cluster_expert"})
            outcome = transaction.complete(
                0,
                artifacts={str(bin_path): content_hash},
                condition_record={"rule": "direct_cluster_commit", "cluster_id": 0, "cluster_size": 16},
                metadata=_metadata(0, str(bin_path), content_hash),
            )
            self.assertEqual(outcome["status"], "committed")
            self.assertEqual(registry.pool_version, 2)  # 1 -> 2, one bump
            # Idempotent rerun: no additional bump.
            rerun = transaction.complete(
                0,
                artifacts={str(bin_path): content_hash},
                condition_record={},
                metadata=_metadata(0, str(bin_path), content_hash),
            )
            self.assertEqual(rerun["status"], "already_committed")
            self.assertEqual(registry.pool_version, 2)

    def test_two_commits_two_bumps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ExpertRegistry()
            transaction = CommitTransaction(root, registry)
            versions = []
            for expert_id in (0, 1):
                bin_path = root / "bin" / "pool_{:04d}.bin".format(expert_id)
                bin_path.parent.mkdir(parents=True, exist_ok=True)
                content_hash = _write(bin_path, ("weights-{}".format(expert_id)).encode())
                transaction.begin(expert_id, {"intent": "cluster_expert"})
                transaction.complete(
                    expert_id,
                    artifacts={str(bin_path): content_hash},
                    condition_record={"rule": "direct_cluster_commit"},
                    metadata=_metadata(expert_id, str(bin_path), content_hash),
                )
                versions.append(registry.pool_version)
            self.assertEqual(versions, [2, 3])
            self.assertEqual(
                [expert.expert_id for expert in registry.get_active_experts()],
                [0, 1],
            )

    def test_hash_mismatch_rejected_without_bump(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ExpertRegistry()
            transaction = CommitTransaction(root, registry)
            bin_path = root / "bin" / "pool_0000.bin"
            bin_path.parent.mkdir(parents=True)
            _write(bin_path, b"real-weights")
            transaction.begin(0, {"intent": "cluster_expert"})
            with self.assertRaises(ValueError):
                transaction.complete(
                    0,
                    artifacts={str(bin_path): "deadbeef" * 16},
                    condition_record={},
                    metadata=_metadata(0, str(bin_path), "deadbeef" * 16),
                )
            self.assertEqual(registry.pool_version, 1)
            self.assertFalse(registry.contains(0))

    def test_rejected_candidates_never_active(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ExpertRegistry()
            transaction = CommitTransaction(root, registry)
            bin_path = root / "bin" / "pool_0000.bin"
            bin_path.parent.mkdir(parents=True)
            content_hash = _write(bin_path, b"weights")
            metadata = _metadata(0, str(bin_path), content_hash)
            metadata.active = False
            metadata.lifecycle_status = ExpertLifecycleStatus.REJECTED
            transaction.begin(0, {"intent": "candidate"})
            transaction.complete(
                0,
                artifacts={str(bin_path): content_hash},
                condition_record={"rule": "rejected"},
                metadata=metadata,
            )
            # A rejected candidate is registered but never part of the
            # active/formal pool.
            self.assertEqual(registry.get_active_experts(), [])
            self.assertEqual(
                [expert.expert_id for expert in registry.get_rejected_candidates()],
                [0],
            )

    def test_snapshot_restore_preserves_pool_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ExpertRegistry()
            transaction = CommitTransaction(root, registry)
            for expert_id in (0, 1):
                bin_path = root / "bin" / "pool_{:04d}.bin".format(expert_id)
                bin_path.parent.mkdir(parents=True, exist_ok=True)
                content_hash = _write(bin_path, ("weights-{}".format(expert_id)).encode())
                transaction.begin(expert_id, {"intent": "cluster_expert"})
                transaction.complete(
                    expert_id,
                    artifacts={str(bin_path): content_hash},
                    condition_record={"rule": "direct_cluster_commit"},
                    metadata=_metadata(expert_id, str(bin_path), content_hash),
                )
            registry_path = root / "expert_registry.json"
            registry.save_atomic(registry_path, allow_overwrite=True)
            restored = ExpertRegistry.load_json(registry_path)
            self.assertEqual(restored.pool_version, 3)  # restored as-is
            # A further commit bumps exactly once from the restored value.
            bin_path = root / "bin" / "pool_0002.bin"
            content_hash = _write(bin_path, b"weights-2")
            transaction = CommitTransaction(root, restored)
            transaction.begin(2, {"intent": "cluster_expert"})
            transaction.complete(
                2,
                artifacts={str(bin_path): content_hash},
                condition_record={"rule": "direct_cluster_commit"},
                metadata=_metadata(2, str(bin_path), content_hash),
            )
            self.assertEqual(restored.pool_version, 4)

    def test_resume_recovery_never_bumps_pool_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ExpertRegistry()
            transaction = CommitTransaction(root, registry)
            bin_path = root / "bin" / "pool_0000.bin"
            bin_path.parent.mkdir(parents=True)
            content_hash = _write(bin_path, b"weights-0")
            transaction.begin(0, {"intent": "cluster_expert"})
            transaction.complete(
                0,
                artifacts={str(bin_path): content_hash},
                condition_record={"rule": "direct_cluster_commit"},
                metadata=_metadata(0, str(bin_path), content_hash),
            )
            self.assertEqual(registry.pool_version, 2)
            # Simulate a crash window: the pending marker still exists while
            # the registry entry already landed. Recovery must clean the
            # marker and never bump pool_version again.
            transaction.pending_path(0).write_text(
                '{"transaction_version": 1, "expert_id": 0, "phase": "pending"}'
            )
            resumed, incomplete, completed = CommitTransaction.resume(root, registry)
            self.assertEqual(
                completed,
                {0: {"transaction_version": 1, "expert_id": 0, "phase": "pending"}},
            )
            self.assertEqual(incomplete, {})
            self.assertFalse(transaction.pending_path(0).exists())
            self.assertEqual(registry.pool_version, 2)
            # The next real commit bumps exactly once.
            bin1 = root / "bin" / "pool_0001.bin"
            content_hash1 = _write(bin1, b"weights-1")
            transaction.begin(1, {"intent": "cluster_expert"})
            transaction.complete(
                1,
                artifacts={str(bin1): content_hash1},
                condition_record={"rule": "direct_cluster_commit"},
                metadata=_metadata(1, str(bin1), content_hash1),
            )
            self.assertEqual(registry.pool_version, 3)

    def test_pending_cleanup_on_abort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ExpertRegistry()
            transaction = CommitTransaction(root, registry)
            transaction.begin(7, {"intent": "cluster_expert"})
            self.assertTrue(transaction.pending_path(7).is_file())
            transaction.abort(7)
            self.assertFalse(transaction.pending_path(7).exists())
            self.assertFalse(registry.contains(7))


if __name__ == "__main__":
    unittest.main()
