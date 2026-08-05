"""V6 Stage E7: candidate validation, commit decisions and transactions."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from compose.experts.registry import POOL_VERSION_INITIAL, ExpertRegistry
from compose.experts.transaction import CommitTransaction, file_sha256
from compose.expansion.candidate_pool import CandidateExpertPool, CandidatePoolConfig
from compose.expansion.v6_candidate import V6CandidateConfig
from compose.expansion.v6_commit import (
    CandidateValidationStats,
    commit_candidates,
    decide_commits,
    evaluate_candidate,
    joint_effect,
)


def _pool(two_slots=True):
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    if not two_slots:
        keys = keys[:1]
    count = 2 if two_slots else 1
    return CandidateExpertPool(
        [nn.Linear(2, 2, bias=False) for _ in range(count)],
        keys,
        CandidatePoolConfig(query_dim=2, slot_count=count, min_support=2),
    )


def _records(count=4):
    return [
        {"sample_id": "s{}".format(index), "old_teacher_set": (99,)}
        for index in range(count)
    ]


def _queries(count=4):
    return torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]][:count],
        dtype=torch.float32,
    )


def _loss_fn(records):
    """loss(record, expert_set): adding candidate 1 helps samples 0-1,
    adding candidate 2 helps samples 2-3; old set alone is neutral."""
    def loss(record, expert_set):
        index = int(record["sample_id"][1:])
        base = 1.0
        if 0 in expert_set and index < 2:
            base -= 0.5  # slot 0 helps first half
        if 1 in expert_set and index >= 2:
            base -= 0.4  # slot 1 helps second half
        return base

    return loss


class EvaluateCandidateTest(unittest.TestCase):
    def test_statistics_computed_from_answer_teacher(self):
        pool = _pool()
        records = _records(4)
        queries = _queries(4)
        stats1 = evaluate_candidate(pool, 0, records, [99], _loss_fn(records), queries)
        stats2 = evaluate_candidate(pool, 1, records, [99], _loss_fn(records), queries)
        # slot 0 (key [1,0]) helps samples 0-1; slot 1 (key [0,1]) helps 2-3
        self.assertEqual(stats1.support_count, 2)
        self.assertEqual(stats2.support_count, 2)
        self.assertAlmostEqual(stats1.mean_gain, 0.25)  # (0.5+0.5+0+0)/4
        self.assertAlmostEqual(stats2.mean_gain, 0.2)
        self.assertEqual(stats1.positive_gain_rate, 0.5)
        self.assertAlmostEqual(stats1.median_gain, 0.25)
        self.assertIsNotNone(stats1.param_cosine)
        self.assertIsNotNone(stats1.key_cosine)
        # All queries are [1,0] -> assigned to slot 0; positives are 0-1.
        self.assertAlmostEqual(stats1.key_accuracy, 1.0)

    def test_single_slot_statistics(self):
        pool = _pool(two_slots=False)
        records = _records(4)
        queries = _queries(4)
        stats = evaluate_candidate(pool, 0, records, [99], _loss_fn(records), queries)
        self.assertIsNone(stats.param_cosine)
        self.assertIsNone(stats.key_cosine)


class DecideCommitsTest(unittest.TestCase):
    def _stats(self, slot_id, support, gain, key_acc, positives=("a", "b"),
               param_cosine=None):
        return CandidateValidationStats(
            slot_id=slot_id,
            support_count=support,
            mean_gain=gain,
            median_gain=gain,
            positive_gain_rate=1.0,
            key_accuracy=key_acc,
            false_activation_rate=0.0,
            param_cosine=param_cosine,
            key_cosine=None,
            positive_sample_ids=positives,
        )

    def test_zero_commits_when_conditions_fail(self):
        config = V6CandidateConfig(slot_count=2)
        decisions = decide_commits(
            [self._stats(0, 1, 0.01, 0.3), self._stats(1, 0, -0.1, 0.1)],
            config, tau_support=8, tau_gain=0.0, tau_key=0.5,
        )
        self.assertEqual([decision.status for decision in decisions],
                         ["rejected", "rejected"])

    def test_both_commit_when_conditions_pass(self):
        config = V6CandidateConfig(slot_count=2)
        decisions = decide_commits(
            [self._stats(0, 10, 0.2, 0.9), self._stats(1, 9, 0.1, 0.8)],
            config, tau_support=8, tau_gain=0.0, tau_key=0.5,
        )
        self.assertEqual([decision.status for decision in decisions],
                         ["provisional", "provisional"])

    def test_redundant_pair_keeps_higher_gain(self):
        config = V6CandidateConfig(slot_count=2)
        decisions = decide_commits(
            [
                self._stats(0, 10, 0.1, 0.9, positives=("a", "b", "c"), param_cosine=0.95),
                self._stats(1, 10, 0.2, 0.9, positives=("a", "b", "c"), param_cosine=0.95),
            ],
            config, tau_support=8, tau_gain=0.0, tau_key=0.5,
            tau_param_cosine=0.0, tau_overlap=0.5,  # always redundant
        )
        self.assertEqual(decisions[0].status, "redundant")
        self.assertEqual(decisions[1].status, "provisional")
        self.assertEqual(decisions[1].reason, "kept_over_redundant_pair")

    def test_single_slot_pool_commit(self):
        config = V6CandidateConfig(slot_count=1)
        decisions = decide_commits(
            [self._stats(0, 10, 0.2, 0.9)],
            config, tau_support=8, tau_gain=0.0, tau_key=0.5,
        )
        self.assertEqual(decisions[0].status, "provisional")


class JointEffectTest(unittest.TestCase):
    def test_joint_gain_of_both_candidates(self):
        pool = _pool()
        records = _records(4)
        effect = joint_effect(pool, records, _loss_fn(records))
        # joint set helps all 4 samples: gains [0.5, 0.5, 0.4, 0.4]
        self.assertAlmostEqual(effect["joint_gain"], 0.45)
        self.assertEqual(effect["joint_support"], 4)


class CommitCandidatesTest(unittest.TestCase):
    def test_transactional_commit_of_one_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            registry_dir = Path(directory)
            registry = ExpertRegistry()
            transaction = CommitTransaction(str(registry_dir), registry)
            pool = _pool()
            decisions = [
                CandidateValidationStats(0, 10, 0.2, 0.2, 1.0, 0.9, 0.0, None, None, ("a",)),
                CandidateValidationStats(1, 0, -0.1, -0.1, 0.0, 0.1, 0.0, None, None, ()),
            ]
            commit_decisions = decide_commits(
                decisions, V6CandidateConfig(slot_count=2),
                tau_support=8, tau_gain=0.0, tau_key=0.5,
            )

            def writer(expert_id, staging_dir):
                checkpoint = Path(staging_dir) / "compose_experts.bin"
                checkpoint.write_bytes("lora-weights-{}".format(expert_id).encode())
                key_file = Path(staging_dir) / "key.json"
                key_file.write_text('{"key": [1.0, 0.0]}', encoding="utf-8")
                return {
                    str(checkpoint): file_sha256(str(checkpoint)),
                    str(key_file): file_sha256(str(key_file)),
                }

            report_dir = Path(directory)
            reports = {}
            for decision in commit_decisions:
                if decision.status == "provisional":
                    report = report_dir / "report_{}.json".format(decision.slot_id)
                    report.write_text('{"ok": true}', encoding="utf-8")
                    reports[decision.slot_id] = str(report)

            committed = commit_candidates(
                transaction, registry, commit_decisions,
                first_expert_id=10, creation_task=1, creation_task_name="ArxivQA",
                created_seed=42, pool_version=registry.pool_version,
                config_hash="cfg", artifact_writers={0: writer, 1: writer},
                validation_reports=reports,
            )
            self.assertEqual(len(committed), 1)
            self.assertEqual(committed[0].committed_expert_id, 10)
            self.assertTrue(registry.contains(10))
            self.assertEqual(registry.pool_version, POOL_VERSION_INITIAL + 1)
            metadata = registry.get(10)
            self.assertEqual(metadata.creation_task, 1)
            self.assertEqual(metadata.creation_task_name, "ArxivQA")
            self.assertEqual(metadata.created_seed, 42)
            self.assertEqual(metadata.support_count, 10)
            self.assertAlmostEqual(metadata.mean_conditional_gain, 0.2)
            self.assertAlmostEqual(metadata.key_accuracy, 0.9)
            self.assertEqual(metadata.config_hash, "cfg")
            self.assertEqual(metadata.pool_version_created, POOL_VERSION_INITIAL)
            # Registry persisted atomically; no pending markers left.
            self.assertFalse(transaction.pending_path(10).exists())
            loaded = ExpertRegistry.load_json(str(registry_dir / "expert_registry.json"))
            self.assertEqual(loaded.pool_version, POOL_VERSION_INITIAL + 1)

    def test_crash_recovery_does_not_double_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            registry_dir = Path(directory)
            registry = ExpertRegistry()
            transaction = CommitTransaction(str(registry_dir), registry)

            def writer(expert_id, staging_dir):
                checkpoint = Path(staging_dir) / "compose_experts.bin"
                checkpoint.write_bytes("w-{}".format(expert_id).encode())
                return {str(checkpoint): file_sha256(str(checkpoint))}

            decisions = [CandidateValidationStats(0, 10, 0.2, 0.2, 1.0, 0.9, 0.0, None, None, ("a",))]
            commit_decisions = decide_commits(
                decisions, V6CandidateConfig(slot_count=1),
                tau_support=8, tau_gain=0.0, tau_key=0.5,
            )
            committed = commit_candidates(
                transaction, registry, commit_decisions,
                first_expert_id=20, creation_task=2, creation_task_name="VizWiz",
                created_seed=42, pool_version=registry.pool_version,
                config_hash="cfg", artifact_writers={0: writer},
                validation_reports={},
            )
            self.assertEqual(len(committed), 1)
            version_after = registry.pool_version
            # Simulate crash after registry write: marker re-appears.
            transaction.pending_path(20).write_text(
                '{"transaction_version": 1, "expert_id": 20, "phase": "pending"}',
                encoding="utf-8",
            )
            recovered, incomplete, completed = CommitTransaction.resume(str(registry_dir))
            self.assertEqual(incomplete, {})
            self.assertIn(20, completed)
            self.assertEqual(recovered.pool_version, version_after)


if __name__ == "__main__":
    unittest.main()
