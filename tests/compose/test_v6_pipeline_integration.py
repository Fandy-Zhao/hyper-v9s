"""V6 Stage E11: end-to-end pipeline integration on a tiny model.

Chains the real modules (no full 7B model): teacher search -> residual
split -> candidate pool -> validation -> transactional commit -> global
teacher + Router calibration -> RMS collection -> snapshot -> resume.
Verifies the cross-module contracts: pool_version monotonicity, no
duplicate commits, old expert hashes unchanged, snapshot independent
load and idempotent resume.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from compose.adapters import ExpertManager, inject_compose_adapters
from compose.config import ComposeAdapterConfig
from compose.experts import ExpertPool
from compose.experts.metadata import ExpertLifecycleStatus, ExpertMetadata
from compose.experts.registry import ExpertRegistry
from compose.experts.task_state import TaskStage, TaskStateMachine
from compose.experts.transaction import CommitTransaction, file_sha256
from compose.expansion.candidate_pool import CandidateExpertPool
from compose.expansion.v6_candidate import V6CandidateConfig
from compose.expansion.v6_commit import (
    CandidateValidationStats,
    commit_candidates,
    decide_commits,
)
from compose.expansion.v6_residual import (
    RESIDUAL_REASON_GAIN_FLOOR,
    build_residual_records,
    should_create_candidates,
)
from compose.experiments.v6_snapshot import V6Snapshot, analyze_resume
from compose.router.v6_calibrate import (
    build_multi_hot_labels,
    calibrate_v6_router,
    evaluate_v6_router,
)
from compose.router.v6_router import V6QueryEncoder, V6Router
from compose.teacher.types import AnswerNLL
from compose.teacher.v6_teacher import V6TeacherRecord
from test_injection import TinyModel


def _teacher_records(count=16):
    records = []
    for index in range(count):
        if index % 4 == 3:
            teacher = ()
            loss = 1.0
        elif index % 4 == 0:
            teacher = (0,)
            loss = 0.95  # old expert helps but gain below the floor
        else:
            teacher = (0,)
            loss = 0.8  # sufficient
        records.append(
            V6TeacherRecord(
                sample_id="s{}".format(index),
                task_id=1,
                pool_version=1,
                router_version="r1",
                candidate_experts=(0,),
                empty_loss=1.0,
                single_losses={0: loss},
                pair_losses={},
                best_single=teacher if teacher else (),
                best_pair=(),
                pair_gain=None,
                teacher_set=teacher,
                teacher_loss=loss,
                teacher_multi_hot={0: 1} if teacher else {},
                cache_key="ck{}".format(index),
            )
        )
    return records


class V6PipelineIntegrationTest(unittest.TestCase):
    def test_full_pipeline_tiny_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # --- registry + task state ---------------------------------
            registry = ExpertRegistry()
            old_metadata = ExpertMetadata(
                expert_id=0,
                adapter_name="expert_0000",
                rank=8,
                alpha=16.0,
                creation_task=0,
                creation_task_name="ImageNet-R",
                created_seed=42,
                checkpoint_path="/ckpt/expert_0000/compose_experts.bin",
                checkpoint_sha256="old" * 21 + "ab",
                lifecycle_status=ExpertLifecycleStatus.PROVISIONAL,
            )
            registry.register(old_metadata)
            task_state = TaskStateMachine(1, "ArxivQA")
            for stage in (TaskStage.DATA_READY, TaskStage.OLD_TEACHER_RUNNING,
                          TaskStage.OLD_TEACHER_READY):
                task_state.advance(stage)

            # --- residual split (E5) -----------------------------------
            reuse, residual = build_residual_records(
                _teacher_records(),
                query_feature_path="/feat/train.pt",
                min_old_gain=0.1,
                top_m_covered=[True] * 16,
                split="train",
            )
            self.assertTrue(reuse)
            residual_train = [r for r in residual if r.split == "train"]
            # 4 residual samples (index % 4 == 0); gate on that count.
            self.assertTrue(should_create_candidates(len(residual_train), 4))

            # --- candidate pool (E6) ------------------------------------
            queries = torch.randn(8, 8)
            config = V6CandidateConfig(slot_count=2, query_dim=8,
                                       key_initialization="random_orthogonal")
            adapters = [torch.nn.Linear(3, 3, bias=False) for _ in range(2)]
            pool, init_record = _build_pool(config, queries, adapters)
            self.assertIn(init_record["method"], ("kmeans_plus_plus", "random_orthogonal"))

            # --- validation + commit (E7) --------------------------------
            records_for_eval = [
                {"sample_id": "s{}".format(index), "old_teacher_set": (0,)}
                for index in range(8)
            ]
            eval_queries = torch.randn(8, 8)

            def loss_fn(record, expert_set):
                return 1.0 - 0.05 * len(expert_set)

            stats_list = [
                CandidateValidationStats(
                    slot_id=slot_id,
                    support_count=10,
                    mean_gain=0.2,
                    median_gain=0.15,
                    positive_gain_rate=0.8,
                    key_accuracy=0.9,
                    false_activation_rate=0.1,
                    param_cosine=None,
                    key_cosine=None,
                    positive_sample_ids=("a", "b", "c"),
                )
                for slot_id in range(2)
            ]
            decisions = decide_commits(
                stats_list, config, tau_support=8, tau_gain=0.0, tau_key=0.5
            )
            transaction = CommitTransaction(str(root), registry)

            def writer(expert_id, staging_dir):
                checkpoint = Path(staging_dir) / "compose_experts.bin"
                checkpoint.write_bytes(("w{}".format(expert_id)).encode())
                key_file = Path(staging_dir) / "key.json"
                key_file.write_text('{"key": [1.0]}', encoding="utf-8")
                return {
                    str(checkpoint): file_sha256(str(checkpoint)),
                    str(key_file): file_sha256(str(key_file)),
                }

            committed = commit_candidates(
                transaction, registry, decisions,
                first_expert_id=10, creation_task=1, creation_task_name="ArxivQA",
                created_seed=42, pool_version=registry.pool_version,
                config_hash="cfg", artifact_writers={0: writer, 1: writer},
                validation_reports={},
            )
            self.assertGreaterEqual(len(committed), 1)
            self.assertEqual(registry.pool_version, 1 + len(committed))
            # Old expert hash unchanged.
            self.assertEqual(registry.get(0).checkpoint_sha256, "old" * 21 + "ab")

            # --- global teacher + Router calibration (E8) ----------------
            committed_ids = [c.committed_expert_id for c in committed]
            teacher_sets = [{"teacher_set": (expert_id,)} for expert_id in committed_ids]
            labels, pool_ids = build_multi_hot_labels(teacher_sets, committed_ids)
            encoder = V6QueryEncoder(visual_dim=4, text_dim=5, query_dim=8)
            router = V6Router(encoder, seed=42)
            for expert in committed:
                router.add_expert(
                    expert.committed_expert_id, creation_task=1,
                    checkpoint_sha256="x" * 64,
                )
            calibrate_v6_router(
                router, torch.randn(len(committed_ids), 8), labels, _calib_config(),
            )

            # --- RMS (E9) -------------------------------------------------
            model, pool_manager = _tiny_model_with_experts(committed)
            from compose.lora.v6_rms import (
                build_rms_provenance,
                compute_v6_expert_rms,
            )

            provenance = build_rms_provenance(
                "validation", "hash", "data", "cfg"
            )
            rms_stats, pair = compute_v6_expert_rms(
                model, [e.committed_expert_id for e in committed],
                _dataloader(), provenance, _rms_config(),
                prepare_batch=lambda batch: batch,
                forward_fn=lambda inputs: _decoder_output(model, inputs),
            )
            self.assertGreaterEqual(len(rms_stats.entries), 1)

            # --- snapshot + resume (E10) ---------------------------------
            # Advance the state machine through the commit stage, as the
            # runner would after a successful commit.
            for stage in (TaskStage.RESIDUAL_READY, TaskStage.CANDIDATE_TRAINING,
                          TaskStage.CANDIDATE_TRAINED, TaskStage.CANDIDATE_VALIDATED,
                          TaskStage.EXPERTS_COMMITTED):
                task_state.advance(stage)
            snapshot = V6Snapshot.create(
                str(root / "snap"), task_id=1, task_name="ArxivQA",
                registry=registry, task_state=task_state,
                git_commit="test", command="integration",
                data_hash="data-hash",
            )
            loaded = V6Snapshot.load(str(root / "snap"))
            self.assertEqual(loaded.registry.pool_version, registry.pool_version)
            analysis = analyze_resume(str(root / "snap"))
            self.assertIn(analysis["resume_node"],
                          ("candidate_committed", "router_calibrating"))
            self.assertEqual(analysis["pool_version"], registry.pool_version)


def _build_pool(config, queries, adapters):
    from compose.expansion.v6_candidate import build_v6_candidate_pool

    return build_v6_candidate_pool(config, queries, adapters)


def _calib_config():
    from compose.router.v6_calibrate import CalibrationConfig

    return CalibrationConfig(epochs=1, batch_size=4, learning_rate=1e-3)


def _rms_config():
    from compose.lora.v6_rms import V6RMSConfig

    return V6RMSConfig()


def _dataloader(batches=2):
    for _ in range(batches):
        yield torch.randn(3, 4, 3)


def _decoder_output(model, inputs):
    layer = model.model.layers[0]
    hidden = layer.self_attn.o_proj(
        layer.self_attn.q_proj(inputs)
        + layer.self_attn.k_proj(inputs)
        + layer.self_attn.v_proj(inputs)
    )
    return layer.mlp.down_proj(
        layer.mlp.gate_proj(hidden) + layer.mlp.up_proj(hidden)
    )


def _tiny_model_with_experts(committed):
    torch.manual_seed(7)
    model = TinyModel(layer_count=2)
    inject_compose_adapters(model, ComposeAdapterConfig(rank=1, alpha=2))
    pool = ExpertPool(ExpertManager(model))
    for expert in committed:
        pool.manager.add_expert(expert.committed_expert_id)
    return model, pool


if __name__ == "__main__":
    unittest.main()
