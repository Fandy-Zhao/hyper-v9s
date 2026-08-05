"""V6 Stage E11: test split never enters training or calibration.

Centralized check that every V6 module that consumes labels or tunes
thresholds refuses "test" data: teacher cache provenance, RMS
calibration, router threshold tuning, router evaluation, anchors and
recall audits.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from compose.expansion.v6_residual import V6ResidualRecord, V6ResidualBuffer
from compose.lora.statistics import RMSStatistics
from compose.lora.v6_rms import V6RMSConfig
from compose.router.v6_calibrate import (
    evaluate_v6_router,
    tune_v6_thresholds,
)
from compose.router.v6_router import V6QueryEncoder, V6Router
from compose.teacher.cache import write_shard


def _provenance(split="test"):
    return {
        "sample_id": "s", "dataset_manifest_hash": "data", "split": split,
        "tokenizer_hash": "tok", "model_identifier": "llava",
        "base_checkpoint_hash": "base", "expert_registry_hash": "reg",
        "expert_checkpoint_hashes": {"0": "h"}, "composition_mode": "direct_sum",
        "rms_statistics_hash": "none", "oracle_config_hash": "cfg",
        "answer_mask_version": "v1", "answer_template_hash": "tpl",
        "target_averaging": "token_mean", "composer_version": "v6",
        "code_version": "head", "pool_version": 1, "router_version": "r1",
    }


def _router():
    router = V6Router(V6QueryEncoder(visual_dim=4, text_dim=5, query_dim=8))
    router.add_expert(0, creation_task=0, checkpoint_sha256="a" * 64)
    return router


class TestSplitLeakageV6Test(unittest.TestCase):
    def test_teacher_cache_rejects_test(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            write_shard(path, [{"sample_id": "a"}], _provenance(split="train"), 0)
            with self.assertRaises(ValueError):
                write_shard(path, [{"sample_id": "a"}], _provenance(split="test"), 0)

    def test_rms_provenance_rejects_test(self):
        with self.assertRaises(ValueError):
            RMSStatistics(_provenance(split="test"))
        with self.assertRaises(ValueError):
            V6RMSConfig(calibration_split="test")

    def test_router_evaluation_rejects_test(self):
        router = _router()
        with self.assertRaisesRegex(ValueError, "test"):
            evaluate_v6_router(
                router, torch.randn(1, 8), [{"teacher_set": (0,)}], split="test"
            )

    def test_router_threshold_tuning_rejects_test(self):
        router = _router()
        with self.assertRaisesRegex(ValueError, "test"):
            tune_v6_thresholds(
                router, torch.randn(1, 8), [{"teacher_set": (0,)}],
                [(0.5, 0.5)], split="test",
            )

    def test_residual_buffer_rejects_test(self):
        with self.assertRaises(ValueError):
            V6ResidualRecord(
                sample_id="s", task_id=0, old_teacher_set=(0,),
                empty_loss=1.0, old_teacher_loss=0.9, old_gain=0.1,
                residual_reason="teacher_gain_below_floor",
                query_feature_path="/f", split="test",
            )

    def test_teacher_cache_provenance_rejects_test(self):
        from compose.teacher.validity import validate_cache_provenance

        with self.assertRaisesRegex(ValueError, "test"):
            validate_cache_provenance(_provenance(split="test"))


if __name__ == "__main__":
    unittest.main()
