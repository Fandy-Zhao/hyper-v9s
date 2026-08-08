import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from llava.constants import IGNORE_INDEX
from compose.experts import ExpertMetadata, ExpertRegistry, ExpertStatus
from compose.oracle.losses import compute_per_sample_nll
from compose.teacher import (
    AnswerNLL, OracleConfig, answer_token_nll, assert_temporal_boundary,
    cache_key, canonical_expert_set, empty_and_singles, load_shard, merge_shards,
    pair_candidates, resume_sample_ids, select_oracle_set, summarize_oracles,
    temporal_expert_ids, validate_oracle_split, write_shard,
)


def config(mode="direct_sum", penalty=.01, threshold=.02):
    return OracleConfig("oracle_" + mode, mode, lambda_expert=penalty, delta_pair_raw=threshold)


def nll(value, tokens=2, correct=False):
    return AnswerNLL(value * tokens, value, tokens, correct)


def provenance(mode="direct_sum", **changes):
    value = {
        "sample_id": "shard", "dataset_manifest_hash": "data", "split": "train",
        "tokenizer_hash": "tok", "model_identifier": "llava", "base_checkpoint_hash": "base",
        "expert_registry_hash": "registry", "expert_checkpoint_hashes": {"0": "expert0"},
        "composition_mode": mode, "rms_statistics_hash": "none" if mode == "direct_sum" else "rms",
        "oracle_config_hash": "direct" if mode == "direct_sum" else "rms-config",
        "answer_mask_version": "v1", "answer_template_hash": "template",
        "target_averaging": "token_mean", "composer_version": "stage03", "code_version": "head",
        "pool_version": 1, "router_version": "compose_router_v1",
    }
    value.update(changes)
    return value


def selected(scores, cfg=None):
    cfg = cfg or config()
    return select_oracle_set(
        sample_id="s", task_id=1, task_name="task", split="train",
        candidate_expert_ids=[0, 1], nll_by_set=scores, config=cfg,
        config_hash="cfg", expert_pool_hash="pool", dataset_manifest_hash="data",
        temporal_scope="post_task_diagnostic",
    )


class AnswerScorerTest(unittest.TestCase):
    def test_shift_uses_logits_before_answer_position(self):
        logits = torch.zeros(1, 4, 3)
        labels = torch.tensor([[IGNORE_INDEX, 2, 1, IGNORE_INDEX]])
        logits[0, 0, 2] = 8.0
        logits[0, 1, 1] = 8.0
        logits[0, 2, 0] = 8.0  # Would be incorrectly used by an unshifted scorer.
        result = answer_token_nll(logits, labels, eos_token_id=0)
        self.assertLess(float(result["mean_nll"][0]), .01)
        self.assertEqual(int(result["token_count"][0]), 2)

    def test_prompt_mask_multitoken_mean_and_details(self):
        logits = torch.tensor([[[3., 0.], [0., 3.], [3., 0.], [0., 3.]]])
        labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 0, 1]])
        result = answer_token_nll(logits, labels, eos_token_id=9, return_per_token=True)
        expected = torch.stack((F.cross_entropy(logits[:, 1], torch.tensor([0])),
                                F.cross_entropy(logits[:, 2], torch.tensor([1])))).mean()
        torch.testing.assert_close(result["mean_nll"], expected.reshape(1))
        self.assertEqual(len(result["per_token_nll"][0]), 2)

    def test_eos_default_excluded_and_configurable(self):
        logits = torch.zeros(1, 3, 4)
        labels = torch.tensor([[IGNORE_INDEX, 1, 2]])
        excluded = answer_token_nll(logits, labels, eos_token_id=2)
        included = answer_token_nll(logits, labels, eos_token_id=2, include_eos=True)
        self.assertEqual(excluded["token_count"].tolist(), [1])
        self.assertEqual(included["token_count"].tolist(), [2])

    def test_empty_answer_raises(self):
        with self.assertRaisesRegex(ValueError, "zero answer tokens"):
            answer_token_nll(torch.zeros(1, 2, 3), torch.full((1, 2), IGNORE_INDEX), eos_token_id=2)

    def test_bf16_finite_and_stage03_regression(self):
        torch.manual_seed(4)
        logits = torch.randn(2, 5, 7, dtype=torch.bfloat16)
        labels = torch.tensor([[IGNORE_INDEX, 1, 2, 3, 4], [IGNORE_INDEX, 2, 1, IGNORE_INDEX, 3]])
        new = answer_token_nll(logits, labels, include_eos=True)
        old = compute_per_sample_nll(logits, labels, return_details=True)
        torch.testing.assert_close(new["mean_nll"], old["mean_nll"])
        torch.testing.assert_close(new["sum_nll"], old["loss_sum"])
        self.assertTrue(torch.isfinite(new["mean_nll"]).all())


class CandidateAndTemporalTest(unittest.TestCase):
    def test_empty_single_pair_generation_and_deduplication(self):
        self.assertEqual(empty_and_singles([2, 0, 2, 1]), ((), (0,), (1,), (2,)))
        pairs = pair_candidates([3, 2, 1, 0], {0: .4, 1: .3, 2: .2, 3: .1})
        self.assertEqual(len(pairs), 6)
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(canonical_expert_set([1, 0]), (0, 1))
        with self.assertRaises(ValueError):
            canonical_expert_set([0, 0])
        with self.assertRaises(ValueError):
            canonical_expert_set([0, 1, 2])

    def test_large_pool_uses_best_single_top4(self):
        pairs = pair_candidates(range(6), {value: float(value) for value in range(6)})
        self.assertEqual(pairs, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))

    def test_archived_unavailable_and_future_experts_excluded(self):
        registry = ExpertRegistry()
        for expert_id in range(4):
            registry.register(ExpertMetadata(
                expert_id, adapter_name=str(expert_id), creation_task=expert_id,
                checkpoint_path="ckpt", checkpoint_sha256="hash",
            ))
        registry.archive(1)
        self.assertEqual(temporal_expert_ids(registry, 2, "historical_only", range(4)), (0,))
        self.assertEqual(temporal_expert_ids(registry, 2, "post_task_diagnostic", [0, 1, 2]), (0, 2))
        self.assertEqual(temporal_expert_ids(registry, 3, "post_task_diagnostic", [0, 2]), (0, 2))
        with self.assertRaisesRegex(ValueError, "future expert"):
            assert_temporal_boundary([0, 2], 1, "post_task_diagnostic", {0: 0, 2: 2})

    def test_missing_checkpoint_is_not_silently_available(self):
        registry = ExpertRegistry()
        registry.register(ExpertMetadata(0, adapter_name="0", creation_task=0))
        with self.assertRaisesRegex(ValueError, "verified checkpoint"):
            temporal_expert_ids(registry, 0, "post_task_diagnostic", [0])

    def test_test_split_rejected(self):
        for value in ("test", "testdev", "full_test_predictions"):
            with self.assertRaisesRegex(ValueError, "test answers"):
                validate_oracle_split(value)


class SelectionTest(unittest.TestCase):
    def test_empty_is_allowed(self):
        record = selected({(): nll(.05), (0,): nll(.2), (1,): nll(.3), (0, 1): nll(.04)})
        self.assertEqual(record.selected.expert_ids, ())

    def test_single_is_best(self):
        record = selected({(): nll(1.), (0,): nll(.3), (1,): nll(.5), (0, 1): nll(.31)})
        self.assertEqual(record.selected.expert_ids, (0,))

    def test_pair_raw_gain_rejected(self):
        record = selected({(): nll(2.), (0,): nll(.5), (1,): nll(.7), (0, 1): nll(.485)})
        self.assertFalse(record.best_pair.valid_pair)
        self.assertEqual(record.selected.expert_ids, (0,))

    def test_pair_passes_both_thresholds(self):
        record = selected({(): nll(2.), (0,): nll(.5), (1,): nll(.7), (0, 1): nll(.4)})
        self.assertTrue(record.best_pair.valid_pair)
        self.assertEqual(record.selected.expert_ids, (0, 1))
        self.assertAlmostEqual(record.best_pair.raw_gain_over_best_single, .1)
        self.assertAlmostEqual(record.best_pair.penalized_gain_over_best_single, .09)

    def test_complexity_penalty_can_reject_pair(self):
        cfg = config(penalty=.1, threshold=.02)
        record = selected({(): nll(2.), (0,): nll(.5), (1,): nll(.7), (0, 1): nll(.45)}, cfg)
        self.assertFalse(record.best_pair.valid_pair)
        self.assertEqual(record.selected.expert_ids, (0,))

    def test_tie_break_score_cardinality_then_lexicographic(self):
        record = selected({(): nll(2.), (0,): nll(.5), (1,): nll(.5)}, config(penalty=0))
        self.assertEqual(record.selected.expert_ids, (0,))
        record = selected({(): nll(.5), (0,): nll(.5), (1,): nll(.5)}, config(penalty=0))
        self.assertEqual(record.selected.expert_ids, ())

    def test_token_count_must_not_change(self):
        with self.assertRaisesRegex(ValueError, "token count"):
            selected({(): nll(1., 1), (0,): nll(.5, 2)})


class CacheTest(unittest.TestCase):
    def test_atomic_roundtrip_resume_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank0.json"
            rows = [{"sample_id": "a", "selected_expert_ids": [0]}]
            write_shard(path, rows, provenance(), 0)
            self.assertEqual(load_shard(path, provenance())["records"], rows)
            self.assertEqual(resume_sample_ids(path, provenance()), ("a",))
            self.assertFalse(list(path.parent.glob("*.tmp")))
            data = json.loads(path.read_text())
            data["records"][0]["selected_expert_ids"] = [1]
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_shard(path, provenance())

    def test_every_provenance_change_invalidates(self):
        fields = ["base_checkpoint_hash", "dataset_manifest_hash", "tokenizer_hash", "rms_statistics_hash",
                  "answer_template_hash", "answer_mask_version", "composition_mode", "oracle_config_hash",
                  "target_averaging", "composer_version", "expert_checkpoint_hashes",
                  "pool_version", "router_version"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            write_shard(path, [{"sample_id": "a"}], provenance(), 0)
            for field in fields:
                changed = provenance(**{field: {"0": "changed"} if field == "expert_checkpoint_hashes" else "changed"})
                with self.assertRaises(ValueError, msg=field):
                    load_shard(path, changed)

    def test_direct_and_rms_are_isolated(self):
        self.assertNotEqual(cache_key(provenance("direct_sum")), cache_key(provenance("rms_calibrated")))

    def test_merge_ranks_detects_duplicate_and_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left, right, output = root / "rank0.json", root / "rank1.json", root / "merged.json"
            write_shard(left, [{"sample_id": "a"}], provenance(), 0)
            write_shard(right, [{"sample_id": "b"}], provenance(), 1)
            merged = merge_shards([left, right], output, provenance(), ["a", "b"])
            self.assertEqual(merged["sample_count"], 2)
            with self.assertRaisesRegex(ValueError, "missing"):
                merge_shards([left], output, provenance(), ["a", "b"])
            write_shard(right, [{"sample_id": "a"}], provenance(), 1)
            with self.assertRaisesRegex(ValueError, "duplicate Oracle samples"):
                merge_shards([left, right], output, provenance(), ["a"])

    def test_test_cache_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "test answers"):
                write_shard(Path(directory) / "bad.json", [], provenance(split="test"), 0)


class MetricsTest(unittest.TestCase):
    def test_rates_synergy_harmful_transitions_and_collapses(self):
        first = selected({(): nll(1.), (0,): nll(.5, correct=False), (1,): nll(.7), (0, 1): nll(.4, correct=True)}).to_dict()
        second = selected({(): nll(.2), (0,): nll(.5), (1,): nll(.6), (0, 1): nll(.7)}).to_dict()
        summary = summarize_oracles([first, second])
        self.assertEqual(summary["PairEvaluatedRate"], 1.0)
        self.assertEqual(summary["PairSelectedRate"], .5)
        self.assertEqual(summary["EmptyOracleRate"], .5)
        self.assertGreater(summary["positive_pair_synergy_rate"], 0)
        self.assertGreater(summary["harmful_pair_rate"], 0)
        self.assertEqual(summary["accuracy_transitions"]["single_wrong_pair_correct"], 1)
        self.assertEqual(summary["selected_set_exact_accuracy"], .5)
        self.assertEqual(summary["best_single_exact_accuracy"], 0.0)
        self.assertEqual(summary["selected_vs_best_single_accuracy_delta"], .5)
        self.assertEqual(summary["pair_selected_accuracy_delta_vs_best_single"], 1.0)
        self.assertEqual(summary["single_expert_metrics"]["0"]["support"], 2)
        self.assertEqual(len(summary["worst_tail_sample_ids"]), 1)
        self.assertIsNone(summary["current_hyper_route_exact_accuracy"])
        self.assertFalse(summary["collapse_checks"]["duplicate_pair"])


class SearcherFloatNllRegression(unittest.TestCase):
    """Regression (smoke run 12, task 1 S2): search_from_nll feeds aggregate
    float NLLs into _score, which used to build AnswerNLL(mean_nll=...) with
    the required sum_nll/token_count missing."""

    def test_search_from_nll_accepts_float_aggregates(self):
        from compose.teacher.teacher import ComposeTeacherSearcher

        searcher = ComposeTeacherSearcher(
            config=OracleConfig(
                "compose_teacher", "rms_calibrated",
                lambda_expert=0.01, delta_pair_raw=0.02,
            ),
            router_version="compose_router_v1",
            pool_version=1,
            top_m=2,
        )
        teacher = searcher.search_from_nll(
            sample_id="s1",
            task_id=1,
            retrieved_top_m=[0, 1],
            nll_by_set={(): 0.4, (0,): 0.2, (1,): 0.3, (0, 1): 0.25},
            pool_size=2,
            provenance={},
        )
        record = teacher.to_dict()
        # The single expert 0 wins the regularized score; the aggregate NLL
        # round-trips through the degenerate single-token AnswerNLL.
        self.assertEqual(record["teacher_set"], [0])
        self.assertAlmostEqual(record["teacher_loss"], 0.2)

    def test_empty_set_scores_with_float_nll(self):
        from compose.teacher.teacher import ComposeTeacherSearcher

        searcher = ComposeTeacherSearcher(
            config=OracleConfig(
                "compose_teacher", "rms_calibrated",
                lambda_expert=0.5, delta_pair_raw=0.02,
            ),
            router_version="compose_router_v1",
            pool_version=1,
            top_m=2,
        )
        teacher = searcher.search_from_nll(
            sample_id="s1",
            task_id=1,
            retrieved_top_m=[0],
            nll_by_set={(): 0.1, (0,): 0.6},
            pool_size=1,
            provenance={},
        )
        self.assertEqual(teacher.to_dict()["teacher_set"], [])


if __name__ == "__main__":
    unittest.main()
