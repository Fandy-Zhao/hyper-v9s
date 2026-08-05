"""V6 Stage E4: answer-supervised old-expert teacher with Top-M retrieval."""

import tempfile
import unittest
from pathlib import Path

from compose.teacher.cache import cache_key, load_shard, write_shard
from compose.teacher.oracle_set import OracleConfig
from compose.teacher.types import AnswerNLL
from compose.teacher.v6_teacher import (
    V6TeacherRecord,
    V6TeacherSearcher,
    build_teacher_multi_hot,
    run_recall_audit,
)


def _config(**changes):
    fields = dict(
        oracle_name="v6",
        composition_mode="direct_sum",
        lambda_expert=0.01,
        delta_pair_raw=0.02,
        top_k_for_pair=4,
        max_pairs=6,
    )
    fields.update(changes)
    return OracleConfig(**fields)


def _provenance(**changes):
    value = {
        "sample_id": "s",
        "dataset_manifest_hash": "data",
        "split": "train",
        "tokenizer_hash": "tok",
        "model_identifier": "llava",
        "base_checkpoint_hash": "base",
        "expert_registry_hash": "registry",
        "expert_checkpoint_hashes": {"0": "h0", "1": "h1", "2": "h2"},
        "composition_mode": "direct_sum",
        "rms_statistics_hash": "none",
        "oracle_config_hash": "cfg",
        "answer_mask_version": "v1",
        "answer_template_hash": "tpl",
        "target_averaging": "token_mean",
        "composer_version": "v6",
        "code_version": "head",
        "pool_version": 2,
        "router_version": "v6_router_v1",
    }
    value.update(changes)
    return value


def _nll(mean: float, tokens: int = 5) -> AnswerNLL:
    return AnswerNLL(sum_nll=mean * tokens, mean_nll=mean, token_count=tokens)


def _searcher(scores, top_m=8, retrieved=(0, 1, 2)):
    """answer_nll_fn returns per-set means from ``scores``; any missing set
    gets a huge loss (not in candidate evaluation)."""
    config = _config()

    def answer_nll(expert_ids):
        key = tuple(sorted(int(value) for value in expert_ids))
        if key in scores:
            return _nll(scores[key])
        return _nll(100.0 + len(key))

    return V6TeacherSearcher(
        config=config,
        router_version="v6_router_v1",
        pool_version=2,
        answer_nll_fn=answer_nll,
        top_m=top_m,
        retrieve_fn=lambda: retrieved,
    )


class V6TeacherSearcherTest(unittest.TestCase):
    def test_teacher_empty_when_no_single_beats_empty(self):
        searcher = _searcher({(): 1.0, (0,): 1.1, (1,): 1.05})
        record = searcher.search(
            "s1", task_id=1, visible_expert_ids=[0, 1, 2], pool_size=3,
            provenance=_provenance(),
        )
        self.assertEqual(record.teacher_set, ())
        self.assertEqual(record.teacher_loss, 1.0)
        self.assertEqual(record.teacher_multi_hot, {})

    def test_teacher_single_when_pair_gain_insufficient(self):
        # single clearly beats empty; pair gain below delta_pair_raw.
        searcher = _searcher({
            (): 2.0, (0,): 1.0, (1,): 1.9, (0, 1): 0.99,
        })
        record = searcher.search(
            "s2", task_id=1, visible_expert_ids=[0, 1, 2], pool_size=3,
            provenance=_provenance(),
        )
        self.assertEqual(record.teacher_set, (0,))
        self.assertLessEqual(record.pair_gain, 0.02)

    def test_teacher_pair_when_conditional_gain_clear(self):
        searcher = _searcher({
            (): 2.0, (0,): 1.0, (1,): 1.5, (0, 1): 0.5,
        })
        record = searcher.search(
            "s3", task_id=1, visible_expert_ids=[0, 1, 2], pool_size=3,
            provenance=_provenance(),
        )
        self.assertEqual(record.teacher_set, (0, 1))
        self.assertGreater(record.pair_gain, 0.02)
        self.assertEqual(record.teacher_multi_hot, {0: 1, 1: 1})

    def test_retrieval_respects_top_m_and_visibility(self):
        searcher = _searcher(
            {(): 1.0, (0,): 0.5, (1,): 0.6}, top_m=8, retrieved=(1,)
        )
        record = searcher.search(
            "s4", task_id=1, visible_expert_ids=[0, 1, 2], pool_size=3,
            provenance=_provenance(),
        )
        self.assertEqual(record.candidate_experts, (1,))

    def test_retrieved_outside_visible_rejected(self):
        searcher = _searcher({(): 1.0}, retrieved=(5,))
        with self.assertRaisesRegex(ValueError, "not visible"):
            searcher.search(
                "s5", task_id=1, visible_expert_ids=[0, 1, 2], pool_size=3,
                provenance=_provenance(),
            )

    def test_expert_count_penalty_applies(self):
        # (0,) and (1,) have equal raw loss; penalty breaks the tie toward
        # the lower cardinality when scores are compared against empty.
        searcher = _searcher({(): 0.9, (0,): 0.8, (1,): 0.8, (0, 1): 0.7})
        record = searcher.search(
            "s6", task_id=1, visible_expert_ids=[0, 1, 2], pool_size=3,
            provenance=_provenance(),
        )
        # pair raw gain 0.1 > delta_pair_raw; penalized gain must exceed 0
        # for the pair to be valid.
        self.assertGreater(record.pair_gain, 0.02)

    def test_cache_key_binds_pool_and_router_versions(self):
        searcher = _searcher({(): 1.0, (0,): 0.5, (1,): 0.6, (0, 1): 0.4})
        base = _provenance()
        record_a = searcher.search(
            "s7", task_id=1, visible_expert_ids=[0, 1, 2], pool_size=3,
            provenance=base,
        )
        record_b = searcher.search(
            "s7", task_id=1, visible_expert_ids=[0, 1, 2], pool_size=3,
            provenance=dict(base, pool_version=3),
        )
        record_c = searcher.search(
            "s7", task_id=1, visible_expert_ids=[0, 1, 2], pool_size=3,
            provenance=dict(base, router_version="v6_router_v2"),
        )
        self.assertNotEqual(record_a.cache_key, record_b.cache_key)
        self.assertNotEqual(record_a.cache_key, record_c.cache_key)

    def test_record_fields_match_task_book(self):
        searcher = _searcher({(): 1.0, (0,): 0.5, (1,): 0.6, (0, 1): 0.4})
        record = searcher.search(
            "s8", task_id=1, visible_expert_ids=[0, 1, 2], pool_size=3,
            provenance=_provenance(),
        )
        payload = record.to_dict()
        for field in (
            "sample_id", "task_id", "pool_version", "router_version",
            "candidate_experts", "empty_loss", "single_losses", "pair_losses",
            "best_single", "best_pair", "pair_gain", "teacher_set",
            "teacher_loss", "teacher_multi_hot", "cache_key",
        ):
            self.assertIn(field, payload)
        self.assertEqual(payload["teacher_set"], [0, 1])


class TeacherCacheV6Test(unittest.TestCase):
    def test_cache_requires_pool_and_router_versions(self):
        missing = {key: value for key, value in _provenance().items()
                   if key not in ("pool_version", "router_version")}
        with self.assertRaisesRegex(ValueError, "pool_version"):
            cache_key(missing)
        with self.assertRaisesRegex(ValueError, "router_version"):
            cache_key({key: value for key, value in _provenance().items()
                       if key != "router_version"})

    def test_shard_round_trip_with_v6_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.json"
            write_shard(path, [{"sample_id": "a"}], _provenance(), 0)
            loaded = load_shard(path, _provenance())["records"]
            self.assertEqual(loaded, [{"sample_id": "a"}])


class RecallAuditTest(unittest.TestCase):
    def test_oracle_recall_at_m(self):
        samples = [
            {"sample_id": "a"},
            {"sample_id": "b"},
            {"sample_id": "c"},
            {"sample_id": "d"},
            {"sample_id": "e"},
        ]
        # full-pool best: a->(0), b->(1), c->(2), d->(), e->(0)
        def full_pool_eval(sample):
            best = {"a": (0,), "b": (1,), "c": (2,), "d": (), "e": (0,)}
            return best[str(sample["sample_id"])]

        retrieved = {
            "a": (0,), "b": (1,), "c": (2,), "d": (), "e": (2,),
        }
        result = run_recall_audit(
            samples, full_pool_eval, retrieved, top_m=2,
            pool_expert_ids=[0, 1, 2],
        )
        self.assertEqual(result.audited_samples, 5)
        self.assertEqual(result.oracles_with_support, 4)
        self.assertEqual(result.recalled, 3)  # e's best (0) not in Top-M (2)
        self.assertAlmostEqual(result.oracle_recall_at_m, 0.75)
        self.assertEqual(result.missed_contributing_expert_ids, [0])

    def test_recall_audit_requires_samples(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            run_recall_audit([], lambda s: (), {}, 2, [0, 1])

    def test_multi_hot_helper(self):
        self.assertEqual(
            build_teacher_multi_hot([0, 2], [0, 1, 2]), {0: 1, 1: 0, 2: 1}
        )


if __name__ == "__main__":
    unittest.main()
