"""The evidence cache must return *computed* values, never plausible ones.

Every property that makes the pruning acceleration safe is a property of this
module, so it is tested directly: a hit is byte-identical to what a miss would
have produced, a different checkpoint cannot reach a previous checkpoint's
entries, and an unreadable entry degrades to a recomputation rather than to a
wrong answer.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from compose.eval.evidence_cache import (  # noqa: E402
    EvidenceCache,
    build_fingerprint,
    evidence_key,
    split_by_cache,
)


def _entries(marker="checkpoint-a"):
    return {"checkpoint_weights_sha256": marker, "question_file_sha256": "q", "scoring": {"k": 1}}


class EvidenceCacheTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="evidence-cache-test-")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _cache(self, marker="checkpoint-a"):
        return EvidenceCache(os.path.join(self.root, "cache"), _entries(marker))

    def test_a_hit_returns_exactly_the_stored_payload(self):
        cache = self._cache()
        payload = {"mean_answer_nll": 1.2345678901234567, "supervised_token_count": 11}
        key = evidence_key("sample-1", [20, 21])
        cache.put(key, payload)

        fresh = self._cache()
        self.assertEqual(fresh.get(key), payload)
        # Not merely equal: the same JSON value, so the persisted file is
        # byte-identical to what a recomputation would have written.
        self.assertEqual(json.loads(json.dumps(fresh.get(key))), json.loads(json.dumps(payload)))

    def test_a_disabled_cache_never_hits(self):
        cache = EvidenceCache(None, _entries())
        key = evidence_key("sample-1", [20, 21])
        cache.put(key, {"text": "a caption"})
        self.assertIsNone(cache.get(key))
        self.assertEqual(cache.stats()["hits"], 0)

    def test_disabled_cache_does_not_touch_the_fingerprint(self):
        """The fingerprint hashes a multi-GB checkpoint; off means not computed."""
        calls = []

        def entries():
            calls.append(1)
            return _entries()

        cache = EvidenceCache(None, entries)
        self.assertFalse(cache.enabled)
        self.assertEqual(calls, [])

    def test_expert_order_does_not_change_the_key(self):
        """An expert *set* is a set: [20,21] and [21,20] are the same evidence."""
        self.assertEqual(evidence_key("s", [20, 21]), evidence_key("s", [21, 20]))

    def test_different_expert_sets_do_not_collide(self):
        self.assertNotEqual(evidence_key("s", [20, 21]), evidence_key("s", [20, 22]))
        self.assertNotEqual(evidence_key("s", [20, 21]), evidence_key("s", [20]))

    def test_a_different_fingerprint_cannot_reach_previous_entries(self):
        key = evidence_key("sample-1", [20, 21])
        self._cache("checkpoint-a").put(key, {"text": "from A"})

        other = self._cache("checkpoint-b")
        self.assertNotEqual(other.fingerprint, self._cache("checkpoint-a").fingerprint)
        self.assertIsNone(other.get(key), "a new checkpoint must miss, not inherit")

    def test_corrupt_entry_is_a_miss_not_a_wrong_answer(self):
        cache = self._cache()
        key = evidence_key("sample-1", [20, 21])
        cache.put(key, {"text": "fine"})
        path = cache._path(key)
        path.write_text("{not json", encoding="utf-8")

        reloaded = self._cache()
        self.assertIsNone(reloaded.get(key))

    def test_fingerprint_descriptor_is_written_for_audit(self):
        cache = self._cache()
        descriptor = cache.root / "fingerprint.json"
        self.assertTrue(descriptor.is_file())
        self.assertEqual(json.loads(descriptor.read_text())["checkpoint_weights_sha256"],
                         "checkpoint-a")

    def test_split_by_cache_partitions_without_reordering(self):
        cache = self._cache()
        hit_key = evidence_key("s", [1, 2])
        cache.put(hit_key, {"cached": True})

        hits, misses = split_by_cache(
            cache, "s", [("global_top2", [1, 2]), ("other", [3, 4])]
        )
        self.assertEqual(hits, {"global_top2": {"cached": True}})
        self.assertEqual(misses, [("other", [3, 4])])

    def test_second_cache_instance_reads_the_first_ones_writes(self):
        """The whole point: one job's evidence serves the next job."""
        first = self._cache()
        key = evidence_key("sample-7", [22, 3])
        first.put(key, {"text": "shared"})
        second = self._cache()
        self.assertEqual(second.get(key), {"text": "shared"})
        self.assertEqual(second.stats(), {"hits": 1, "misses": 0, "requests": 1, "hit_rate": 1.0})

    def test_build_fingerprint_is_order_insensitive(self):
        self.assertEqual(
            build_fingerprint({"a": 1, "b": 2}), build_fingerprint({"b": 2, "a": 1})
        )
        self.assertNotEqual(build_fingerprint({"a": 1}), build_fingerprint({"a": 2}))


if __name__ == "__main__":
    unittest.main()
