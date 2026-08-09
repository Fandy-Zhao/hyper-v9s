"""BLOCKING-bug regression: duplicate question_ids in UCIT train files.

VizWiz / Flickr30k / CLEVR train.json repeat question_ids (32.7% /
31.9% / 1.5% of records). The pipeline keys features, teacher selections,
residuals, cluster membership and the training manifest by sample_id
(= record["id"] when present, else question_id); two records sharing a
question_id collide and manifest construction crashes:

    ValueError: duplicate sample 00007196 in training manifest

reproduced deterministically on the formal seed42 run, task2 (VizWiz)
S5, 2026-08-09 (clusters.json held 71 duplicated sample_ids across the
two clusters; 2000 assignments / 1929 unique ids).

The fix (task_run.S1) assigns every record a unique deterministic
internal id "<question_id>#<absolute index in the train file>" before
features are computed; every downstream keying site (query_features,
nll_eval, selections, residuals, cluster manifest, training manifest,
compose selection dataset) is id-first, so the annotated id flows
through untouched.
"""

import unittest

from compose.experiments.task_run import _assign_unique_record_ids
from compose.expansion.expert_formation import (
    FormedExpert,
    build_training_manifest,
)


def _expert(sample_ids, expert_id=7):
    return FormedExpert(
        expert_id=expert_id,
        cluster_id=0,
        sample_ids=tuple(sample_ids),
        size=len(sample_ids),
        centroid=(0.0,) * 128,
        key_mode="learnable",
        creation_task=2,
    )


class DuplicateSampleIdRegressionTest(unittest.TestCase):
    def test_duplicate_question_ids_get_unique_deterministic_ids(self):
        """VizWiz-like input: repeated question_ids must yield unique ids,
        preserving order and original fields."""
        records = [
            {"question_id": "00015026", "text": "q0"},
            {"question_id": "00019984", "text": "q1"},
            {"question_id": "00015026", "text": "q0-dup"},
            {"question_id": "00019984", "text": "q1-dup"},
            {"question_id": "00000001", "text": "q2"},
        ]
        annotated = _assign_unique_record_ids(records)
        ids = [record["id"] for record in annotated]
        self.assertEqual(len(ids), len(set(ids)), ids)
        # original fields are preserved; id keeps the question_id visible
        self.assertEqual(annotated[0]["text"], "q0")
        self.assertEqual(annotated[0]["question_id"], "00015026")
        self.assertTrue(annotated[0]["id"].startswith("00015026#"))
        # deterministic across calls
        again = _assign_unique_record_ids(records)
        self.assertEqual(ids, [record["id"] for record in again])

    def test_manifest_build_succeeds_with_unique_ids(self):
        """The crash scenario, fixed: the same question_id appears twice in
        the residual set but each record has its own id; the manifest
        builds without raising and keeps both records."""
        records = [
            {"sample_id": "00015026#0", "old_teacher_set": ()},
            {"sample_id": "00015026#2", "old_teacher_set": ()},
        ]
        manifest = build_training_manifest(
            [_expert(tuple(r["sample_id"] for r in records))], records
        )
        self.assertEqual(len(manifest), 2)
        self.assertEqual(
            [row["sample_id"] for row in manifest],
            ["00015026#0", "00015026#2"],
        )

    def test_manifest_build_raises_on_colliding_sample_ids(self):
        """Guard documentation: with the raw colliding ids the manifest
        builder still rejects the ambiguous input."""
        records = [
            {"sample_id": "00015026", "old_teacher_set": ()},
            {"sample_id": "00015026", "old_teacher_set": ()},
        ]
        with self.assertRaises(ValueError):
            build_training_manifest([_expert(("00015026",))], records)


if __name__ == "__main__":
    unittest.main()
