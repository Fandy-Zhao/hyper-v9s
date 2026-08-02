import unittest

from compose.experts.metadata import ExpertMetadata, ExpertStatus


class ExpertMetadataTest(unittest.TestCase):
    def test_schema_is_json_safe_and_deduplicates_reuse_tasks(self):
        metadata = ExpertMetadata(
            expert_id=2,
            adapter_name="ucit-2",
            rank=8,
            alpha=16,
            reuse_tasks=[3, 1, 3],
        )
        self.assertEqual(metadata.reuse_tasks, [3, 1])
        self.assertEqual(metadata.to_dict()["status"], "registered")

    def test_unknown_fields_are_preserved_in_extra(self):
        restored = ExpertMetadata.from_dict(
            {"expert_id": 0, "adapter_name": "zero", "future_field": 7}
        )
        self.assertEqual(restored.extra["future_field"], 7)

    def test_legacy_fields_and_status_remain_loadable(self):
        restored = ExpertMetadata.from_dict(
            {"expert_id": 1, "name": "legacy", "status": "training"}
        )
        self.assertEqual(restored.adapter_name, "legacy")
        self.assertIs(restored.status, ExpertStatus.TRAINABLE)

    def test_invalid_values_and_archived_selection_fail(self):
        for kwargs in ({"rank": 0}, {"alpha": 0}, {"support_count": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ExpertMetadata(expert_id=0, adapter_name="bad", **kwargs)
        with self.assertRaisesRegex(ValueError, "archived"):
            ExpertMetadata(
                expert_id=0,
                adapter_name="bad",
                status=ExpertStatus.ARCHIVED,
                active=True,
            )


if __name__ == "__main__":
    unittest.main()
