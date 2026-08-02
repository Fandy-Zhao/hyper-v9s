import json
import tempfile
import unittest
from pathlib import Path

from compose.experts import (
    ExpertMetadata,
    ExpertRegistry,
    load_registry_checkpoint,
    save_registry_checkpoint,
)


class RegistryCheckpointTest(unittest.TestCase):
    def test_atomic_json_round_trip_has_provenance_and_checksum(self):
        registry = ExpertRegistry()
        registry.register(ExpertMetadata(expert_id=3, adapter_name="three"))
        registry.set_active_ids([3])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            checksum = save_registry_checkpoint(
                registry,
                path,
                source_git_commit="5668cfd",
                source_model_identifier="test-model",
                source_adapter_config_hash="abc123",
                timestamp="2026-08-02T00:00:00+00:00",
                run_id="stage02-test",
            )
            restored, provenance, loaded_checksum = load_registry_checkpoint(path)
            self.assertEqual(checksum, loaded_checksum)
            self.assertEqual(restored.active_expert_ids, (3,))
            self.assertEqual(provenance["source_git_commit"], "5668cfd")
            self.assertNotIn("weights", json.loads(path.read_text()))
            with self.assertRaises(FileExistsError):
                registry.save_json(path)

    def test_missing_checkpoint_is_clear(self):
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            load_registry_checkpoint("/definitely/missing/registry.json")


if __name__ == "__main__":
    unittest.main()
