"""Test I (spec §27): the Compose router has no global calibration -- no
bias/temperature head, no learned calibration block; selection is pure
cosine matching with the fixed config thresholds."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from compose.router.functional_query import ComposeQueryEncoder
from compose.router.router import (
    ComposeRouter,
    load_compose_router_checkpoint,
    save_compose_router_checkpoint,
)


def _router(seed=41):
    torch.manual_seed(seed)
    router = ComposeRouter(
        ComposeQueryEncoder(seed=seed), top_m=8, tau_none=0.5, tau_second=0.5
    )
    for expert_id in range(3):
        router.add_expert(
            expert_id, creation_task=0, checkpoint_sha256="sha",
            key=F.normalize(torch.randn(128), dim=0),
        )
    return router


class NoGlobalCalibrationTest(unittest.TestCase):
    def test_router_has_no_calibration_module(self):
        router = _router()
        names = [name for name, _ in router.named_parameters()]
        self.assertFalse(any("calibr" in name for name in names))
        self.assertFalse(any("temperat" in name for name in names))
        # The query encoder's Linear biases are legitimate structure; a
        # router-level bias head (outside the encoder) is what's forbidden.
        router_level = [name for name in names if not name.startswith("query_encoder")]
        self.assertFalse(any("bias" in name for name in router_level))
        # Only the query encoder + the expert keys exist.
        expected_prefixes = {"query_encoder", "key_store"}
        self.assertTrue(
            all(name.split(".")[0] in expected_prefixes for name in names),
            names,
        )

    def test_checkpoint_has_no_calibration_block(self):
        router = _router()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "router.pt")
            save_compose_router_checkpoint(path, router, pool_version=1, config_hash="cfg")
            state = torch.load(path, map_location="cpu", weights_only=False)
        self.assertNotIn("calibration", state)
        self.assertNotIn("global_bias", state)
        self.assertNotIn("temperature", state)
        self.assertEqual(state["kind"], "compose_router")

    def test_selection_depends_only_on_queries_and_keys(self):
        router = _router()
        keys = [router.key_store.keys[str(i)].detach() for i in range(3)]
        # Identical query -> identical selection (functional, deterministic).
        query = F.normalize(0.9 * keys[0] + 0.5 * keys[1], dim=0).unsqueeze(0)
        first = router.select(query, router.expert_ids)
        second = router.select(query, router.expert_ids)
        self.assertEqual(first.sets, second.sets)
        self.assertTrue(torch.equal(first.probabilities, second.probabilities))
        # Perturbing the query changes the matching (pure cosine).
        altered = F.normalize(0.9 * keys[2] + 0.1 * keys[0], dim=0).unsqueeze(0)
        self.assertNotEqual(
            first.sets, router.select(altered, router.expert_ids).sets
        )

    def test_thresholds_are_config_constants(self):
        router = _router()
        self.assertEqual(router.tau_none, 0.5)
        self.assertEqual(router.tau_second, 0.5)
        # The router has no learnable threshold parameters.
        threshold_params = [
            name for name, parameter in router.named_parameters()
            if "tau" in name or "threshold" in name
        ]
        self.assertEqual(threshold_params, [])

    def test_checkpoint_round_trip_preserves_no_calibration(self):
        router = _router()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "router.pt")
            save_compose_router_checkpoint(path, router, pool_version=1, config_hash="cfg")
            restored = ComposeRouter(ComposeQueryEncoder(seed=41))
            load_compose_router_checkpoint(path, restored)
        self.assertEqual(restored.expert_ids, router.expert_ids)
        self.assertEqual(restored.tau_none, 0.5)
        self.assertEqual(restored.pool_version, 1)


if __name__ == "__main__":
    unittest.main()
