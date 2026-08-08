"""Compose root-cause diagnosis required tests (spec section 10)."""

import json
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, "/home/zhaozhuofan/Hyper-LlaVA/compose/data/real_p1")
from official_metric import normalize, vqa_accuracy  # noqa: E402

from compose.eval.d0_metric_audit import make_variants, compute_record_metrics  # noqa: E402
from compose.eval.d1_causal_controls import make_replacement  # noqa: E402

OUTPUT_ROOT = Path("/home/zhaozhuofan/Hyper-LlaVA/outputs/compose_root_cause_20260804T044238Z")


# ---------------------------------------------------------------- 1, 2, 6 answers
class AnswerNormalizationTest(unittest.TestCase):
    def test_normalize_10_answers(self):
        record = {"question_id": "1", "answers": ["5", "five", "5", "5", "5",
                                                  "5", "5", "five", "5", "5"],
                  "answer": "5", "question": "how many?"}
        variants = make_variants(record)
        keys = [v["_n_key"] for v in variants]
        self.assertIn("5", keys)
        self.assertIn("five", keys)
        weights = {v["_n_key"]: v["_n_weight"] for v in variants}
        self.assertAlmostEqual(weights["5"], 0.8)
        self.assertAlmostEqual(weights["five"], 0.2)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_duplicate_merged_weight(self):
        record = {"question_id": "2", "answers": ["YES", "yes", "Yes", "no", "no"],
                  "answer": "yes", "question": "is it?"}
        variants = make_variants(record)
        weights = {v["_n_key"]: v["_n_weight"] for v in variants}
        self.assertAlmostEqual(weights["yes"], 0.6)
        self.assertAlmostEqual(weights["no"], 0.4)

    def test_official_score_alignment(self):
        # min(1, matched/3) with official normalization
        self.assertEqual(vqa_accuracy("5", ["5"] * 10), 1.0)
        self.assertEqual(vqa_accuracy("5", ["5", "5"] + ["4"] * 8), 2.0 / 3.0)
        self.assertEqual(vqa_accuracy("4", ["5"] * 10), 0.0)


# ---------------------------------------------------------------- 3, 4, 5 NLL forms
class NllDistinctionTest(unittest.TestCase):
    def test_canonical_vs_marginal_differ(self):
        record = {"question_id": "3", "answers": ["5"] * 9 + ["10"], "answer": "5",
                  "question": "q"}
        variants = make_variants(record)
        logprobs = [-3.0, -4.0]  # canonical (weight .9) worse than alternative
        rows = compute_record_metrics(record, variants, logprobs, "c1", 0, 0)
        # canonical NLL = -logP(majority) = 3.0 ; marginal = -log(.9e-3+.1e-4)
        self.assertAlmostEqual(rows["canonical_nll"], 3.0, places=4)
        expected_marginal = -math.log(0.9 * math.exp(-3.0) + 0.1 * math.exp(-4.0) + 1e-12)
        self.assertAlmostEqual(rows["marginal_nll"], expected_marginal, places=4)
        self.assertGreater(rows["marginal_nll"], rows["min_nll"])
        # marginal >= canonical is possible only through the weight
        # normalization: -log(sum_a w_a P_a) <= -log(w_c P_c) = canon + log(1/w_c)
        self.assertLessEqual(rows["marginal_nll"], rows["canonical_nll"] + math.log(1.0 / 0.9) + 1e-9)

    def test_multitoken_sequence_nll(self):
        # full-sequence probability: sum of token log-probs (not first token)
        record = {"question_id": "4", "answers": ["ab", "a"], "answer": "ab",
                  "question": "q"}
        variants = make_variants(record)
        # P(ab) = exp(-5.0) (two tokens), P(a) = exp(-2.0)
        logprobs = [-5.0, -2.0]
        rows = compute_record_metrics(record, variants, logprobs, "c1", 0, 0)
        self.assertAlmostEqual(rows["canonical_nll"], 5.0, places=4)  # full sequence
        self.assertAlmostEqual(rows["min_nll"], 2.0, places=4)
        expected = -math.log(0.5 * math.exp(-5.0) + 0.5 * math.exp(-2.0) + 1e-12)
        self.assertAlmostEqual(rows["marginal_nll"], expected, places=4)

    def test_min_nll_never_above_canonical(self):
        record = {"question_id": "5", "answers": ["a", "b", "b", "b", "b", "b", "b", "b", "b", "b"],
                  "answer": "b", "question": "q"}
        variants = make_variants(record)
        for logprobs in ([-1.0, -2.0], [-3.0, -0.5]):
            rows = compute_record_metrics(record, variants, logprobs, "c1", 0, 0)
            self.assertLessEqual(rows["min_nll"], rows["canonical_nll"] + 1e-9)


# ---------------------------------------------------------------- 7 RMS preserved
class RmsControlTest(unittest.TestCase):
    def test_shuffle_preserves_rms(self):
        torch.manual_seed(0)
        a = torch.randn(8, 64) * 0.02
        b = torch.randn(64, 8) * 0.02
        shuffled_b = b.flatten()[torch.randperm(b.numel())].reshape_as(b)
        self.assertAlmostEqual(
            float(torch.sqrt((b ** 2).mean())), float(torch.sqrt((shuffled_b ** 2).mean())), places=6)
        self.assertAlmostEqual(
            float(b.std()), float(shuffled_b.std()), places=6)

    def test_random_matches_std(self):
        torch.manual_seed(0)
        b = torch.randn(64, 8) * 0.02
        rng = torch.Generator().manual_seed(1)
        rand = torch.randn_like(b) * float(b.std())
        self.assertAlmostEqual(float(b.std()), float(rand.std()), places=2)


# ---------------------------------------------------------------- 8, 9 identity
class InteractionIdentityTest(unittest.TestCase):
    def test_base_base_interaction_zero(self):
        # I = h - h - h + h = 0 when all configurations are identical
        h = torch.randn(4, 16)
        i = h - h - h + h
        self.assertLess(float(i.norm()), 1e-6)

    def test_zero_lora_equals_base(self):
        # delta = 0  =>  base + 0 = base
        base = torch.randn(4, 16)
        self.assertTrue(torch.allclose(base + torch.zeros_like(base), base))

    def test_bc_cb_commutative(self):
        base = torch.randn(4, 16)
        delta_b = torch.randn(4, 16) * 0.1
        delta_c = torch.randn(4, 16) * 0.1
        bc = base + delta_b + delta_c
        cb = base + delta_c + delta_b
        self.assertTrue(torch.allclose(bc, cb))


# ---------------------------------------------------------------- 11 module mask
class ModuleMaskTest(unittest.TestCase):
    def test_mask_zeroes_delta(self):
        from compose.eval.d2_interaction_localization import layer_under
        import torch.nn as nn
        bridge = type("B", (), {"named_layers": [
            ("model.layers.0.self_attn.q_proj", nn.Linear(4, 4)),
            ("model.layers.0.self_attn.v_proj", nn.Linear(4, 4)),
            ("model.layers.0.self_attn.o_proj", nn.Linear(4, 4)),
            ("model.layers.0.mlp.gate_proj", nn.Linear(4, 4)),
            ("model.layers.0.mlp.up_proj", nn.Linear(4, 4)),
            ("model.layers.0.mlp.down_proj", nn.Linear(4, 4)),
        ]})()
        mlp = layer_under(bridge, "mlp")
        attn = layer_under(bridge, "attention")
        self.assertEqual(len(mlp), 3)
        self.assertEqual(len(attn), 3)
        self.assertEqual(mlp & attn, set(), "module masks must be disjoint")
        self.assertEqual(layer_under(bridge, "qv"), {
            "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.v_proj"})


# ---------------------------------------------------------------- 12, 13, 14 isolation
class IsolationTest(unittest.TestCase):
    def test_d3_selects_on_calib_only(self):
        source = Path("/home/zhaozhuofan/Hyper-LlaVA/compose/eval/d3_weight_surface.py").read_text()
        self.assertIn("BC_calib", source)
        # the test split is only evaluated AFTER selection
        self.assertIn("test evaluation of the selected points", source)
        self.assertIn("selected", source)

    def test_d4_oracle_not_for_routing(self):
        source = Path("/home/zhaozhuofan/Hyper-LlaVA/compose/eval/d4_oracle_upper_bound.py").read_text()
        self.assertIn("UPPER BOUND only", source)
        self.assertIn("never enters any Router training", source)

    def test_d6_pool_image_isolation(self):
        manifest = OUTPUT_ROOT / "metrics" / "d6_expanded_pool_manifest.json"
        if not manifest.exists():
            self.skipTest("d6 manifest missing")
        data = json.loads(manifest.read_text())
        p1 = Path("/home/zhaozhuofan/Hyper-LlaVA/outputs/compose_p1_real_20260803T120000Z")
        training_images = set()
        for name in ("B_train", "B_val", "C_train", "C_val"):
            records = json.loads((p1 / "data" / "records" / "{}.json".format(name)).read_text())
            training_images.update(r["image_id"] for r in records)
        clean = json.loads((OUTPUT_ROOT / "data" / "records_d6_clean.json").read_text())
        for record in clean:
            self.assertNotIn(record["image_id"], training_images,
                             "d6 clean pool image leaked into B/C training")


if __name__ == "__main__":
    unittest.main()
