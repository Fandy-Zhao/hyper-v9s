import unittest

from compose.eval.eval_controlled import exact_match, summarize


class ControlledEvalTest(unittest.TestCase):
    def test_exact_match_is_trimmed_and_case_insensitive(self):
        self.assertTrue(exact_match(" Circle ", "circle"))
        self.assertFalse(exact_match("three.", "three"))

    def test_summary(self):
        result = summarize([
            {"nll": 0.5, "correct": True, "generation_seconds": 0.2},
            {"nll": 1.5, "correct": False, "generation_seconds": 0.4},
        ], 2.0)
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["mean_nll"], 1.0)
        self.assertEqual(result["accuracy_percent"], 50.0)
        self.assertAlmostEqual(result["median_generation_seconds"], 0.3)
