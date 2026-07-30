import unittest

from compose.eval.summarize_controlled import bootstrap_ci


class SummarizeControlledTest(unittest.TestCase):
    def test_bootstrap_interval_is_deterministic_and_positive(self):
        values = [0.1, 0.2, 0.3, 0.4]
        first = bootstrap_ci(values, seed=7, draws=200)
        second = bootstrap_ci(values, seed=7, draws=200)
        self.assertEqual(first, second)
        self.assertGreater(first[0], 0)

    def test_bootstrap_interval_preserves_negative_direction(self):
        interval = bootstrap_ci([-0.4, -0.3, -0.2, -0.1], seed=8, draws=200)
        self.assertLess(interval[1], 0)
