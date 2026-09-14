"""The residual characteriser decides a verdict, so its edges need pinning.

``compare_recipe`` says *whether* two arms agree.  This module says what a "no"
means, and the two readings it separates -- a diverging trajectory and a
stationary rounding floor -- are the same number at any single window.  A
characteriser that cannot tell them apart would let a real divergence through
labelled as arithmetic, which is the failure mode the whole equivalence exercise
exists to prevent.  So the tests below pin the three statistics against cases
whose answer is known by construction:

* a synthetic residual with a *known* mean and spread must come back with that
  mean, that spread, and a t-statistic of the size they imply;
* a residual built by *adding noise to a fixed floor* must show a flat absolute
  column, however the relative column behaves;
* a residual built by *compounding* must grow the absolute column, which is the
  reading that must never be available to a real divergence;
* a partially written window must be dropped, because it is not a residual at
  all and it is large enough to dominate a max-over-windows statistic.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from compose.experiments.window_residual import (  # noqa: E402
    _blocks,
    _complete_windows,
    analyse,
)


def _series(pairs, left=1.0):
    """``(step, left value, right value, rel, abs)`` rows from relative values."""
    return [
        (index, left, left * (1.0 + rel), rel, left * rel)
        for index, rel in enumerate(pairs)
    ]


class TestCompleteness:
    def test_a_short_final_window_is_dropped(self):
        """A live run appends micro-steps, so its last window is a fraction.

        Averaging 3 of 8 micro-steps is an average over different rows, which
        moves that window by whole percent -- an editorial artefact with no
        bearing on the recipe that would otherwise own the max column.
        """
        left = {0: [{}] * 4, 1: [{}] * 4, 2: [{}] * 4}
        right = {0: [{}] * 2, 1: [{}] * 2, 2: [{}] * 1}
        complete, dropped = _complete_windows(left, right, steps=0)
        assert complete == [0, 1]
        assert dropped == [2]

    def test_a_fully_written_run_drops_nothing(self):
        left = {index: [{}] * 4 for index in range(5)}
        right = {index: [{}] * 2 for index in range(5)}
        complete, dropped = _complete_windows(left, right, steps=0)
        assert complete == [0, 1, 2, 3, 4]
        assert dropped == []

    def test_steps_limit_is_applied_before_dropping_is_reported(self):
        """``--steps 2`` must not report window 5 as an incomplete one."""
        left = {index: [{}] * 4 for index in range(6)}
        right = {index: [{}] * 2 for index in range(6)}
        complete, dropped = _complete_windows(left, right, steps=2)
        assert complete == [0, 1]
        assert dropped == []

    def test_windows_only_one_arm_has_are_not_compared(self):
        left = {0: [{}] * 4, 1: [{}] * 4}
        right = {0: [{}] * 2}
        complete, _ = _complete_windows(left, right, steps=0)
        assert complete == [0]


class TestStatistics:
    def test_a_known_mean_and_spread_come_back_unchanged(self):
        # symmetric +/-5e-3, so the mean is exactly zero and the sd is 5e-3
        rels = [5e-3, -5e-3] * 50
        result = analyse(_series(rels), "answer_loss")
        assert result["n"] == 100
        assert result["mean_rel"] == pytest.approx(0.0, abs=1e-12)
        assert result["sd_rel"] == pytest.approx(5e-3, rel=1e-9)
        assert result["t_statistic"] == pytest.approx(0.0, abs=1e-9)

    def test_a_systematic_offset_is_visible_as_a_large_t(self):
        """The reading that distinguishes a moved objective from rounding."""
        rels = [1e-2, 1e-2, 1e-2, 1e-2 + 1e-9] * 25
        result = analyse(_series(rels), "answer_loss")
        assert abs(result["t_statistic"]) > 4.0

    def test_extreme_value_prediction_is_projected_to_every_ladder_rung(self):
        result = analyse(_series([5e-3, -5e-3] * 50), "answer_loss")
        prediction = result["extreme_value_prediction"]
        assert set(prediction) == {"100", "500", "1000"}
        # sigma*sqrt(2 ln n) grows with n, so the projection must too
        assert prediction["100"] < prediction["500"] < prediction["1000"]
        # ...and only logarithmically: 10x the windows must not double it
        assert prediction["1000"] < 2.0 * prediction["100"]

    def test_the_observed_max_is_compared_against_its_prediction(self):
        """A stationary process' worst window sits under its own prediction."""
        rels = [5e-3, -5e-3] * 50
        result = analyse(_series(rels), "answer_loss")
        assert result["max_abs_rel"] == pytest.approx(5e-3, rel=1e-9)
        assert result["extreme_value_ratio"] < 1.0


class TestGrowthReading:
    def test_a_fixed_floor_under_a_decaying_loss_does_not_grow_the_absolute_column(self):
        """The signature §3.6 reads as arithmetic rather than divergence.

        The loss decays, so a constant absolute perturbation grows *relative* to
        it.  If the absolute column were read as the verdict this case would be
        indistinguishable from compounding -- which is exactly why both columns
        are reported.
        """
        abs_noise = 3e-3
        series = [
            (index, 2.0 - 1.6 * index / 99, 0.0, 0.0, abs_noise)
            for index in range(100)
        ]
        blocks = _blocks(series)
        absolute = [block["mean_abs_abs"] for block in blocks]
        assert max(absolute) == pytest.approx(min(absolute), rel=1e-9)

    def test_compounding_grows_the_absolute_column(self):
        """A divergence must show here, and must not be mistakeable for the above."""
        series = [
            (index, 2.0 - 1.6 * index / 99, 0.0, 0.0, 1e-4 * (1.5 ** (index / 10)))
            for index in range(100)
        ]
        absolute = [block["mean_abs_abs"] for block in _blocks(series)]
        assert absolute[-1] > 3.0 * absolute[0]

    def test_blocks_partition_the_series_without_gaps(self):
        blocks = _blocks(_series([1e-3] * 100))
        assert len(blocks) == 5
        assert sum(block["n"] for block in blocks) == 100
        for earlier, later in zip(blocks, blocks[1:]):
            assert later["first_step"] == earlier["last_step"] + 1

    def test_an_empty_series_reports_nothing_rather_than_raising(self):
        assert analyse([], "answer_loss")["n"] == 0
        assert _blocks([]) == []
