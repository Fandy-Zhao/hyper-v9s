"""The A/B gate has to be trustworthy before its verdict means anything.

A comparator that reports "equivalent" for everything would pass the V8 brief
while proving nothing, so these tests pin its behaviour at the two edges that
matter: it must call a reordered data stream a MISMATCH even when every loss is
identical, and it must call a pure summation-order change EQUIVALENT.
"""

import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from compose.experiments.compare_recipe import (  # noqa: E402
    compare_arms,
    compare_windows,
    load_arm,
    optimizer_steps,
)

ACCUM = 4


def _row(step, sample, answer, key, routes=("NewNew",), experts=((16, 18),)):
    return {
        "step": step,
        "sample_ids": [sample],
        "route_types": list(routes),
        "selected_expert_ids": [list(e) for e in experts],
        # The trainer writes this as the sorted *union* of the current-task
        # experts selected anywhere in the micro-batch, so a one-sample row
        # holding two experts stores two ids.  It is deliberately not
        # per-sample shaped here, exactly as the real records are not.
        "selected_current_ids": [16, 18],
        "old_old_noop": False,
        "local_batch_size": 1,
        "gradient_window_synced": False,
        "total_loss": answer + 0.1 * key,
        "answer_loss": answer,
        "key_loss": key,
        "selected_current_key_grad_norm": 1e-3,
        "selected_current_lora_grad_norm": 0.2,
    }


def _wide_row(step, samples, answer, key, experts):
    """One micro-step row covering several samples, as a bs>1 arm writes them.

    ``compose/v7/hf_trainer.py`` stores ``key_loss`` as a sum over the
    micro-batch (``key_loss = per_sample.sum()``), so by default the caller
    passes a value that grows with the width.  Pass a constant instead to model
    the per-micro-batch mean the trainer would need for the objective to be
    width-invariant.
    """
    return {
        "step": step,
        "sample_ids": list(samples),
        "route_types": ["NewNew"] * len(samples),
        "selected_expert_ids": [list(e) for e in experts],
        "selected_current_ids": sorted({e for group in experts for e in group}),
        "old_old_noop": False,
        "local_batch_size": len(samples),
        "gradient_window_synced": False,
        "total_loss": answer + 0.1 * key,
        "answer_loss": answer,
        "key_loss": key,
        "selected_current_key_grad_norm": 1e-3,
        "selected_current_lora_grad_norm": 0.2,
    }


def _split_arm(windows, width, world=1, convention="mixed", offset=0.0):
    """Lay the same windows out at a given micro-batch ``width`` and ``world``.

    Every sample keeps the same losses and the same routing decision; only how
    many samples share a row, and which rank holds them, changes.

    ``convention`` is the one the trainer stores its losses in.  ``"mixed"`` is
    what ``compose/v7/hf_trainer.py`` does today, and the mix is the whole
    problem: ``key_loss`` is a per-micro-batch sum, so its row value grows with
    the width, while the answer loss is the model's per-micro-batch mean, which
    does not.  ``"means"`` is the trainer with that fixed, and is the only
    convention for which the objective is width-invariant.
    """
    ranks = {rank: [] for rank in range(world)}
    for step, samples in enumerate(windows):
        for position in range(0, len(samples), width):
            group = samples[position : position + width]
            key_scale = len(group) if convention == "mixed" else 1
            row = _wide_row(
                step,
                group,
                answer=0.5 + offset,
                key=0.05 * key_scale + offset,
                experts=[(16, 18)] * len(group),
            )
            ranks[(position // width) % world].append(row)
    return ranks


def _arm(rows_per_rank):
    return {rank: rows for rank, rows in enumerate(rows_per_rank)}


def _stream(offset=0.0, order=None):
    samples = ["s{}".format(i) for i in range(ACCUM * 2)]
    if order is not None:
        samples = [samples[i] for i in order]
    return [_row(1, name, 0.5 + offset, 0.05) for name in samples]


def test_identical_arms_are_bit_identical():
    rows = _stream()
    result = compare_arms(_arm([rows]), _arm([rows]), ACCUM, 0, 1e-6, 1e-3)
    assert result["verdict"] == "BIT_IDENTICAL"
    assert result["ranks"]["rank0"]["optimizer_steps"]["total_loss"]["bits_identical"]


def test_reordered_stream_is_a_mismatch_even_with_identical_losses():
    """The reordering that a broken sampler would cause must not slip through."""
    left = _stream()
    right = _stream(order=[i for i in reversed(range(ACCUM * 2))])
    result = compare_arms(_arm([left]), _arm([right]), ACCUM, 0, 1e-6, 1e-3)
    assert result["verdict"] == "MISMATCH"
    detail = result["ranks"]["rank0"]["discrete"]["sample_ids"]
    assert detail["num_differing"] > 0
    assert detail["first_divergence"] == 0


def test_summation_order_noise_is_equivalent_not_mismatch():
    left = _stream()
    right = _stream(offset=1e-5)
    result = compare_arms(_arm([left]), _arm([right]), ACCUM, 0, 1e-4, 1e-3)
    assert result["verdict"] == "EQUIVALENT"
    assert not result["ranks"]["rank0"]["optimizer_steps"]["total_loss"]["bits_identical"]


def test_route_change_is_a_mismatch():
    left = _stream()
    right = _stream()
    right[3]["route_types"] = ["Residual"]
    right[3]["selected_expert_ids"] = [[16]]
    result = compare_arms(_arm([left]), _arm([right]), ACCUM, 0, 1e-6, 1e-3)
    assert result["verdict"] == "MISMATCH"
    assert result["ranks"]["rank0"]["discrete"]["route_types"]["first_divergence"] == 3


def test_large_numeric_drift_is_a_mismatch_even_with_matching_routes():
    left = _stream()
    right = _stream(offset=0.5)
    result = compare_arms(_arm([left]), _arm([right]), ACCUM, 0, 1e-6, 1e-3)
    assert result["verdict"] == "MISMATCH"
    assert "loss or gradient norm" in result["reasons"][0]


def test_step_limit_truncates_both_arms():
    rows = _stream()
    result = compare_arms(_arm([rows]), _arm([rows]), ACCUM, steps=1, abs_tol=0, rel_tol=0)
    assert result["ranks"]["rank0"]["rows_compared"] == ACCUM
    assert result["ranks"]["rank0"]["optimizer_steps"]["compared_steps"] == 1


def test_rank_sets_must_agree():
    with pytest.raises(SystemExit):
        compare_arms(_arm([_stream()]), _arm([_stream(), _stream()]), ACCUM, 0, 1e-6, 1e-3)


def test_optimizer_steps_average_within_the_window():
    rows = [_row(1, "s", 0.0, 0.0), _row(1, "s", 1.0, 1.0)]
    windows = optimizer_steps(rows, 2)
    assert windows == [
        {
            "micro_start": 0,
            "micro_end": 1,
            "total_loss": 0.55,
            "answer_loss": 0.5,
            "key_loss": 0.5,
        }
    ]


def _window_arm(windows, world=1, offset=0.0):
    """One arm, samples dealt round-robin across ``world`` ranks.

    This is what the real trainer produces when the sampler's megabatch is split
    differently: the same samples, a different rank and a different position
    inside the window.
    """
    ranks = {rank: [] for rank in range(world)}
    for step, samples in enumerate(windows):
        for position, sample in enumerate(samples):
            row = _row(step, sample, 0.5 + offset, 0.05)
            row["selected_expert_ids"] = [[16, 18]]
            ranks[position % world].append(row)
    return ranks


WINDOWS = [["s{}".format(i) for i in range(8)], ["s{}".format(i) for i in range(8, 16)]]


def test_window_mode_survives_a_different_rank_split_and_order():
    """The S6 case: same samples per optimizer step, regrouped and reordered."""
    left = _window_arm(WINDOWS, world=1)
    reordered = [list(reversed(window)) for window in WINDOWS]
    right = _window_arm(reordered, world=4, offset=1e-6)
    result = compare_windows(left, right, 0, 1e-4, 1e-3)
    assert result["verdict"] == "EQUIVALENT"
    assert result["sample_sets_identical"]
    assert result["routing_identical"]
    assert not result["losses"]["total_loss"]["bit_identical"]


def test_window_mode_is_bit_identical_when_nothing_moves():
    result = compare_windows(_window_arm(WINDOWS, 2), _window_arm(WINDOWS, 2), 0, 0, 0)
    assert result["verdict"] == "BIT_IDENTICAL"


def test_window_mode_catches_a_swapped_sample():
    """A window that swaps one sample for another must not pass as equivalent."""
    swapped = [list(WINDOWS[0]), list(WINDOWS[1])]
    swapped[1][3] = "s999"
    result = compare_windows(_window_arm(WINDOWS, 1), _window_arm(swapped, 1), 0, 1e-4, 1e-3)
    assert result["verdict"] == "MISMATCH"
    assert not result["sample_sets_identical"]
    assert result["num_steps_with_different_samples"] == 1
    assert "different samples" in result["reasons"][0]


def test_window_mode_catches_a_routing_flip_on_one_sample():
    """Same samples, same losses, one different routing decision."""
    right = _window_arm(WINDOWS, 1)
    right[0][5]["route_types"] = ["Residual"]
    right[0][5]["selected_current_ids"] = [16]
    result = compare_windows(_window_arm(WINDOWS, 1), right, 0, 1e-4, 1e-3)
    assert result["verdict"] == "MISMATCH"
    assert result["sample_sets_identical"]
    assert not result["routing_identical"]
    assert "routing decision differs" in result["reasons"][0]


def test_window_mode_catches_a_loss_drift():
    result = compare_windows(
        _window_arm(WINDOWS, 1), _window_arm(WINDOWS, 1, offset=0.25), 0, 1e-6, 1e-3
    )
    assert result["verdict"] == "MISMATCH"
    assert result["sample_sets_identical"] and result["routing_identical"]
    assert "moved beyond" in result["reasons"][0]


def test_window_mode_reads_the_per_sample_expert_field():
    """The regression that made this comparator lie.

    ``selected_current_ids`` is a per-row union whose length tracks the batch
    width, not the sample count.  Reading it positionally paired every sample
    with an unrelated expert at any width above one, so two arms that agreed
    perfectly were reported as a routing MISMATCH.  The per-sample decision
    lives in ``selected_expert_ids``.
    """
    left = _split_arm(WINDOWS, width=1)
    right = _split_arm(WINDOWS, width=4)
    result = compare_windows(left, right, 0, 1e-6, 1e-3)
    assert result["sample_sets_identical"]
    assert result["routing_identical"], result["routing_divergences"]
    assert result["routing_unverifiable_samples"] == 0


def test_window_mode_catches_a_sum_field_amplified_by_the_micro_batch():
    """The defect the gate exists to catch, pinned as a test.

    ``compose/v7/hf_trainer.py`` stores ``key_loss`` as a per-micro-batch *sum*
    while the answer loss is the model's per-micro-batch *mean*.  HF divides
    every micro-step by ``gradient_accumulation_steps`` before backward, and
    widening the micro-batch shrinks ``GA`` by the same factor, so nothing
    cancels: the summed term reaches the optimizer multiplied by the width --
    an effective key weight of 0.4 instead of 0.1 at micro 4.  The window-mean
    objective sees it; the answer term, which is a genuine mean, stays flat and
    must not be blamed for it.
    """
    left = _split_arm(WINDOWS, width=1)
    right = _split_arm(WINDOWS, width=4)
    result = compare_windows(left, right, 0, 1e-6, 1e-3)
    assert result["sample_sets_identical"] and result["routing_identical"]
    assert result["verdict"] == "MISMATCH", result["reasons"]
    key = result["losses"]["key_loss"]
    assert key["objective_ratio"] > 3.9, key
    assert key["scales_with_row_width"] is True
    assert abs(result["losses"]["answer_loss"]["max_rel_diff"]) < 1e-12
    assert result["losses"]["answer_loss"]["scales_with_row_width"] is False
    assert "scale(s) with the row width" in result["reasons"][0]


def test_window_mode_is_equivalent_for_mean_fields_across_widths():
    """The same A/B once the trainer stores per-micro-batch means.

    This is the shape the micro-batch lever has to take to be shippable, and the
    test above is the shape it must not.
    """
    left = _split_arm(WINDOWS, width=1, convention="means")
    right = _split_arm(WINDOWS, width=4, convention="means", offset=1e-6)
    result = compare_windows(left, right, 0, 1e-4, 1e-3)
    assert result["verdict"] == "EQUIVALENT", result["reasons"]
    assert result["losses"]["total_loss"]["scales_with_row_width"] is False
    assert not result["losses"]["total_loss"]["bit_identical"]


def test_window_mode_still_catches_a_drift_in_mean_fields():
    right = _split_arm(WINDOWS, width=4, convention="means", offset=0.25)
    result = compare_windows(
        _split_arm(WINDOWS, width=1, convention="means"), right, 0, 1e-6, 1e-3
    )
    assert result["verdict"] == "MISMATCH"
    assert result["sample_sets_identical"] and result["routing_identical"]
    assert "moved beyond" in result["reasons"][0]


def test_window_mode_still_catches_a_routing_flip_at_a_wider_micro_batch():
    """Widening the rows must not blind the gate to a genuine flip."""
    right = _split_arm(WINDOWS, width=4)
    row = right[0][1]
    row["selected_expert_ids"][2] = [17, 19]
    result = compare_windows(_split_arm(WINDOWS, width=4), right, 0, 1e-6, 1e-3)
    assert result["verdict"] == "MISMATCH"
    assert result["sample_sets_identical"]
    assert not result["routing_identical"]
    assert "routing decision differs" in result["reasons"][0]


def test_window_mode_still_catches_a_loss_drift_at_a_wider_micro_batch():
    right = _split_arm(WINDOWS, width=4)
    right[0][0]["answer_loss"] += 0.25
    right[0][0]["total_loss"] += 0.25
    result = compare_windows(_split_arm(WINDOWS, width=4), right, 0, 1e-6, 1e-3)
    assert result["verdict"] == "MISMATCH"
    assert "moved beyond" in result["reasons"][0]


def test_window_mode_reports_routing_it_cannot_verify():
    """Missing expert ids must not read as agreement between two blanks."""
    left = _split_arm(WINDOWS, width=1)
    right = _split_arm(WINDOWS, width=1)
    for row in right[0]:
        del row["selected_expert_ids"]
    result = compare_windows(left, right, 0, 1e-6, 1e-3)
    assert result["routing_unverifiable_samples"] > 0
    assert result["verdict"] != "BIT_IDENTICAL"
    assert any("could not be compared" in reason for reason in result["reasons"])


def test_window_mode_still_reads_a_per_sample_legacy_field():
    """Records whose ``selected_current_ids`` *is* per-sample stay comparable."""
    left = _split_arm(WINDOWS, width=1)
    right = _split_arm(WINDOWS, width=1)
    for row in right[0]:
        del row["selected_expert_ids"]
        row["selected_current_ids"] = [[16, 18]]
    result = compare_windows(left, right, 0, 1e-6, 1e-3)
    assert result["routing_unverifiable_samples"] == 0
    assert result["routing_identical"]


def test_window_mode_reports_the_window_size():
    result = compare_windows(_window_arm(WINDOWS, 4), _window_arm(WINDOWS, 1), 0, 1e-4, 1e-3)
    assert result["steps_compared"] == 2
    assert result["window_size_min"] == result["window_size_max"] == 8


def test_load_arm_reads_ranks_from_a_directory(tmp_path):
    for rank in (0, 1):
        with open(tmp_path / "train_steps.rank{}.jsonl".format(rank), "w") as handle:
            for line in _stream():
                handle.write(json.dumps(line) + "\n")
    arm = load_arm(str(tmp_path))
    assert sorted(arm) == [0, 1]
    assert len(arm[0]) == ACCUM * 2
