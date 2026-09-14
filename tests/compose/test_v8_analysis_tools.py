"""The two report-generating tools must be right about the one thing each can
silently get wrong.

Both tools exist so that no number in `V8_SPEED_BASELINE.md` or
`V8_SPEEDUP_REPORT.md` is hand-added, which is worth exactly as much as the
tools are trustworthy.  Each has a single judgement call that a plausible
implementation would get wrong without failing loudly:

* `stage_budget` must not trust a stage marker that a repair pass re-stamped.
  A pipeline that is resumed re-stamps markers it had already written, and those
  stamps are *later* than the work they claim to bound, so trusting them
  silently shortens a stage by hours.  The real run needed this twice, so the
  fallback is pinned here with both a trusted and a re-stamped marker.
* `baseline_summary` must report `peak_allocated_bytes` as a maximum, not a
  mean.  A mean of per-step peaks is not a peak and understates headroom.
"""

import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from compose.experiments import baseline_summary, stage_budget  # noqa: E402

HOUR = 3600.0


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _fake_task(root, index, *, train_runtime, s3_offset, rms_offset, prune_end,
               train_log_mtime, re_stamped=False):
    """Build one task's artifacts and return the mtimes actually stamped.

    ``re_stamped`` writes the s3/s4 markers at the *end* of the run, which is
    what a repair pass does: the file exists but its mtime no longer bounds the
    stage it names.
    """
    d = root / f"task{index}"
    _write(d / "logs" / "training.log",
           f"{{'train_runtime': {train_runtime}, 'train_samples_per_second': 1.8}}\n")
    _write(d / "logs" / "rms.log", "rms\n")
    os.utime(d / "logs" / "training.log", (train_log_mtime, train_log_mtime))
    rms_log_mtime = train_log_mtime + rms_offset
    os.utime(d / "logs" / "rms.log", (rms_log_mtime, rms_log_mtime))

    for name in ("s0_full_data", "s1_fixed_queries", "s2_candidates"):
        _write(d / "stages" / f"{name}.done", "{}\n")
        os.utime(d / "stages" / f"{name}.done", (train_log_mtime, train_log_mtime))

    if re_stamped:
        s3 = prune_end
        s4 = prune_end
    else:
        s3 = train_log_mtime + s3_offset
        s4 = train_log_mtime + rms_offset
    _write(d / "stages" / "s3_training.done", "{}\n")
    _write(d / "stages" / "s4_rms.done", "{}\n")
    os.utime(d / "stages" / "s3_training.done", (s3, s3))
    os.utime(d / "stages" / "s4_rms.done", (s4, s4))

    _write(d / "stages" / "s5_pruning_commit.done", "{}\n")
    os.utime(d / "stages" / "s5_pruning_commit.done", (prune_end, prune_end))
    return {"s3": s3, "s4": s4, "rms_log": rms_log_mtime}


def test_stage_budget_trusts_a_marker_that_matches_its_log(tmp_path):
    """The normal case: s3 and s4 are within seconds of their stage logs."""
    t0 = 1_700_000_000.0
    _fake_task(tmp_path, 0, train_runtime=4 * HOUR, s3_offset=2.0,
               rms_offset=1800.0, prune_end=t0 + 1800.0 + 600.0,
               train_log_mtime=t0)
    result = stage_budget.stage_budget(tmp_path)
    row = result["tasks"][0]

    assert row["sources"]["rms_start"] == "s3_training.done"
    assert row["sources"]["rms_end"] == "s4_rms.done"
    # Measured from s3, which the marker places 2 s after the training log stopped
    # -- the two clocks are not interchangeable, and this is the gap between the
    # audit's marker-based stage budget and a log-to-log reading.
    assert row["rms_h"] == pytest.approx((1800.0 - 2.0) / HOUR)
    assert row["prune_h"] == pytest.approx(600.0 / HOUR)
    # Training comes from the trainer's own clock, not from a marker difference.
    assert row["train_h"] == pytest.approx(4.0)


def test_stage_budget_refuses_a_re_stamped_marker(tmp_path):
    """A repair pass wrote s3/s4 long after the work: both must be ignored.

    This is task 0 and task 5 of the real run.  If the markers were trusted, the
    RMS stage would be measured as a negative or absurd interval; with the
    fallback it is the training-log-to-rms-log span.
    """
    t0 = 1_700_000_000.0
    rms_offset = 3 * HOUR
    # Markers stamped 9 h after training started, i.e. well past the RMS stage.
    _fake_task(tmp_path, 0, train_runtime=4 * HOUR, s3_offset=2.0,
               rms_offset=rms_offset, prune_end=t0 + 9 * HOUR,
               train_log_mtime=t0, re_stamped=True)
    result = stage_budget.stage_budget(tmp_path)
    row = result["tasks"][0]

    assert "re-stamped" in row["sources"]["rms_start"]
    assert "re-stamped" in row["sources"]["rms_end"]
    # Both boundaries fall back to the stage logs, so RMS is log-to-log.
    assert row["rms_h"] == pytest.approx(rms_offset / HOUR)
    assert row["prune_h"] == pytest.approx((9 * HOUR - rms_offset) / HOUR)


def _step_row(step, *, wall=64.0, allreduce=0.1, peak_alloc=16 * 2**30,
              peak_resv=17 * 2**30, samples=32, tokens=20700, world=2):
    return {
        "step": step, "window_wall_time": wall, "step_body_time": 56.0,
        "allreduce_time": allreduce, "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_resv, "samples": samples, "tokens": tokens,
        "world_size": world, "mean_seq_len": 648.0, "padding_ratio": 0.0,
        "model_forwards": 32, "lora_expert_evaluations": 28672,
        "compose_layers_forward": 14336, "micro_steps": 32,
    }


def test_baseline_summary_peaks_are_maxima_and_warmup_is_dropped(tmp_path):
    """Three warm-up steps carry a bogus 30 GiB peak; it must not survive.

    A mean-of-peaks implementation would report ~16.x GiB here and hide the
    difference, so the assertion is on the maximum specifically.
    """
    path = tmp_path / "profile_steps.jsonl"
    rows = [_step_row(s, wall=99.0, peak_alloc=30 * 2**30) for s in (1, 2, 3)]
    rows += [_step_row(s, peak_alloc=16 * 2**30) for s in (4, 5, 6, 7)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    kept = baseline_summary.load_trace(path, skip_steps=3)
    assert [r["step"] for r in kept] == [4, 5, 6, 7]

    stats = baseline_summary.trace_stats(kept)
    assert stats["peak_allocated_gib_max"] == pytest.approx(16.0, abs=0.01)
    # 64 GiB of 32 samples at world 2 over a 64 s step.
    assert stats["samples_per_sec_global"] == pytest.approx(1.0)
    assert stats["steps_per_sec"] == pytest.approx(1 / 64.0)


def test_baseline_summary_reports_median_and_max_for_skewed_phases(tmp_path):
    """`allreduce_time` is skewed; the mean alone misreports it.

    One 12 s stall in twenty windows leaves the mean at ~0.7 s while the typical
    window is 0.06 s -- the difference between a bandwidth problem and a
    straggler, which is the whole of baseline report section 4.1.
    """
    path = tmp_path / "profile_steps.jsonl"
    rows = [_step_row(s, allreduce=0.06) for s in range(5, 24)]
    rows.append(_step_row(24, allreduce=12.0))
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    stats = baseline_summary.trace_stats(baseline_summary.load_trace(path, 4))
    assert stats["allreduce_time_median"] == pytest.approx(0.06)
    assert stats["allreduce_time_max"] == pytest.approx(12.0)
    assert stats["allreduce_time"] > 0.5  # the mean is pulled off by the stall


def test_baseline_summary_cross_rank_anti_correlation(tmp_path):
    """Two anti-correlated ranks must produce a negative correlation."""
    spans = {"r0": [0.06, 0.06, 5.0, 0.06, 6.0], "r1": [6.0, 5.0, 0.06, 6.0, 0.06]}
    pairs = {
        label: [{"step": i + 5, "allreduce_time": v, "window_wall_time": 60.0,
                 "step_body_time": 50.0} for i, v in enumerate(vals)]
        for label, vals in spans.items()
    }
    out = baseline_summary.cross_rank(pairs)
    assert out["corr_allreduce_time_r0_vs_r1"] == pytest.approx(-1.0, abs=0.2)


def test_baseline_summary_reads_the_startup_sidecar(tmp_path):
    """One-shot phases live in the sidecar, not in the step rows."""
    path = tmp_path / "profile_steps.jsonl"
    path.write_text(json.dumps(_step_row(4)) + "\n" + json.dumps(_step_row(5)) + "\n")
    (tmp_path / "profile_steps.jsonl.startup.json").write_text(
        json.dumps({"startup": {"query_load": 12.5, "model_build": 40.0},
                    "startup_total_sec": 52.5})
    )
    side = json.loads((tmp_path / "profile_steps.jsonl.startup.json").read_text())
    assert side["startup"]["query_load"] == 12.5
