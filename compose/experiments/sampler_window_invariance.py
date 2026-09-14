"""Does the micro-batch / accumulation split change what an optimizer step sees?

This is the structural premise behind the S6 acceleration, and it is the reason
``compose.experiments.compare_recipe --mode window`` is a valid test when the two
arms have different micro-batch widths or world sizes.

LLaVA builds its sampler as::

    LengthGroupedSampler(self.args.train_batch_size,               # per_device x n_gpu
                         world_size=self.args.world_size * self.args.gradient_accumulation_steps)

so the *megabatch* -- the unit that is shuffled, length-sorted and then handed out
to the ranks -- has size::

    megabatch = (per_device x n_gpu) x (world x GA)

The ``n_gpu`` factor is a trap, and an earlier revision of this file fell into
it.  ``TrainingArguments.n_gpu`` is the number of devices *this process* can see,
and ``_setup_devices`` sets it to **1** whenever the run is distributed -- which
every launch here is, under ``torch.distributed.run``.  So::

    train_batch_size = per_device x 1 = per_device
    megabatch        = per_device x world x GA = effective_batch

and one optimizer step consumes exactly one megabatch.  The DataLoader's batch
size is ``per_device``, *not* ``per_device x world``: a world-2 run with
``per_device 1`` iterates batches of one sample and each rank accumulates ``GA``
of them.

This is measured, not inferred from the source.  Two real runs of this trainer
pin it:

  * the world-2 / GA-32 baseline logs **32 samples per rank per optimizer step**
    (32 micro-steps of one sample), so a step is 64 samples globally, and
    39,743 / 64 = 621 steps covers task 4 -- which is the step count the
    production task-4 run actually took;
  * the world-1 / micro-4 / GA-16 arm logs **16 micro-steps of 4**, also 64 per
    step.  Same effective batch, same window.

A third check is free and independent: the one-GPU arm takes 126.97 s per step
and the two-GPU baseline 63.76 s for the same 64 samples -- a ratio of 2.00,
which is what "the same work on half the devices" looks like.

**The invariance is over the split** -- how ``per_device x world x GA`` is
factored -- **holding the effective batch fixed.**  Every such split walks the
same samples in the same steps; what changes is only the order inside a step and
which rank receives which sample.  Those do move the floating-point summation
order, which is why an S6 comparison cannot be bit-identical -- but they do not
move the recipe.

**Change the effective batch and the megabatch changes with it**, the steps
cover different samples, and the arms are no longer comparable.  The last split
below is a control that demonstrates exactly that: it must come out *different*,
or this tool is not measuring what it claims to.  Note what the control is *not*
allowed to be: with ``n_gpu == 1``, changing only the world size leaves the
effective batch -- and so the window -- untouched, which is why row 4 is a
positive case rather than a control.

The dataset's own ``modality_lengths`` is reconstructed here exactly as
``compose.train.data.LazySupervisedDataset`` computes it (word count of the
conversations, negated when the sample has no image), so the check runs against
the same lengths the trainer feeds the sampler.

Usage::

    python -m compose.experiments.sampler_window_invariance \
        --data-path $FORMAL/task4/data/train_full.json \
        --report docs/reports/data/sampler_window_invariance.json

    # no dataset needed -- two synthetic modalities of the right shape
    python -m compose.experiments.sampler_window_invariance --synthetic
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

#: Splits compared.  Each entry is ``(label, base, other, expect_same_windows)``
#: where an arm is ``(per_device, world_size, gradient_accumulation_steps)``.
#:
#: There is deliberately no ``n_gpu`` component: under every launch this
#: repository uses it is 1, and a free parameter whose only correct value is 1
#: is a trap rather than a knob.  Every arm below therefore has
#: ``effective_batch = per_device x world x GA``.
SPLITS: Sequence[Tuple[str, Tuple[int, int, int], Tuple[int, int, int], bool]] = (
    (
        "shipped config at world 1: (micro 1, GA 64) vs (micro 4, GA 16)",
        (1, 1, 64),
        (4, 1, 16),
        True,
    ),
    (
        "shipped config at world 2: (micro 1, GA 32) vs (micro 4, GA 8)",
        (1, 2, 32),
        (4, 2, 8),
        True,
    ),
    (
        "production shape at world 4: (micro 1, GA 16) vs (micro 4, GA 4)",
        (1, 4, 16),
        (4, 4, 4),
        True,
    ),
    (
        "world size alone, effective batch held at 64: (micro 1, GA 32, "
        "world 2) vs (micro 4, GA 16, world 1)",
        (1, 2, 32),
        (4, 1, 16),
        True,
    ),
    (
        "CONTROL, different effective batch (64 vs 128) at the same world "
        "size: (micro 1, GA 32, world 2) vs (micro 4, GA 16, world 2)",
        (1, 2, 32),
        (4, 2, 16),
        False,
    ),
)

SEED = 42

#: Mirrors ``compose.train.data._KNOWN_MISSING_IMAGES``: records whose image is
#: known to be absent are dropped before the lengths are computed.
KNOWN_MISSING_IMAGES = {
    "OCR-VQA/images/1421539896.jpg",
    "OCR-VQA/images/141393394.jpg",
    "OCR-VQA/images/316881791.jpg",
    "OCR-VQA/images/140445692.jpg",
    "OCR-VQA/images/142153990X.jpg",
    "OCR-VQA/images/689852649.jpg",
}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=None, help="train_full.json")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--report", default=None)
    return parser.parse_args(argv)


def modality_lengths(data_path: str) -> List[int]:
    """The sampler input, rebuilt without instantiating the dataset."""
    with open(data_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    records = [
        record
        for record in records
        if "image" not in record or record["image"] not in KNOWN_MISSING_IMAGES
    ]
    lengths = []
    for record in records:
        length = sum(
            len(message["value"].split()) for message in record["conversations"]
        )
        lengths.append(length if "image" in record else -length)
    return lengths


def synthetic_lengths(n: int = 3000) -> List[int]:
    """Two modalities, the dataset's sign convention, without the dataset."""
    import torch

    generator = torch.Generator().manual_seed(7)
    images = (torch.rand(n // 2, generator=generator) * 400 + 100).round().long()
    text = (-(torch.rand(n - n // 2, generator=generator) * 200 + 20)).round().long()
    values = images.tolist() + text.tolist()
    return [int(value) for value in values]


def _effective_batch(arm: Tuple[int, int, int]) -> int:
    """Samples one optimizer step consumes, summed over the ranks."""
    per_device, world, accumulation = arm
    return per_device * world * accumulation


def _megabatch(arm: Tuple[int, int, int]) -> int:
    """The sampler's own unit: what gets shuffled, sorted and chopped.

    Identical to ``_effective_batch`` for every arm this repository can launch,
    because ``train_batch_size`` is ``per_device x n_gpu`` with ``n_gpu == 1``
    under a distributed run.  The two are kept separate anyway: they are equal
    by the argument above, not by definition, and the report reads better with
    the sampler's unit named rather than assumed.
    """
    per_device, world, accumulation = arm
    return per_device * world * accumulation


def _stream(lengths: Sequence[int], arm: Tuple[int, int, int]) -> List[int]:
    """The flat index list the trainer's DataLoader iterates for this arm.

    ``LengthGroupedSampler`` is constructed without a generator, so
    ``get_modality_length_grouped_indices`` draws from the global torch RNG;
    seeding here reproduces the trainer's stream given that the trainer seeds
    before it builds its dataloader.  The real run's own ``sample_ids`` remain
    the authority -- this reproduces the *sampler*, and a mismatch would show up
    as the two arms disagreeing on windows rather than as a silent pass.
    """
    import torch

    from llava.train.llava_trainer import get_modality_length_grouped_indices

    per_device, world, accumulation = arm
    torch.manual_seed(SEED)
    return get_modality_length_grouped_indices(
        list(lengths), per_device, world * accumulation
    )


def _step_windows(
    stream: Sequence[int], arm: Tuple[int, int, int]
) -> List[Tuple[int, ...]]:
    """The sample set each optimizer step consumes, globally.

    This is the real pipeline, not a model of it.  Three facts compose:

    1. ``Trainer.get_train_dataloader`` builds ``DataLoader(batch_size=
       train_batch_size)`` over the sampler and hands it to
       ``accelerator.prepare``.
    2. Torch's ``BatchSampler`` therefore chops the flat stream into
       ``train_batch_size`` = ``per_device``-sized batches, in stream order
       (``n_gpu`` is 1 per process, see the module docstring).
    3. Accelerate's ``BatchSamplerShard`` (``split_batches=False``, the default,
       which is what this trainer uses) yields, on process ``r``, exactly the
       batches whose index is ``r mod world`` -- ``_iter_with_no_split``.

    Rank ``r`` therefore runs the batches at indices ``r, r+world, r+2*world,
    ...``, and its ``GA``-th consecutive batch ends an optimizer step.  A step's
    global window is the union over the ranks.

    ``get_length_grouped_indices`` lays a megabatch out as ``world x GA``
    consecutive chunks of ``per_device`` each, so a rank's ``GA`` batches per
    megabatch land exactly on a megabatch boundary: one optimizer step consumes
    one whole megabatch, for every arm whose megabatch is that size.
    """
    per_device, world, accumulation = arm
    batch_size = per_device
    batches = [
        tuple(stream[i : i + batch_size]) for i in range(0, len(stream), batch_size)
    ]
    # Rank r owns batches r, r+world, ...; take GA of them per step.
    windows: List[Tuple[int, ...]] = []
    steps = min(
        (len(batches) - rank) // world for rank in range(world)
    ) // accumulation
    for step in range(steps):
        samples: List[int] = []
        for rank in range(world):
            rank_batches = batches[rank::world]
            start = step * accumulation
            for batch in rank_batches[start : start + accumulation]:
                samples.extend(batch)
        windows.append(tuple(sorted(samples)))
    return windows


def compare(
    lengths: Sequence[int],
    base: Tuple[int, int, int],
    other: Tuple[int, int, int],
) -> Dict[str, object]:
    """Do the two arms put the same samples into an optimizer step?

    The window is the effective batch, which is what one step consumes globally
    and therefore the unit the recipe is defined over.  ``megabatch`` is
    reported alongside it because it is the sampler's own unit -- the thing that
    is actually shuffled and chopped -- and the whole claim is that the two
    coincide.  ``megabatch_equal`` is that claim stated without reference to any
    of the reconstruction below, so the reconstruction can be checked against it
    rather than trusted.
    """
    window = _effective_batch(base)
    left, right = _stream(lengths, base), _stream(lengths, other)
    left_windows = _step_windows(left, base)
    right_windows = _step_windows(right, other)
    size = min(len(left_windows), len(right_windows))
    diverging = [
        index
        for index, (a, b) in enumerate(zip(left_windows, right_windows))
        if a != b
    ]
    return {
        "effective_batch": window,
        "megabatch_left": _megabatch(base),
        "megabatch_right": _megabatch(other),
        # The model-free statement, and the one the whole comparison rests on:
        # two arms are only comparable if the sampler chopped them out of the
        # same unit.  The reconstruction below should agree with it exactly --
        # that agreement is checked, not assumed.
        "megabatch_equal": _megabatch(base) == _megabatch(other),
        "samples": len(lengths),
        "steps_left": len(left_windows),
        "steps_right": len(right_windows),
        # Not always equal, and legitimately so: the final megabatch is short
        # (here 63 samples), so whether it completes one more accumulation
        # window depends on how wide the micro-batch is.  ``pad_to_multiple``
        # (``v7_require_full_coverage``) is what removes that tail ambiguity in
        # a real run.  Only the ``min`` of the two is compared below.
        "step_counts_agree": len(left_windows) == len(right_windows),
        "index_sequence_identical": left[: min(len(left), len(right))]
        == right[: min(len(left), len(right))],
        # Mechanically reconstructed from the sampler, torch's ``BatchSampler``
        # and accelerate's ``BatchSamplerShard`` -- see ``_step_windows``.  The
        # authoritative check is still the trainer's own ``sample_ids`` via
        # ``compare_recipe --mode window``; this reproduces the mechanism, and
        # the two are cross-checked in ``main``.
        "window_contents_identical": not diverging,
        "first_step_samples_identical": left_windows[0] == right_windows[0],
        "num_diverging_windows": len(diverging),
        "first_diverging_windows": diverging[:5],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.synthetic:
        lengths = synthetic_lengths()
        source = "synthetic (two modalities)"
    elif args.data_path:
        lengths = modality_lengths(args.data_path)
        source = args.data_path
    else:
        raise SystemExit("give --data-path or --synthetic")

    images = sum(1 for value in lengths if value > 0)
    payload: Dict[str, object] = {
        "method": "compose.experiments.sampler_window_invariance",
        "source": source,
        "seed": SEED,
        "samples": len(lengths),
        "samples_with_image": images,
        "samples_without_image": len(lengths) - images,
        "splits": {},
    }
    expectations_met = True
    for label, base, other, expect_same in SPLITS:
        result = compare(lengths, base, other)
        result["base"] = list(base)
        result["other"] = list(other)
        result["expect_same_windows"] = expect_same
        # Two checks, and the second is the one that makes the first mean
        # something.  The windows must come out as the split says they should
        # (it says so only because the algebra above does), *and* the mechanism
        # must agree with the algebra -- a reconstruction that reported
        # "everything moved" would be as useless as one that reported
        # "everything agrees".  Both directions are checked.
        result["as_expected"] = result["window_contents_identical"] is expect_same
        result["mechanism_agrees_with_algebra"] = (
            result["window_contents_identical"] is result["megabatch_equal"]
        )
        expectations_met = (
            expectations_met
            and result["as_expected"]
            and result["mechanism_agrees_with_algebra"]
        )
        payload["splits"][label] = result
        print("{}".format(label))
        print(
            "  effective batch {}; megabatch {} vs {} (equal: {}); index "
            "sequences identical: {} | steps {} vs {}; first-step samples "
            "identical {}, all {} comparable windows identical {}".format(
                result["effective_batch"],
                result["megabatch_left"],
                result["megabatch_right"],
                result["megabatch_equal"],
                result["index_sequence_identical"],
                result["steps_left"],
                result["steps_right"],
                result["first_step_samples_identical"],
                min(result["steps_left"], result["steps_right"]),
                result["window_contents_identical"],
            ),
            flush=True,
        )
    payload["invariant_splits_hold_megabatch"] = all(
        result["megabatch_equal"]
        for result in payload["splits"].values()
        if result["expect_same_windows"]
    )
    payload["mechanism_agrees_with_algebra"] = all(
        result["mechanism_agrees_with_algebra"]
        for result in payload["splits"].values()
    )
    payload["expectations_met"] = expectations_met
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(
        "VERDICT: {}".format(
            "THE WINDOW IS THE EFFECTIVE BATCH, INVARIANT TO HOW IT IS SPLIT -- "
            "micro-batch, world size and accumulation are all free as long as "
            "per_device x world x GA is held fixed, because megabatch equals "
            "effective batch and one optimizer step consumes one megabatch.  The "
            "mechanism and the megabatch algebra agree on every split, and the "
            "control -- which differs in the effective batch -- moves.  Confirm "
            "on the trainer's own sample_ids with "
            "compose.experiments.compare_recipe --mode window"
            if expectations_met
            else "NOT AS EXPECTED -- the reconstructed windows and the megabatch "
            "algebra disagree, so the comparability claim does not hold as "
            "stated and the S6/A-B design needs revisiting"
        )
    )
    return 0 if expectations_met else 1


if __name__ == "__main__":
    sys.exit(main())
