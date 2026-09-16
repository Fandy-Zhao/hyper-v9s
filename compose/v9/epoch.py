"""Deterministic global-batch epoch padding.

The formal contract is that a task's whole declared split is consumed by
optimizer steps whose effective batch is exactly ``GLOBAL_BATCH``.  Sampling
alone does not give that, for two independent reasons that are easy to miss
because each looks reasonable on its own:

* ``len(train_dataloader)`` per rank is ``ceil(ceil(N / micro) / world)`` once
  accelerate shards the batches.  When that count is odd, the last micro-batch
  opens a gradient window that nothing closes.
* transformers 4.33.3 then derives
  ``num_update_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps``
  and its escape hatch for a trailing partial window,
  ``is_last_step_and_steps_less_than_grad_acc``, only fires when the *entire*
  epoch is shorter than one accumulation window.  For a real task it is not, so
  the trailing micro-batch is forwarded, backwarded, and then dropped: its
  gradients never reach an ``optimizer.step()``.

The result is a run that reports full coverage -- the forward-side audit counts
those samples, because they really were forwarded -- while a handful of them
never contributed a gradient.  That is a different guarantee from the one the
formal pipeline claims, so it is fixed here rather than reported.

The fix pads the epoch to a whole number of global batches.  Because

    padded = ceil(N / (world * micro * accum)) * (world * micro * accum)

is divisible by ``world * micro``, every rank gets the same number of
micro-batches and that number is divisible by ``accum`` -- so every window
closes, the floor division above becomes exact, and the number of optimizer
steps is ``ceil(N / GLOBAL_BATCH)`` by construction rather than by luck.

Order is precomputed in ``__init__``, not in ``__iter__``.  A sampler that
draws from the global RNG while iterating depends on how much RNG every other
part of startup consumed, which would make "same seed, same epoch, same world
size" reproducible in intention only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import Sampler

__all__ = [
    "EpochPlan",
    "GlobalBatchPaddedSampler",
    "global_batch",
    "plan_epoch",
    "verify_epoch_plan",
]


class V9EpochError(RuntimeError):
    pass


def global_batch(world_size: int, per_device_batch: int, grad_accum: int) -> int:
    """The effective batch one optimizer step must average over."""
    world = int(world_size)
    micro = int(per_device_batch)
    accum = int(grad_accum)
    if world < 1 or micro < 1 or accum < 1:
        raise V9EpochError(
            "world_size, per_device_batch and grad_accum must all be >= 1, got "
            "{} {} {}".format(world, micro, accum)
        )
    return world * micro * accum


@dataclass(frozen=True)
class EpochPlan:
    """What an epoch is supposed to look like, before any of it is executed."""

    declared_unique_samples: int
    world_size: int
    per_device_batch: int
    grad_accum: int
    padded_epoch_samples: int
    padding_duplicate_count: int
    per_rank_micro_batches: int
    expected_optimizer_steps: int

    @property
    def global_batch(self) -> int:
        return global_batch(self.world_size, self.per_device_batch, self.grad_accum)

    def as_dict(self) -> Dict[str, int]:
        return {
            "declared_unique_samples": self.declared_unique_samples,
            "padded_epoch_samples": self.padded_epoch_samples,
            "padding_duplicate_count": self.padding_duplicate_count,
            "world_size": self.world_size,
            "per_device_batch": self.per_device_batch,
            "grad_accum": self.grad_accum,
            "global_batch": self.global_batch,
            "per_rank_micro_batches": self.per_rank_micro_batches,
            "expected_optimizer_steps": self.expected_optimizer_steps,
        }


def plan_epoch(
    declared_samples: int,
    world_size: int,
    per_device_batch: int,
    grad_accum: int,
) -> EpochPlan:
    """The padded epoch a task of ``declared_samples`` samples must run.

    Pure arithmetic, no I/O: the trainer asserts against it at startup and the
    task-completion predicate recomputes it from the recorded split size, so a
    disagreement between the two is a disagreement about the contract rather
    than about a number one side copied from the other.
    """
    declared = int(declared_samples)
    if declared < 1:
        raise V9EpochError("declared_samples must be >= 1, got {}".format(declared))
    unit = global_batch(world_size, per_device_batch, grad_accum)
    padded = int(math.ceil(declared / unit)) * unit
    per_rank_micro = padded // (int(world_size) * int(per_device_batch))
    return EpochPlan(
        declared_unique_samples=declared,
        world_size=int(world_size),
        per_device_batch=int(per_device_batch),
        grad_accum=int(grad_accum),
        padded_epoch_samples=padded,
        padding_duplicate_count=padded - declared,
        per_rank_micro_batches=per_rank_micro,
        expected_optimizer_steps=padded // unit,
    )


class GlobalBatchPaddedSampler(Sampler):
    """A deterministic epoch order padded to a whole number of global batches.

    ``__len__`` is the padded length, so the batch sampler above it yields
    exactly ``padded / per_device_batch`` batches on each rank with every batch
    full -- no partial trailing batch, and therefore no window that cannot be
    closed.

    The padding repeats the *beginning* of this epoch's own order.  Any
    deterministic rule would do; taking it from the same permutation the epoch
    is already following keeps the duplicates inside the epoch's own
    distribution instead of introducing a position-dependent rule that has to
    be reasoned about separately.
    """

    def __init__(
        self,
        dataset_length: int,
        world_size: int,
        per_device_batch: int,
        grad_accum: int,
        seed: int = 42,
        epoch: int = 0,
    ) -> None:
        length = int(dataset_length)
        if length < 1:
            raise V9EpochError("dataset_length must be >= 1, got {}".format(length))
        self.dataset_length = length
        self.plan = plan_epoch(length, world_size, per_device_batch, grad_accum)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self._order = self._build_order(self.epoch)

    def _build_order(self, epoch: int) -> List[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(epoch))
        order = torch.randperm(self.dataset_length, generator=generator).tolist()
        extra = self.plan.padding_duplicate_count
        if extra:
            order = order + order[:extra]
        return order

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle deterministically for ``epoch``, padding included.

        Present so a multi-epoch run keeps the same guarantee per epoch rather
        than only for the first: the padding is a function of the order, so it
        has to be rebuilt whenever the order is.
        """
        self.epoch = int(epoch)
        self._order = self._build_order(self.epoch)

    def __len__(self) -> int:
        return int(self.plan.padded_epoch_samples)

    def __iter__(self) -> Iterator[int]:
        return iter(self._order)


def verify_epoch_plan(
    plan: EpochPlan,
    per_rank_dataloader_length: Optional[int] = None,
    observed_optimizer_steps: Optional[int] = None,
) -> Dict[str, int]:
    """Check a plan against what actually ran, and return the numbers.

    Called at two points with two different halves filled in: the trainer
    checks the dataloader it built before the first step, and the completion
    predicate checks the optimizer steps afterwards.  Failing here is a task
    failure, not a warning -- a run whose step count is short has already
    supervised a different fraction of its data than it reports.
    """
    if plan.expected_optimizer_steps < 1:
        raise V9EpochError("plan expects no optimizer steps at all: {}".format(plan))
    expected_micro = plan.padded_epoch_samples // (
        plan.world_size * plan.per_device_batch
    )
    if plan.per_rank_micro_batches != expected_micro:
        raise V9EpochError(
            "plan is internally inconsistent: {} micro-batches per rank for "
            "{} padded samples".format(plan.per_rank_micro_batches, plan.padded_epoch_samples)
        )
    if plan.per_rank_micro_batches % plan.grad_accum:
        raise V9EpochError(
            "padded epoch still leaves {} unclosed micro-batch(es) per rank "
            "({} micro-batches, accumulation {}); padding did not take".format(
                plan.per_rank_micro_batches % plan.grad_accum,
                plan.per_rank_micro_batches,
                plan.grad_accum,
            )
        )
    if per_rank_dataloader_length is not None and int(per_rank_dataloader_length) != (
        plan.per_rank_micro_batches
    ):
        raise V9EpochError(
            "the dataloader yields {} micro-batches per rank but the contract "
            "requires {}; the sampler is not the padded one".format(
                per_rank_dataloader_length, plan.per_rank_micro_batches
            )
        )
    if observed_optimizer_steps is not None and int(observed_optimizer_steps) != (
        plan.expected_optimizer_steps
    ):
        raise V9EpochError(
            "the task took {} optimizer steps but covering {} declared samples "
            "in global batches of {} requires {}".format(
                observed_optimizer_steps,
                plan.declared_unique_samples,
                plan.global_batch,
                plan.expected_optimizer_steps,
            )
        )
    return plan.as_dict()
