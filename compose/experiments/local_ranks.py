"""Single-GPU multi-rank launcher: the one thing ``torchrun`` cannot do here.

This machine has exactly one accelerator.  ``torchrun --nproc_per_node=2``
launches two processes and sets ``LOCAL_RANK=0`` and ``LOCAL_RANK=1``, so rank 1
asks CUDA for device 1 and dies with an invalid-ordinal error; and NCCL 2.20
refuses the obvious workaround with

    ncclInvalidUsage: Duplicate GPU detected: rank 1 and rank 0 both on CUDA
    device d8000

which is a hard check with no override (verified against this box's NCCL
2.20.5, with and without MPS active).  So a *real* two-rank run on this
hardware needs both of two things that ``torchrun`` will not do:

1. every rank pinned to device 0, which means ``LOCAL_RANK=0`` for all of them
   while ``RANK`` still differs -- the rendezvous contract is otherwise
   identical to ``torchrun``'s (``RANK`` / ``WORLD_SIZE`` / ``MASTER_ADDR`` /
   ``MASTER_PORT``, ``env://``);
2. the ``gloo`` backend, which does carry CUDA tensors between processes
   sharing one device (verified here for ``all_reduce`` and
   ``all_gather_object``).

This is a **development launcher**, not a deployment one.  It exists so the
distributed reduction, the rank-0 decision broadcast and the cross-rank
metadata-hash assertion are executed by two real ranks before formal training,
instead of being argued about.  It is deliberately not on the path a multi-GPU
host takes: there, ``--training-launcher torchrun`` (the default) uses NCCL and
one device per rank, which is what the method was written for.

Usage::

    python -m compose.experiments.local_ranks --nproc-per-node 2 -- \\
        -m compose.train.train_compose --ddp_backend gloo ...

Everything after ``--`` is the command each rank runs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Optional, Sequence


class LocalRanksError(RuntimeError):
    """Raised when the launcher cannot honour what it promises."""


def build_child_env(
    rank: int, world_size: int, master_addr: str, master_port: str, base_env
) -> dict:
    """The rendezvous environment ``torchrun`` would set, minus the rank trap.

    ``LOCAL_RANK`` is pinned to 0 for every rank: that is the whole point.  A
    child that trusts ``LOCAL_RANK`` to pick its device then lands on device 0
    by construction rather than by accident.
    """
    env = dict(base_env)
    env.update(
        {
            "RANK": str(int(rank)),
            "WORLD_SIZE": str(int(world_size)),
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": str(int(world_size)),
            "MASTER_ADDR": str(master_addr),
            "MASTER_PORT": str(master_port),
        }
    )
    return env


def launch(command: List[str], world_size: int, master_addr: str, master_port: str,
           env=None, extra_env=None) -> int:
    """Run ``command`` once per rank, all pinned to device 0.  Returns the code."""
    base = dict(os.environ if env is None else env)
    if extra_env:
        base.update({str(k): str(v) for k, v in extra_env.items()})
    children = []
    try:
        for rank in range(int(world_size)):
            children.append(
                subprocess.Popen(
                    command,
                    env=build_child_env(rank, world_size, master_addr, master_port, base),
                )
            )
        codes = [child.wait() for child in children]
    except BaseException:
        for child in children:
            child.terminate()
        for child in children:
            child.wait()
        raise
    failed = [code for code in codes if code != 0]
    if failed:
        raise LocalRanksError(
            "{} of {} ranks exited non-zero: {}".format(len(failed), len(codes), codes)
        )
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--master-addr", default="127.0.0.1")
    # Deliberately not 29500: a stale torchrun rendezvous on the default port is
    # the classic way to hang a run for an hour before anyone looks.
    parser.add_argument("--master-port", default="29611")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="the per-rank command, after a literal --",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise LocalRanksError("no per-rank command given; put it after --")
    if command[0].startswith("-"):
        # ``-- -m compose.train.train_compose`` names a module, not a binary.
        # Each rank needs its own interpreter, and it is this one.
        command = [sys.executable] + command
    if "--ddp_backend" not in command:
        raise LocalRanksError(
            "add --ddp_backend gloo: NCCL cannot place two ranks on one device, "
            "which is the only configuration this launcher supports"
        )
    print(
        "[local-ranks] {} ranks, all pinned to device 0, command: {}".format(
            args.nproc_per_node, " ".join(command)
        ),
        flush=True,
    )
    code = launch(
        command,
        args.nproc_per_node,
        args.master_addr,
        args.master_port,
        env=os.environ,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
