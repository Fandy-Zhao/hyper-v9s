"""The metrics writer and the metrics reader must name the same files.

``final_diagnostics`` re-reads everything the step loop wrote, so the two sides
have to agree on the path convention exactly.  They did not: the writer kept the
un-suffixed name at ``world_size == 1`` while the reader keyed off
``torch.distributed.is_initialized()``, which ``torchrun --nproc_per_node 1``
also makes true.  A single-process run therefore trained to completion, raised
``FileNotFoundError`` on the last thing it does, and reported failure -- with an
empty ``v7_training_diagnostics.json`` to show for it.
"""

import os
import sys
from pathlib import Path

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from compose.v7.hf_trainer import ranked_metrics_paths  # noqa: E402


BASE = Path("/runs/arm/training/train_steps.jsonl")


def test_single_process_keeps_the_un_suffixed_name():
    assert ranked_metrics_paths(BASE, 1) == [BASE]


def test_multi_process_is_per_rank():
    assert ranked_metrics_paths(BASE, 2) == [
        Path("/runs/arm/training/train_steps.rank0.jsonl"),
        Path("/runs/arm/training/train_steps.rank1.jsonl"),
    ]


def test_the_written_path_is_always_one_the_reader_checks():
    """Round-trip: every rank writes a file that final_diagnostics will open."""
    for world_size in (1, 2, 4):
        paths = ranked_metrics_paths(BASE, world_size)
        assert len(paths) == max(1, world_size)
        for rank in range(world_size):
            assert paths[rank] in paths


def test_a_non_positive_world_size_is_treated_as_single_process():
    # Defensive: the value can only be 1 in practice, but the reader must never
    # build an empty path list and silently report zero rows.
    assert ranked_metrics_paths(BASE, 0) == [BASE]
