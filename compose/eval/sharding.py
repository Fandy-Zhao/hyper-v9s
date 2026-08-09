"""4-GPU sample-sharding helpers (spec §13/§15/§17/§18).

The 4-GPU execution strategy shards the *samples* of a stage across four
worker processes (one physical GPU each) and merges the partial outputs on
the rank-0 orchestrator. Every helper here is order-preserving and
deterministic, so the merged artifact is byte-identical to the single-GPU
artifact whenever per-sample computation is deterministic:

- ``shard_records``/``shard_indices`` partition a record list into
  contiguous, disjoint, covering shards;
- ``merge_partial_maps`` merges ``{sample_id: value}`` partial JSON outputs
  and verifies the union matches the expected id set exactly (0 missing,
  0 duplicates);
- ``merge_partial_answer_files`` concatenates per-chunk answer lines back
  into the canonical answers file in original record order.

Sharding never touches the per-sample candidate space (teacher search
candidates are fixed by the selections file, which is shared verbatim).
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence


def shard_indices(total: int, num_shards: int, shard_index: int):
    """Contiguous [start, stop) slice of ``total`` for one shard.

    Uses ``ceil(total / num_shards)`` per shard so every shard is a prefix
    of the same size (standard eval-task chunking); the last shard may be
    shorter. All shards are disjoint and cover the whole list.
    """
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError(
            "invalid shard selection {}/{}".format(shard_index, num_shards)
        )
    size = int(math.ceil(total / num_shards))
    start = shard_index * size
    stop = min(start + size, total)
    return start, stop


def shard_records(records: Sequence[Any], num_shards: int, shard_index: int) -> List[Any]:
    start, stop = shard_indices(len(records), num_shards, shard_index)
    return list(records[start:stop])


def record_ids(records: Sequence[Dict[str, Any]]) -> List[str]:
    """Id-first sample ids, mirroring the pipeline's keying convention."""
    return [
        str(record.get("id", record.get("question_id"))) for record in records
    ]


def merge_partial_maps(
    partial_paths: Sequence[str],
    expected_ids: Sequence[str],
    value_transform=lambda value: value,
) -> Dict[str, Any]:
    """Merge per-shard ``{sample_id: value}`` JSON outputs into one map.

    Verifies the merged id set equals ``expected_ids`` exactly (every
    expected id present once; no foreign or duplicate ids) and returns the
    merged map keyed in ``expected_ids`` order.
    """
    merged: Dict[str, Any] = {}
    for path in partial_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for sample_id, value in payload.items():
            sample_id = str(sample_id)
            if sample_id in merged:
                raise ValueError(
                    "duplicate sample {} across shards (merge of {})".format(
                        sample_id, path
                    )
                )
            merged[sample_id] = value_transform(value)
    expected = [str(sample_id) for sample_id in expected_ids]
    if sorted(merged) != sorted(expected):
        missing = sorted(set(expected) - set(merged))
        foreign = sorted(set(merged) - set(expected))
        raise ValueError(
            "shard merge mismatch: {} missing, {} foreign ({} shards)".format(
                len(missing), len(foreign), len(partial_paths)
            )
        )
    return {sample_id: merged[sample_id] for sample_id in expected}


def partial_path(output: str, shard_index: int) -> str:
    """Canonical per-shard output path: ``<output>.rank<index>``."""
    return "{}.rank{}".format(output, shard_index)


def merge_partial_answer_files(
    answer_paths: Sequence[str], expected_lines: int
) -> str:
    """Concatenate per-chunk answers.jsonl in chunk order.

    Chunks are contiguous record slices, so concatenation restores the
    original record order exactly. Verifies the total line count.
    """
    merged = []
    for path in answer_paths:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        if not lines:
            continue
        merged.extend(lines)
    if len(merged) != expected_lines:
        raise ValueError(
            "answer merge mismatch: expected {} lines, got {} ({})".format(
                expected_lines, len(merged), len(answer_paths)
            )
        )
    return "\n".join(merged) + "\n"
