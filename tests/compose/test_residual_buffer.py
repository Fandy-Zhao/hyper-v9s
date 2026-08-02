from dataclasses import replace

from compose.cli.build_residual_buffer import resume_or_write
from compose.expansion.residual_buffer import ResidualBuffer, ResidualRecord


def make(sample="x", task=1, split="train"):
    return ResidualRecord(sample, task, "ArxivQA", split, "data:1", "dmh", "image:1", "prompt:1", "answer:1", "qhash",
                          (0,), (0,), True, 0.7, 1.0, 0.9, 0.1, 0.8, "2026-08-03T00:00:00Z", "commit", "config")


def test_buffer_atomic_shard_round_trip(tmp_path):
    buffer = ResidualBuffer("teacher_buffer")
    buffer.add(make())
    path = tmp_path / "rank0.json"
    buffer.write_shard(path, 0)
    restored, rank = ResidualBuffer.load_shard(path)
    assert rank == 0 and restored.records == buffer.records


def test_ddp_merge_requires_exact_expected_ids(tmp_path):
    paths = []
    for rank, sample in enumerate(("a", "b")):
        shard = ResidualBuffer("teacher_buffer"); shard.add(make(sample)); path = tmp_path / f"r{rank}.json"; shard.write_shard(path, rank); paths.append(path)
    merged = ResidualBuffer.merge_shards(paths, tmp_path / "merged.json", ["a", "b"], "teacher_buffer")
    assert [row.sample_id for row in merged.records] == ["a", "b"]


def test_buffer_resume_preserves_matching_shard_and_rejects_mismatch(tmp_path):
    path = tmp_path / "buffer.json"
    original = ResidualBuffer("teacher_buffer"); original.add(make())
    _, resumed = resume_or_write(original, path)
    assert not resumed
    reconstructed = ResidualBuffer("teacher_buffer")
    reconstructed.add(replace(make(), creation_timestamp="later", source_git_commit="later-commit"))
    restored, resumed = resume_or_write(reconstructed, path)
    assert resumed and restored.records == original.records
    conflicting = ResidualBuffer("teacher_buffer"); conflicting.add(replace(make(), residual_gain=99.0))
    import pytest
    with pytest.raises(ValueError, match="does not match"):
        resume_or_write(conflicting, path)
