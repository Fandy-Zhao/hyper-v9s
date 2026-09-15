"""CPU coverage for direct V7 query-cache consumption by V9-S."""

import json

import pytest
import torch
from torch.nn import functional as F

from compose.v7 import query_cache as qc
from compose.v9.data import resolve_split_query_source


def _manifest(tmp_path):
    records = [{"id": "sample-{}".format(index), "image": "x.jpg",
                "conversations": [{"from": "human", "value": "Q"}]}
               for index in range(3)]
    declared = tmp_path / "train.json"
    declared.write_text(json.dumps(records), encoding="utf-8")
    split = tmp_path / "cache" / "task0" / "train"
    split.mkdir(parents=True)
    clip = tmp_path / "clip"
    clip.mkdir()
    contract = qc.build_split_contract(
        git_sha="x" * 40, git_branch="test", task_index=0, task_name="task",
        split="train", source_dataset_path=str(declared), records=records,
        image_folder=str(tmp_path), backbone_name="clip", backbone_path=str(clip),
        backbone_hash="y" * 64, query_impl_hash="impl", world_size=1, batch_size=1,
    )
    ids = [record["id"] for record in records]
    queries = F.normalize(torch.randn(3, qc.QUERY_DIM), dim=-1)
    qc.write_split_cache(str(split), contract=contract, sample_ids=ids, queries=queries,
                         merge_audit={}, worker_stats={}, runtime={}, created_at="test")
    metadata = json.loads((split / "metadata.json").read_text(encoding="utf-8"))
    payload = {"kind": "v7_fixed_query_cache_manifest", "schema_version": 1,
               "query_mode": "v7_fixed", "query_dim": qc.QUERY_DIM,
               "dtype": "float32", "cache_root": str(tmp_path / "cache"),
               "git_sha": "x" * 40, "git_branch": "test", "tasks": {"task": {"train": {
                   "path": str(split / "queries.pt"), "metadata_path": str(split / "metadata.json"),
                   "contract_hash": metadata["contract_hash"], "sample_count": len(ids),
                   "sample_id_hash": metadata["sample_id_set_hash"],
                   "query_hash": metadata["query_tensor_hash"],
               }}}}
    manifest = tmp_path / "query_cache_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, ids, queries, split


def test_v9_resolves_manifest_tensor_in_declared_row_order(tmp_path):
    manifest, ids, expected_queries, split = _manifest(tmp_path)
    source = resolve_split_query_source(str(manifest), task_index=0, split="train", expected_ids=ids)
    assert source.sample_ids == tuple(ids)
    assert torch.equal(source.queries, expected_queries)
    assert source.tensor_path == str((split / "queries.pt").resolve())
    assert source.contract_record()["query_value_hash"]


def test_v9_rejects_manifest_tensor_when_declared_split_is_not_covered(tmp_path):
    manifest, ids, _queries, _split = _manifest(tmp_path)
    with pytest.raises(ValueError, match="misses"):
        resolve_split_query_source(str(manifest), task_index=0, split="train", expected_ids=ids + ["absent"])
