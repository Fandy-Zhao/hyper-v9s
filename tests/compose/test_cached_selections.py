"""CPU tests for cache-driven final-evaluation selections (0903 spec §21).

Verifies that test-time Global Top-2 selections computed from the fixed
query cache over a committed key pool equal the direct router output over
the same rows, that the resulting manifest is exactly what
``eval_task --selection-manifest`` parses, that the question-file id
sequence is enforced fail-closed, and that uncommitted (current) pools are
rejected.
"""

import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from compose.v7 import cached_selections
from compose.v7 import query_cache as qc
from compose.v7.inference import V7InferenceRouter
from compose.v7.pool import V7ExpertKeyPool
from compose.v7.routing import GlobalTop2Router

QUERY_DIM = 1536
CONTRACT_GIT = "x" * 40
CONTRACT_BRANCH = "test"
CONTRACT_BACKBONE = "y" * 64
CONTRACT_IMPL = "v7_fixed_layernorm_concat_l2_v1"


def committed_pool_state(seed=11):
    """Four historical experts (two task0, two task1) as an export_state."""
    generator = torch.Generator().manual_seed(seed)
    pool = V7ExpertKeyPool()
    for expert_id, origin in enumerate((0, 0, 1, 1)):
        key = F.normalize(torch.randn(QUERY_DIM, generator=generator), dim=0)
        pool.add(expert_id, key, origin, "current", True, rms_state={})
    pool.commit((0, 1, 2, 3), {expert_id: {} for expert_id in pool.current_ids})
    return pool.export_state()


def build_synthetic_question_cache(tmp_path, count=8, seed=7, prefix="v7_t1_test"):
    """task0/test split cache + matching declared question file (task0)."""
    records = [
        {
            "question_id": "{}_{}".format(prefix, index),
            "image": "T0/test/img_{:05d}.jpg".format(index),
            "conversations": [
                {"from": "human", "value": "<image>\nQ{}?".format(index)},
                {"from": "gpt", "value": "A"},
            ],
        }
        for index in range(count)
    ]
    declared_path = tmp_path / "test_full.json"
    declared_path.write_text(json.dumps(records), encoding="utf-8")

    split_dir = tmp_path / "query_cache" / "task0" / "test"
    split_dir.mkdir(parents=True, exist_ok=True)
    contract = qc.build_split_contract(
        git_sha=CONTRACT_GIT, git_branch=CONTRACT_BRANCH, task_index=0,
        task_name="ImageNet-A", split="test",
        source_dataset_path=str(declared_path), records=records,
        image_folder=str(tmp_path),
        backbone_name="clip-vit-large-patch14-336",
        backbone_path=str(tmp_path / "clip"), backbone_hash=CONTRACT_BACKBONE,
        query_impl_hash=CONTRACT_IMPL, world_size=2, batch_size=32,
    )
    generator = torch.Generator().manual_seed(seed)
    queries = F.normalize(torch.randn(count, QUERY_DIM, generator=generator), dim=-1)
    ids = [str(record["question_id"]) for record in records]
    qc.write_split_cache(
        str(split_dir), contract=contract, sample_ids=ids, queries=queries,
        merge_audit={}, worker_stats={}, runtime={}, created_at="2026-09-03T00:00:00",
    )
    metadata = json.loads((split_dir / "metadata.json").read_text())
    manifest = {
        "kind": "v7_fixed_query_cache_manifest",
        "schema_version": qc.QUERY_SCHEMA_VERSION,
        "query_mode": qc.QUERY_MODE,
        "query_dim": qc.QUERY_DIM,
        "dtype": qc.QUERY_DTYPE,
        "cache_root": str(tmp_path / "query_cache"),
        "git_sha": CONTRACT_GIT,
        "git_branch": CONTRACT_BRANCH,
        "tasks": {
            "ImageNet-A": {
                "test": {
                    "path": str(split_dir / "queries.pt"),
                    "metadata_path": str(split_dir / "metadata.json"),
                    "contract_hash": metadata["contract_hash"],
                    "sample_count": metadata["num_declared_samples"],
                    "sample_id_hash": metadata["sample_id_set_hash"],
                    "query_hash": metadata["query_tensor_hash"],
                    "source_dataset_sha256": metadata["contract"]["source_dataset_sha256"],
                }
            }
        },
    }
    manifest_path = tmp_path / "query_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    key_path = tmp_path / "v7_keys.pt"
    torch.save(committed_pool_state(), key_path)
    return manifest_path, declared_path, queries, ids, key_path


class TestCachedSelections:
    def test_matches_direct_router(self, tmp_path):
        manifest_path, declared_path, queries, ids, key_path = (
            build_synthetic_question_cache(tmp_path)
        )
        output = tmp_path / "selections" / "task0.json"
        audit = cached_selections.write_cached_selections(
            str(manifest_path), str(key_path), str(declared_path),
            0, str(output), device="cpu",
        )
        assert audit["encoder_calls"] == 0
        assert audit["count"] == len(ids)
        assert audit["sequence_matches_cache"]
        assert audit["question_task_index"] == 0
        # model-task index derived from the committed pool's origins (max)
        assert audit["model_task_index"] == 1
        assert audit["visible_expert_ids"] == [0, 1, 2, 3]
        # manifest selections must equal direct router rows over the cache
        state = torch.load(key_path, map_location="cpu", weights_only=False)
        pool = V7ExpertKeyPool.from_state(state)
        rows = GlobalTop2Router(pool)(queries).expert_ids
        manifest = json.loads(output.read_text(encoding="utf-8"))
        assert set(manifest) == set(ids)
        for position, sample_id in enumerate(ids):
            assert manifest[sample_id] == {
                "global_top2": [int(value) for value in rows[position]]
            }
        # every route is a committed (old-old or old-new-typed) expert pair
        for entry in manifest.values():
            assert len(entry["global_top2"]) == 2
            assert set(entry["global_top2"]) <= set(audit["visible_expert_ids"])
        assert sum(audit["selected_histogram"].values()) == len(ids)

    def test_manifest_is_eval_task_consumable(self, tmp_path):
        # eval_task --selection-manifest parses {sample_id: {"global_top2": ...}}
        # with str(record.get("id", record.get("question_id"))) keys.
        manifest_path, declared_path, queries, ids, key_path = (
            build_synthetic_question_cache(tmp_path)
        )
        output = tmp_path / "selections.json"
        cached_selections.write_cached_selections(
            str(manifest_path), str(key_path), str(declared_path),
            0, str(output), device="cpu",
        )
        manifest = json.loads(output.read_text(encoding="utf-8"))
        records = json.loads(declared_path.read_text(encoding="utf-8"))
        for record in records:
            sample_id = str(record.get("id", record.get("question_id")))
            pair = manifest[sample_id]["global_top2"]
            assert len(pair) == len(set(pair)) == 2
            assert all(int(value) == value for value in pair)

    def test_question_ids_are_the_cache_sequence(self, tmp_path):
        # evaluation question files carry question_id, matching the cache
        # sample-id binding used per record at generation time.
        manifest_path, declared_path, queries, ids, key_path = (
            build_synthetic_question_cache(tmp_path)
        )
        records = json.loads(declared_path.read_text(encoding="utf-8"))
        assert all("id" not in record for record in records)
        assert [qc.sample_id_of(record) for record in records] == ids
        manifest = qc.V7CacheManifest.locate(str(manifest_path))
        reader = manifest.reader(0, "test")
        assert list(reader.sample_ids) == ids

    def test_id_sequence_mismatch_fails_closed(self, tmp_path):
        manifest_path, declared_path, queries, ids, key_path = (
            build_synthetic_question_cache(tmp_path)
        )
        records = json.loads(declared_path.read_text(encoding="utf-8"))
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(records[::-1]), encoding="utf-8")
        with pytest.raises(ValueError, match="does not match the cache"):
            cached_selections.write_cached_selections(
                str(manifest_path), str(key_path), str(bad),
                0, str(tmp_path / "o.json"), device="cpu",
            )
        truncated = tmp_path / "short.json"
        truncated.write_text(json.dumps(records[:4]), encoding="utf-8")
        with pytest.raises(ValueError, match="does not match the cache"):
            cached_selections.write_cached_selections(
                str(manifest_path), str(key_path), str(truncated),
                0, str(tmp_path / "o.json"), device="cpu",
            )
        assert not (tmp_path / "o.json").exists()

    def test_rejects_uncommitted_pool(self, tmp_path):
        # committed-only inference: routing on a pool with current candidates
        # must fail closed (same guard as V7InferenceRouter).
        manifest_path, declared_path, queries, ids, key_path = (
            build_synthetic_question_cache(tmp_path)
        )
        state = torch.load(key_path, map_location="cpu", weights_only=False)
        for raw_id in sorted(state["metadata"], key=int):
            state["metadata"][raw_id]["lifecycle"] = "current"
        dirty = tmp_path / "current_keys.pt"
        torch.save(state, dirty)
        with pytest.raises(ValueError, match="current candidates"):
            cached_selections.write_cached_selections(
                str(manifest_path), str(dirty), str(declared_path),
                0, str(tmp_path / "o.json"), device="cpu",
            )
        assert not (tmp_path / "o.json").exists()

    def test_content_binding_fails_closed_on_backbone_drift(self, tmp_path):
        # routing on rows whose contract came from another backbone must abort
        manifest_path, declared_path, queries, ids, key_path = (
            build_synthetic_question_cache(tmp_path)
        )
        with pytest.raises(ValueError, match="query_backbone_hash"):
            cached_selections.write_cached_selections(
                str(manifest_path), str(key_path), str(declared_path),
                0, str(tmp_path / "o.json"), device="cpu",
                backbone_hash="0" * 64, impl_hash=CONTRACT_IMPL,
            )
        assert not (tmp_path / "o.json").exists()
        audit = cached_selections.write_cached_selections(
            str(manifest_path), str(key_path), str(declared_path),
            0, str(tmp_path / "ok.json"), device="cpu",
            backbone_hash=CONTRACT_BACKBONE, impl_hash=CONTRACT_IMPL,
        )
        assert audit["count"] == len(ids)

    def test_state_missing_split_fails_closed(self, tmp_path):
        # a manifest without the question task's test split is not routable
        manifest_path, declared_path, queries, ids, key_path = (
            build_synthetic_question_cache(tmp_path)
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # keep the entry valid for the constructor but drop the test split
        manifest["tasks"]["ImageNet-A"]["train"] = manifest["tasks"]["ImageNet-A"].pop("test")
        broken = tmp_path / "broken_manifest.json"
        broken.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="test"):
            cached_selections.write_cached_selections(
                str(broken), str(key_path), str(declared_path),
                0, str(tmp_path / "o.json"), device="cpu",
            )
        assert not (tmp_path / "o.json").exists()
