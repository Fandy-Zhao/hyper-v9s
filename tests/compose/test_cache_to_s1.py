"""CPU tests for the S1 cache adapter (0903 spec §5-8).

Verifies that a payload emitted from the binary fixed-query cache is
schema-compatible with the legacy live-CLIP S1 output: every downstream
consumer (validate_query_cache_contract / queries_from_cache / the
records map V7QueryDataset builds) accepts it, the per-sample query
values equal the cache rows, the id sequence is enforced exactly
(fail-closed on any mismatch), and query_hash keeps the legacy semantics.
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from compose.v7 import cache_to_s1, query_cache as qc
from compose.v7.workflow import queries_from_cache, validate_query_cache_contract

QUERY_DIM = 1536
CONTRACT_GIT = "x" * 40
CONTRACT_BRANCH = "test"
CONTRACT_BACKBONE = "y" * 64
CONTRACT_IMPL = "v7_fixed_layernorm_concat_l2_v1"


def build_synthetic_task(tmp_path, count=8, seed=7):
    """Cache root with one split (task0/train) + matching manifest."""
    records = [
        {
            "id": "v7_t0_train_{}".format(index),
            "image": "T0/train/img_{:05d}.jpg".format(index),
            "conversations": [
                {"from": "human", "value": "<image>\nQ{}?".format(index)},
                {"from": "gpt", "value": "A"},
            ],
        }
        for index in range(count)
    ]
    declared_path = tmp_path / "train_full.json"
    declared_path.write_text(json.dumps(records), encoding="utf-8")

    split_dir = tmp_path / "query_cache" / "task0" / "train"
    split_dir.mkdir(parents=True, exist_ok=True)
    contract = qc.build_split_contract(
        git_sha=CONTRACT_GIT, git_branch=CONTRACT_BRANCH, task_index=0,
        task_name="ImageNet-R", split="train",
        source_dataset_path=str(declared_path), records=records,
        image_folder=str(tmp_path),
        backbone_name="clip-vit-large-patch14-336",
        backbone_path=str(tmp_path / "clip"), backbone_hash=CONTRACT_BACKBONE,
        query_impl_hash=CONTRACT_IMPL, world_size=2, batch_size=32,
    )
    generator = torch.Generator().manual_seed(seed)
    queries = F.normalize(torch.randn(count, QUERY_DIM, generator=generator), dim=-1)
    ids = [str(record["id"]) for record in records]
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
            "ImageNet-R": {
                "train": {
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
    clip_dir = tmp_path / "clip"
    clip_dir.mkdir(exist_ok=True)
    provenance = {
        "resolved_path": str(clip_dir.resolve()),
        "configuration_files": {"config.json": "0" * 64},
        "backbone_hash": CONTRACT_BACKBONE,
    }
    return manifest_path, declared_path, queries, ids, provenance, split_dir


class TestCachedPayloadCompatibility:
    def test_payload_schema_and_values(self, tmp_path):
        manifest_path, declared_path, queries, ids, provenance, split_dir = (
            build_synthetic_task(tmp_path)
        )
        manifest = qc.V7CacheManifest(str(manifest_path))
        output = tmp_path / "features" / "train.json"
        audit = cache_to_s1.write_split_payload_from_cache(
            manifest, 0, "train", str(declared_path), str(output), provenance,
        )
        assert audit["declared_count"] == audit["cached_count"] == len(ids)
        assert audit["encoder_calls"] == 0
        payload = json.loads(output.read_text(encoding="utf-8"))
        # envelope byte-compatible with the legacy S1 payload
        assert payload["schema_version"] == 1
        assert payload["feature_source"] == "frozen_clip_l14_336"
        assert payload["query_mode"] == "v7_fixed"
        assert payload["query_encoder_hash"] == CONTRACT_IMPL
        assert payload["query_encoder_provenance"]["module_hash"] == CONTRACT_IMPL
        assert payload["query_origin"]["kind"] == cache_to_s1.PAYLOAD_ORIGIN_KIND
        assert payload["query_origin"]["manifest_sha256"] == audit["manifest_sha256"]
        # records: file order preserved, values bit-equal to cache rows
        assert list(payload["records"]) == ids
        for position, sample_id in enumerate(ids):
            row = torch.tensor(payload["records"][sample_id]["query"])
            assert torch.equal(row, queries[position])

    def test_downstream_consumers_accept_payload(self, tmp_path):
        manifest_path, declared_path, queries, ids, provenance, split_dir = (
            build_synthetic_task(tmp_path)
        )
        manifest = qc.V7CacheManifest(str(manifest_path))
        output = tmp_path / "features" / "train.json"
        cache_to_s1.write_split_payload_from_cache(
            manifest, 0, "train", str(declared_path), str(output), provenance,
        )
        # the orchestrator's post-S1 validation (workflow.py)
        backbone = validate_query_cache_contract(
            (str(output),), expected_backbone="clip-vit-large-patch14-336",
            expected_path=str(Path(provenance["resolved_path"])),
        )
        assert backbone["backbone_hash"] == CONTRACT_BACKBONE
        # queries_from_cache reads sorted-id rows: must equal cache rows
        rows, sorted_ids = queries_from_cache(str(output), len(ids))
        assert sorted_ids == tuple(sorted(ids))
        reader = qc.V7QuerySplitReader(str(split_dir))
        assert torch.equal(rows, reader.get_batch(sorted_ids))
        # the per-id map V7QueryDataset builds has every train id
        payload = json.loads(output.read_text(encoding="utf-8"))
        fixed_queries = {
            str(sample_id): torch.tensor(value["query"], dtype=torch.float32)
            for sample_id, value in payload["records"].items()
        }
        assert all(sample_id in fixed_queries for sample_id in ids)

    def test_query_hash_legacy_semantics(self, tmp_path):
        manifest_path, declared_path, queries, ids, provenance, split_dir = (
            build_synthetic_task(tmp_path)
        )
        manifest = qc.V7CacheManifest(str(manifest_path))
        output = tmp_path / "features" / "train.json"
        cache_to_s1.write_split_payload_from_cache(
            manifest, 0, "train", str(declared_path), str(output), provenance,
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
        # legacy records_query_hash: sha256 of {sid: query_list} sorted by id
        legacy = {
            str(sample_id): payload["records"][sample_id]["query"]
            for sample_id in sorted(payload["records"])
        }
        expected = hashlib.sha256(
            json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        assert payload["query_hash"] == expected

    def test_id_sequence_mismatch_fails_closed(self, tmp_path):
        manifest_path, declared_path, queries, ids, provenance, split_dir = (
            build_synthetic_task(tmp_path)
        )
        manifest = qc.V7CacheManifest(str(manifest_path))
        records = json.loads(declared_path.read_text(encoding="utf-8"))
        reordered = [records[0], *records[2:], records[1]]  # move last id forward
        bad = tmp_path / "reordered.json"
        bad.write_text(json.dumps(reordered), encoding="utf-8")
        with pytest.raises(ValueError, match="does not match the cache"):
            cache_to_s1.write_split_payload_from_cache(
                manifest, 0, "train", str(bad), str(tmp_path / "o.json"), provenance,
            )
        extra = [dict(value) for value in records]
        extra[0]["id"] = "v7_t0_train_999"
        bad = tmp_path / "extra.json"
        bad.write_text(json.dumps(extra), encoding="utf-8")
        with pytest.raises(ValueError, match="does not match the cache"):
            cache_to_s1.write_split_payload_from_cache(
                manifest, 0, "train", str(bad), str(tmp_path / "o.json"), provenance,
            )
        assert not (tmp_path / "o.json").exists()  # nothing written on failure


class TestEmitS1Entry:
    def test_emit_s1_audit(self, tmp_path, monkeypatch):
        manifest_path, declared_path, queries, ids, provenance, split_dir = (
            build_synthetic_task(tmp_path)
        )
        monkeypatch.setattr(cache_to_s1, "git_head", lambda: CONTRACT_GIT)
        monkeypatch.setattr(cache_to_s1, "git_branch", lambda: CONTRACT_BRANCH)
        output = tmp_path / "features" / "train.json"
        audit = cache_to_s1.emit_s1_payloads_from_cache(
            str(manifest_path), 0, str(Path(provenance["resolved_path"])),
            {"train": str(declared_path)}, {"train": str(output)},
            backbone_provenance=provenance,
        )
        assert audit["source"] == cache_to_s1.PAYLOAD_ORIGIN_KIND
        assert audit["encoder_calls"] == 0
        assert audit["splits"][0]["split"] == "train"
        assert audit["splits"][0]["sequence_matches_cache"]
        assert audit["manifest_sha256"] == qc.sha256_file(str(manifest_path))
        assert output.is_file()

    def test_emit_records_git_drift_but_binds_content(self, tmp_path, monkeypatch):
        # Post-cache commits change HEAD: content (backbone/impl/data) is the
        # runtime invariant; git drift is recorded for the report, not silent.
        manifest_path, declared_path, queries, ids, provenance, split_dir = (
            build_synthetic_task(tmp_path)
        )
        monkeypatch.setattr(cache_to_s1, "git_head", lambda: "0" * 40)
        monkeypatch.setattr(cache_to_s1, "git_branch", lambda: "feature-branch")
        output = tmp_path / "features" / "train.json"
        audit = cache_to_s1.emit_s1_payloads_from_cache(
            str(manifest_path), 0, str(Path(provenance["resolved_path"])),
            {"train": str(declared_path)}, {"train": str(output)},
            backbone_provenance=provenance,
        )
        assert audit["producer_git_sha"] == CONTRACT_GIT
        assert audit["runtime_git_sha"] == "0" * 40
        assert audit["runtime_contract"]["schema_match"]
        assert output.is_file()

    def test_emit_fails_closed_on_backbone_mismatch(self, tmp_path, monkeypatch):
        manifest_path, declared_path, queries, ids, provenance, split_dir = (
            build_synthetic_task(tmp_path)
        )
        monkeypatch.setattr(cache_to_s1, "git_head", lambda: CONTRACT_GIT)
        monkeypatch.setattr(cache_to_s1, "git_branch", lambda: CONTRACT_BRANCH)
        bad = dict(provenance)
        bad["backbone_hash"] = "0" * 64  # live backbone differs from the cache's
        with pytest.raises(ValueError, match="backbone"):
            cache_to_s1.emit_s1_payloads_from_cache(
                str(manifest_path), 0, str(Path(provenance["resolved_path"])),
                {"train": str(declared_path)}, {"train": str(tmp_path / "o.json")},
                backbone_provenance=bad,
            )
        assert not (tmp_path / "o.json").exists()
