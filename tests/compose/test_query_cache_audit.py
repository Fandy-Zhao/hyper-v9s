"""CPU audit tests for the V7 fixed-query cache (shard/merge/audit logic).

The single-vs-dual GPU numerical equivalence (spec check 12) and the
Top-2 routing smoke (check 19) require real GPUs and run through
``python -m compose.eval.precompute_v7_queries --mode gate``; every
sharding/merge/hash/invalidation rule is exercised here on the CPU.
"""

import json
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from compose.v7 import query_cache as qc
from compose.v7.query import full_train_task_center

QUERY_DIM = 1536


def fake_records(count: int) -> list:
    records = []
    for index in range(count):
        records.append({
            "image": "T0/train/img_{:05d}.jpg".format(index),
            "conversations": [
                {"from": "human", "value": "<image>\nQuestion {}?".format(index)},
                {"from": "gpt", "value": "Answer"},
            ],
        })
    return records


def fake_ids(count: int) -> list:
    return ["v7_t0_train_{}".format(index) for index in range(count)]


def normalized_rows(rows: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(rows, QUERY_DIM, generator=generator), dim=-1)


def base_contract(tmp_path, world_size: int = 2) -> qc.SplitContract:
    records = fake_records(64)
    split_file = tmp_path / "declared.json"
    split_file.write_text(json.dumps(records), encoding="utf-8")
    return qc.build_split_contract(
        git_sha="x" * 40,
        git_branch="test",
        task_index=0,
        task_name="ImageNet-R",
        split="train",
        source_dataset_path=str(split_file),
        records=records,
        image_folder=str(tmp_path),
        backbone_name="clip-vit-large-patch14-336",
        backbone_path=str(tmp_path / "clip"),
        backbone_hash="y" * 64,
        query_impl_hash="v7_fixed_layernorm_concat_l2_v1",
        world_size=world_size,
        batch_size=32,
    )


def write_fake_partials(tmp_path, ids, queries, world_size, contract_hash,
                        task_name="ImageNet-R"):
    partials = []
    for rank in range(world_size):
        rank_ids = qc.rank_sample_slice(ids, world_size, rank)
        rows = [ids.index(value) for value in rank_ids]
        path = str(tmp_path / "partial.r{}.pt".format(rank))
        qc.write_partial(
            path, rank=rank, world_size=world_size, contract_hash=contract_hash,
            task_index=0, task_name=task_name, split="train",
            sample_ids=rank_ids, queries=queries[rows], stats={},
        )
        partials.append(path)
    return partials


class TestShardPlan:
    def test_deterministic(self):
        ids = fake_ids(7)
        assert qc.rank_sample_slice(ids, 2, 0) == qc.rank_sample_slice(ids, 2, 0)
        assert qc.rank_sample_slice(ids, 2, 0) == [ids[i] for i in range(7) if i % 2 == 0]

    def test_no_overlap(self):
        ids = fake_ids(9)
        r0 = set(qc.rank_sample_slice(ids, 2, 0))
        r1 = set(qc.rank_sample_slice(ids, 2, 1))
        assert r0.isdisjoint(r1)

    def test_complete_union_and_no_drop(self):
        ids = fake_ids(9)  # odd count exercises the ceil split
        r0 = qc.rank_sample_slice(ids, 2, 0)
        r1 = qc.rank_sample_slice(ids, 2, 1)
        assert len(r0) + len(r1) == len(ids)
        assert sorted(r0 + r1) == sorted(ids)
        # every worker keeps declared-file order restricted to its residue
        assert r0 == [ids[i] for i in range(0, 9, 2)]
        assert r1 == [ids[i] for i in range(1, 9, 2)]


def run_fake_partials(tmp_path, count=17, world_size=2, seed=1):
    contract = base_contract(tmp_path, world_size)
    ids = fake_ids(count)
    queries = normalized_rows(count, seed=seed)
    partials = write_fake_partials(
        tmp_path, ids, queries, world_size, contract.contract_hash())
    return ids, queries, contract, partials


class TestMergeAudit:
    def test_merge_reconstructs_declared_order(self, tmp_path):
        ids, queries, contract, partials = run_fake_partials(tmp_path)
        merged_ids, merged, audit = qc.merge_partials(
            partials, contract_hash=contract.contract_hash(),
            declared_ids=ids, declared_count=len(ids))
        assert merged_ids == ids
        assert audit["merged_count"] == len(ids)
        assert audit["count_equality"] is True
        assert torch.equal(merged, queries)
        assert audit["rank0_sample_count"] == 9 and audit["rank1_sample_count"] == 8

    def test_world1_is_identity(self, tmp_path):
        ids, queries, contract, partials = run_fake_partials(tmp_path, count=11, world_size=1)
        assert len(partials) == 1
        merged_ids, merged, audit = qc.merge_partials(
            partials, contract_hash=contract.contract_hash(),
            declared_ids=ids, declared_count=len(ids))
        assert merged_ids == ids and torch.equal(merged, queries)

    def test_missing_partial_detected(self, tmp_path):
        ids, queries, contract, partials = run_fake_partials(tmp_path)
        with pytest.raises(ValueError, match="missing partial ranks"):
            qc.merge_partials(
                partials[:1], contract_hash=contract.contract_hash(),
                declared_ids=ids, declared_count=len(ids))

    def test_missing_sample_detected(self, tmp_path):
        ids, queries, contract, partials = run_fake_partials(tmp_path, count=10)
        # rank0 reports only 4 rows instead of 5 -> declared ids contain a
        # sample nobody covered.
        shorter = ids[:9]
        with pytest.raises(ValueError, match="coverage mismatch"):
            qc.merge_partials(
                partials, contract_hash=contract.contract_hash(),
                declared_ids=shorter, declared_count=len(shorter))

    def test_duplicate_partial_detected(self, tmp_path):
        ids, queries, contract, partials = run_fake_partials(tmp_path, count=10)
        with pytest.raises(ValueError, match="duplicate partial"):
            qc.merge_partials(
                partials + [partials[0]], contract_hash=contract.contract_hash(),
                declared_ids=ids, declared_count=len(ids))

    def test_unknown_sample_detected(self, tmp_path):
        ids, queries, contract, partials = run_fake_partials(tmp_path, count=8)
        # declared ids now contain an extra id nobody computed.
        with pytest.raises(ValueError, match="coverage mismatch"):
            qc.merge_partials(
                partials, contract_hash=contract.contract_hash(),
                declared_ids=ids + ["v7_t0_train_999"],
                declared_count=len(ids) + 1)

    def test_rank_out_of_order_rejected(self, tmp_path):
        contract = base_contract(tmp_path)
        ids = fake_ids(8)
        queries = normalized_rows(8, seed=2)
        rank_ids = list(reversed(qc.rank_sample_slice(ids, 2, 0)))
        bad = str(tmp_path / "bad.pt")
        qc.write_partial(
            bad, rank=0, world_size=2, contract_hash=contract.contract_hash(),
            task_index=0, task_name="ImageNet-R", split="train",
            sample_ids=rank_ids, queries=queries[[ids.index(v) for v in rank_ids]],
            stats={})
        good_rows = [ids.index(v) for v in qc.rank_sample_slice(ids, 2, 1)]
        good = str(tmp_path / "good.pt")
        qc.write_partial(
            good, rank=1, world_size=2, contract_hash=contract.contract_hash(),
            task_index=0, task_name="ImageNet-R", split="train",
            sample_ids=qc.rank_sample_slice(ids, 2, 1),
            queries=queries[good_rows], stats={})
        # correct set, wrong within-rank order -> per-rank determinism check
        with pytest.raises(ValueError, match="deterministic slice"):
            qc.merge_partials(
                [bad, good], contract_hash=contract.contract_hash(),
                declared_ids=ids, declared_count=len(ids))

    def test_corrupted_partial_detected(self, tmp_path):
        contract = base_contract(tmp_path)
        ids = fake_ids(4)
        queries = normalized_rows(4, seed=3)
        good = str(tmp_path / "good.pt")
        qc.write_partial(
            good, rank=0, world_size=1, contract_hash=contract.contract_hash(),
            task_index=0, task_name="ImageNet-R", split="train",
            sample_ids=ids, queries=queries, stats={})
        # truncated bytes
        truncated = str(tmp_path / "trunc.pt")
        with open(good, "rb") as handle:
            data = handle.read()
        with open(truncated, "wb") as handle:
            handle.write(data[: len(data) // 2])
        with pytest.raises(Exception):
            qc.load_partial(truncated)
        # wrong dtype contract
        payload = torch.load(good, map_location="cpu")
        payload["dtype"] = "bfloat16"
        wrong_dtype = str(tmp_path / "dtype.pt")
        torch.save(payload, wrong_dtype)
        with pytest.raises(ValueError, match="dtype"):
            qc.load_partial(wrong_dtype)
        # non-finite rows
        payload = torch.load(good, map_location="cpu")
        payload["queries"][0, 0] = float("nan")
        nan_path = str(tmp_path / "nan.pt")
        torch.save(payload, nan_path)
        with pytest.raises(ValueError, match="non-finite"):
            qc.load_partial(nan_path)

    def test_query_dim_enforced(self, tmp_path):
        ids = fake_ids(3)
        with pytest.raises(ValueError, match="1536"):
            qc.write_partial(
                str(tmp_path / "x.pt"), rank=0, world_size=1,
                contract_hash="h", task_index=0, task_name="T", split="train",
                sample_ids=ids, queries=torch.zeros(3, 1535), stats={})

    def test_stale_contract_rejected(self, tmp_path):
        contract = base_contract(tmp_path)
        other = base_contract(tmp_path)
        other_contract = qc.SplitContract(
            **{**contract.__dict__,
               "source_dataset_sha256": "stale" * 16}) if hasattr(other, "__dict__") else None
        ids = fake_ids(4)
        queries = normalized_rows(4, seed=4)
        partial = str(tmp_path / "p.pt")
        qc.write_partial(
            partial, rank=0, world_size=1, contract_hash=contract.contract_hash(),
            task_index=0, task_name="ImageNet-R", split="train",
            sample_ids=ids, queries=queries, stats={})
        # fresh run computes a different contract hash -> stale shard fails
        newer = base_contract(tmp_path, world_size=1)
        assert qc.partial_valid_for(partial, newer.contract_hash(), ids) is False
        assert qc.partial_valid_for(partial, contract.contract_hash(), ids) is True
        with pytest.raises(ValueError, match="stale contract"):
            qc.merge_partials(
                [partial], contract_hash=newer.contract_hash(),
                declared_ids=ids, declared_count=len(ids))
        assert other_contract is not None  # silence unused-name lint


class TestAtomicWrites:
    def test_no_tmp_leftovers_on_success(self, tmp_path):
        ids = fake_ids(5)
        queries = normalized_rows(5, seed=5)
        qc.write_partial(
            str(tmp_path / "p.pt"), rank=0, world_size=1, contract_hash="h",
            task_index=0, task_name="T", split="train",
            sample_ids=ids, queries=queries, stats={})
        assert list(tmp_path.glob("*.tmp")) == []
        assert list(tmp_path.glob("*.pt")) == [tmp_path / "p.pt"]
        assert (tmp_path / "p.pt.json").is_file()

    def test_no_partial_on_failure(self, tmp_path):
        ids = fake_ids(3)
        with pytest.raises(ValueError):
            qc.write_partial(
                str(tmp_path / "bad.pt"), rank=0, world_size=1,
                contract_hash="h", task_index=0, task_name="T", split="train",
                sample_ids=ids, queries=torch.zeros(3, 1535), stats={})
        assert list(tmp_path.glob("bad.pt*")) == []

    def test_official_cache_atomic_and_readable(self, tmp_path):
        contract = base_contract(tmp_path)
        ids = fake_ids(6)
        queries = normalized_rows(6, seed=6)
        cache_dir = tmp_path / "cache"
        qc.write_split_cache(
            str(cache_dir), contract=contract, sample_ids=ids, queries=queries,
            merge_audit={"merged_count": 6}, worker_stats={},
            runtime={"physical_gpus": [0, 1]},
            created_at="2026-09-03T00:00:00+0000")
        # garbage tmp from a crashed writer must not confuse readers
        (cache_dir / "queries.pt.junk.tmp").write_bytes(b"junk")
        loaded_ids, loaded = qc.read_split_cache(str(cache_dir))
        assert list(loaded_ids) == ids and torch.equal(loaded, queries)
        metadata = json.loads((cache_dir / "metadata.json").read_text())
        assert metadata["runtime"]["physical_gpus"] == [0, 1]
        assert metadata["num_declared_samples"] == metadata["num_saved_queries"] == 6
        assert metadata["num_unique_sample_ids"] == 6


class TestHashesAndValidity:
    def test_metadata_hash_validation(self, tmp_path):
        contract = base_contract(tmp_path)
        ids = fake_ids(8)
        queries = normalized_rows(8, seed=7)
        cache_dir = tmp_path / "cache"
        qc.write_split_cache(
            str(cache_dir), contract=contract, sample_ids=ids, queries=queries,
            merge_audit={}, worker_stats={}, runtime={},
            created_at="2026-09-03T00:00:00+0000")
        metadata = json.loads((cache_dir / "metadata.json").read_text())
        assert metadata["sample_id_set_hash"] == qc.sample_id_set_hash(ids)
        assert metadata["query_tensor_hash"] == qc.query_tensor_hash(ids, queries)
        assert qc.split_cache_valid(str(cache_dir), contract.contract_hash()) is True
        # stale (different contract) cache must be invalidated
        other = base_contract(tmp_path, world_size=1)
        assert qc.split_cache_valid(str(cache_dir), other.contract_hash()) is False
        # tamper one value -> metadata hash no longer matches content
        payload = torch.load(cache_dir / "queries.pt", map_location="cpu")
        payload["queries"][0, 5] += 1.0
        torch.save(payload, cache_dir / "queries.pt")
        assert qc.query_tensor_hash(ids, payload["queries"]) != metadata["query_tensor_hash"]

    def test_stale_official_cache_detected(self, tmp_path):
        contract = base_contract(tmp_path)
        ids = fake_ids(4)
        queries = normalized_rows(4, seed=8)
        cache_dir = tmp_path / "cache"
        qc.write_split_cache(
            str(cache_dir), contract=contract, sample_ids=ids, queries=queries,
            merge_audit={}, worker_stats={}, runtime={},
            created_at="2026-09-03T00:00:00+0000")
        assert qc.split_cache_valid(str(cache_dir), contract.contract_hash())
        # missing metadata = invalid
        (cache_dir / "metadata.json").unlink()
        assert qc.split_cache_valid(str(cache_dir), contract.contract_hash()) is False

    def test_content_hash_changes_with_question(self):
        records = fake_records(2)
        first = qc.split_content_hash(records)
        records[1]["conversations"][0]["value"] = "<image>\nOther question?"
        second = qc.split_content_hash(records)
        assert first != second
        records[0]["image"] = "T0/other/img.jpg"
        assert qc.split_content_hash(records) != second


class TestNormAndCenter:
    def test_norm_stats_of_normalized_queries(self):
        queries = normalized_rows(10, seed=9)
        stats = qc.norm_stats(queries)
        assert 1.0 - 1e-4 <= stats["norm_min"] <= stats["norm_max"] <= 1.0 + 1e-4

    def test_center_equivalence_merged_vs_direct(self, tmp_path):
        ids, queries, contract, partials = run_fake_partials(tmp_path, count=17)
        _merged_ids, merged, _audit = qc.merge_partials(
            partials, contract_hash=contract.contract_hash(),
            declared_ids=ids, declared_count=len(ids))
        center_a, coverage_a = full_train_task_center(queries, len(ids))
        center_b, coverage_b = full_train_task_center(merged, len(ids))
        assert torch.equal(center_a, center_b)
        assert coverage_a["num_queries_used_for_center"] == len(ids)
        assert coverage_b["num_queries_used_for_center"] == len(ids)

    def test_center_rejects_partial_train_sets(self):
        queries = normalized_rows(10, seed=10)
        with pytest.raises(ValueError, match="mismatch"):
            full_train_task_center(queries, 9)
        with pytest.raises(ValueError, match="shape"):
            full_train_task_center(torch.zeros(3, 100), 3)

    def test_center_written_and_loadable(self, tmp_path):
        contract = base_contract(tmp_path)
        ids = fake_ids(5)
        queries = normalized_rows(5, seed=11)
        cache_dir = tmp_path / "cache"
        qc.write_split_cache(
            str(cache_dir), contract=contract, sample_ids=ids, queries=queries,
            merge_audit={}, worker_stats={}, runtime={},
            created_at="2026-09-03T00:00:00+0000")
        center_path, payload = qc.write_task_center(
            str(tmp_path), contract=contract, queries=queries,
            source_query_cache_hash="cafebabe", created_at="2026-09-03")
        assert payload["num_queries_used_for_center"] == 5
        loaded = qc.load_task_center(str(center_path))
        assert loaded["num_queries_used_for_center"] == 5
        expected = F.normalize(queries.float().mean(dim=0), dim=0)
        assert torch.allclose(loaded["center"], expected, atol=1e-7)


class TestComparisons:
    def test_identical_queries_match_exactly(self):
        queries = normalized_rows(6, seed=12)
        comparison = qc.compare_by_sample_id(
            ["s{}".format(i) for i in range(6)], queries,
            ["s{}".format(i) for i in range(6)], queries.clone())
        assert comparison["exact_bit_equal"] is True
        assert comparison["cosine_mean"] == 1.0
        assert comparison["max_abs_diff"] == 0.0

    def test_perturbation_reported(self):
        queries = normalized_rows(4, seed=13)
        perturbed = queries.clone()
        perturbed[0, 0] += 1e-4
        perturbed = F.normalize(perturbed, dim=-1)
        comparison = qc.compare_by_sample_id(
            ["s{}".format(i) for i in range(4)], queries,
            ["s{}".format(i) for i in range(4)], perturbed)
        assert comparison["exact_bit_equal"] is False
        assert comparison["max_abs_diff"] > 0.0
        assert comparison["cosine_min"] >= 1.0 - 1e-2
        assert comparison["cosine_mean"] <= 1.0 + 1e-6  # fp round-up tolerance

    def test_cosine_min_threshold(self):
        # synthetic adversarial: one row flipped half-way
        queries = F.normalize(torch.randn(3, QUERY_DIM), dim=-1)
        flipped = queries.clone()
        flipped[1] = -flipped[1]
        comparison = qc.compare_by_sample_id(
            ["a", "b", "c"], queries, ["a", "b", "c"], flipped)
        assert comparison["cosine_min"] == pytest.approx(-1.0, abs=1e-5)
        assert comparison["cosine_min"] < 1.0 - 1e-6  # would FAIL the gate


class TestRecordContract:
    def test_sample_id_priority(self):
        assert qc.sample_id_of({"id": "a", "question_id": "b"}) == "a"
        assert qc.sample_id_of({"question_id": "b"}) == "b"
        with pytest.raises(ValueError):
            qc.sample_id_of({"image": "x.jpg"})

    def test_content_hash_stable_order(self):
        records = fake_records(3)
        assert qc.split_content_hash(records) == qc.split_content_hash(
            [dict(value) for value in records])
        assert qc.split_content_hash(records) != qc.split_content_hash(records[:2])


# ---------------------------------------------------------------------------
# Runtime readers (0903 spec §5-6): one read-only, sample_id-keyed cache
# entry point for every downstream stage; misses fail closed.
# ---------------------------------------------------------------------------

CONTRACT_GIT = "x" * 40
CONTRACT_BRANCH = "test"
CONTRACT_BACKBONE = "y" * 64
CONTRACT_IMPL = "v7_fixed_layernorm_concat_l2_v1"


def split_ids(split: str, count: int) -> list:
    return ["v7_t0_{}_{}".format(split, index) for index in range(count)]


def write_synthetic_split(tmp_path, split, count, seed):
    """One tiny split cache artifact (queries.pt + metadata.json)."""
    declared_path = tmp_path / "declared.{}.json".format(split)
    declared_path.write_text(json.dumps(fake_records(count)), encoding="utf-8")
    contract = qc.build_split_contract(
        git_sha=CONTRACT_GIT, git_branch=CONTRACT_BRANCH, task_index=0,
        task_name="ImageNet-R", split=split,
        source_dataset_path=str(declared_path),
        records=fake_records(count), image_folder=str(tmp_path),
        backbone_name="clip-vit-large-patch14-336",
        backbone_path=str(tmp_path / "clip"), backbone_hash=CONTRACT_BACKBONE,
        query_impl_hash=CONTRACT_IMPL, world_size=2, batch_size=32,
    )
    directory = tmp_path / "query_cache" / "task0" / split
    directory.mkdir(parents=True, exist_ok=True)
    ids = split_ids(split, count)
    queries = normalized_rows(count, seed=seed)
    qc.write_split_cache(
        str(directory), contract=contract, sample_ids=ids, queries=queries,
        merge_audit={}, worker_stats={}, runtime={},
        created_at="2026-09-03T00:00:00",
    )
    return directory, ids, queries, contract


def write_synthetic_manifest(tmp_path, splits=("train", "val", "test")):
    """Cache root + manifest mirroring the real precompute layout."""
    tasks = {}
    for split in splits:
        directory, ids, _, contract = write_synthetic_split(
            tmp_path, split, 8, seed={"train": 1, "val": 2, "test": 3}[split])
        metadata = json.loads((directory / "metadata.json").read_text())
        tasks.setdefault("ImageNet-R", {})[split] = {
            "path": str(directory / "queries.pt"),
            "metadata_path": str(directory / "metadata.json"),
            "contract_hash": metadata["contract_hash"],
            "sample_count": metadata["num_declared_samples"],
            "sample_id_hash": metadata["sample_id_set_hash"],
            "query_hash": metadata["query_tensor_hash"],
            "source_dataset_sha256": metadata["contract"]["source_dataset_sha256"],
        }
    manifest = {
        "kind": "v7_fixed_query_cache_manifest",
        "schema_version": qc.QUERY_SCHEMA_VERSION,
        "query_mode": qc.QUERY_MODE,
        "query_dim": qc.QUERY_DIM,
        "dtype": qc.QUERY_DTYPE,
        "cache_root": str(tmp_path / "query_cache"),
        "git_sha": CONTRACT_GIT,
        "git_branch": CONTRACT_BRANCH,
        "tasks": tasks,
    }
    path = tmp_path / "query_cache_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return str(path), manifest


class TestV7QuerySplitReader:
    def test_get_and_get_batch_are_detached_and_order_preserving(self, tmp_path):
        directory, ids, queries, _ = write_synthetic_split(
            tmp_path, "train", 8, seed=1)
        reader = qc.V7QuerySplitReader(str(directory))
        assert reader.sample_ids == tuple(ids)
        assert reader.n == 8
        row = reader.get(ids[3])
        assert tuple(row.shape) == (QUERY_DIM,)
        assert row.dtype == torch.float32
        assert row.device.type == "cpu"
        assert not row.requires_grad  # read-only, never enters an autograd graph
        assert torch.equal(row, queries[3])
        batch = reader.get_batch([ids[7], ids[0], ids[3]])
        assert tuple(batch.shape) == (3, QUERY_DIM)
        assert torch.equal(batch[2], queries[3])  # caller order, not file order
        assert all(reader.contains(value) for value in ids)
        assert not reader.contains("not-a-sample")

    def test_expected_count_mismatch_fails(self, tmp_path):
        directory, _, _, _ = write_synthetic_split(tmp_path, "train", 8, seed=1)
        with pytest.raises(ValueError):
            qc.V7QuerySplitReader(str(directory), expected_count=7)

    def test_miss_raises_query_cache_miss_error(self, tmp_path):
        directory, ids, _, _ = write_synthetic_split(tmp_path, "val", 8, seed=2)
        reader = qc.V7QuerySplitReader(str(directory))
        with pytest.raises(qc.QueryCacheMissError):
            reader.get("v7_t0_train_999")  # wrong split id: hard miss
        with pytest.raises(qc.QueryCacheMissError):
            reader.get_batch([ids[0], "missing_0", "missing_1"])  # fail closed
        assert reader.get_batch(ids[:2]).shape[0] == 2

    def test_validate_split_audit(self, tmp_path):
        directory, ids, queries, _ = write_synthetic_split(
            tmp_path, "train", 8, seed=1)
        audit = qc.validate_split(str(directory))
        assert audit["saved_count"] == 8
        assert audit["unique_sample_id_count"] == 8
        assert audit["duplicate_sample_ids"] == 0
        assert audit["query_dim"] == QUERY_DIM
        assert audit["query_dtype"] == "float32"
        assert audit["all_finite"]
        assert audit["norm_in_tolerance"]
        assert audit["num_declared_samples"] == audit["num_saved_queries"] == 8
        assert audit["query_tensor_hash"] == qc.query_tensor_hash(ids, queries)

    def test_corrupt_or_foreign_payload_is_rejected(self, tmp_path):
        directory, _, _, _ = write_synthetic_split(tmp_path, "test", 8, seed=3)
        # foreign kind
        payload = torch.load(str(directory / "queries.pt"), map_location="cpu")
        payload["kind"] = "something_else"
        bad = tmp_path / "foreign" / "test"
        bad.mkdir(parents=True)
        torch.save(payload, str(bad / "queries.pt"))
        with pytest.raises(ValueError):
            qc.V7QuerySplitReader(str(bad))
        # wrong dim
        payload = torch.load(str(directory / "queries.pt"), map_location="cpu")
        payload["query_dim"] = 1024
        payload["queries"] = payload["queries"][:, :1024]
        bad = tmp_path / "wrongdim" / "test"
        bad.mkdir(parents=True)
        torch.save(payload, str(bad / "queries.pt"))
        with pytest.raises(ValueError):
            qc.V7QuerySplitReader(str(bad))
        # duplicate ids
        payload = torch.load(str(directory / "queries.pt"), map_location="cpu")
        dup_ids = list(payload["sample_ids"]) + [payload["sample_ids"][0]]
        payload["sample_ids"] = dup_ids
        payload["queries"] = torch.cat([payload["queries"], payload["queries"][:1]])
        bad = tmp_path / "dups" / "test"
        bad.mkdir(parents=True)
        torch.save(payload, str(bad / "queries.pt"))
        with pytest.raises(ValueError):
            qc.V7QuerySplitReader(str(bad))


class TestV7CacheManifest:
    def test_locate_and_task_index_roundtrip(self, tmp_path):
        manifest_path, _ = write_synthetic_manifest(tmp_path)
        manifest = qc.V7CacheManifest(manifest_path)
        assert manifest.task_index("ImageNet-R") == 0
        assert manifest.task_name(0) == "ImageNet-R"
        assert manifest.manifest_sha256() == qc.sha256_file(manifest_path)
        assert qc.V7CacheManifest.locate(str(tmp_path)).path == manifest_path
        assert qc.V7CacheManifest.locate(manifest_path).path == manifest_path

    def test_runtime_contract_pass_path_binds_split_metadata(self, tmp_path):
        manifest_path, _ = write_synthetic_manifest(tmp_path)
        manifest = qc.V7CacheManifest(manifest_path)
        verdict = manifest.validate_runtime_contract(
            git_sha=CONTRACT_GIT, git_branch=CONTRACT_BRANCH,
            backbone_hash=CONTRACT_BACKBONE, impl_hash=CONTRACT_IMPL,
            required=("train", "val", "test"),
        )
        assert verdict["schema_match"]
        assert verdict["problems"] == []
        assert set(verdict["runtime_checks"]) == {
            "task0.train", "task0.val", "task0.test"}
        # reader is count-bound to the manifest declaration
        assert manifest.reader(0, "train").n == 8

    def test_fail_closed_on_git_mismatch(self, tmp_path):
        manifest_path, _ = write_synthetic_manifest(tmp_path)
        manifest = qc.V7CacheManifest(manifest_path)
        with pytest.raises(ValueError, match="git"):
            manifest.validate_runtime_contract(git_sha="0" * 40)

    def test_fail_closed_on_backbone_mismatch(self, tmp_path):
        manifest_path, _ = write_synthetic_manifest(tmp_path)
        manifest = qc.V7CacheManifest(manifest_path)
        with pytest.raises(ValueError, match="backbone"):
            manifest.validate_runtime_contract(
                git_sha=CONTRACT_GIT, git_branch=CONTRACT_BRANCH,
                backbone_hash="0" * 64)

    def test_fail_closed_on_missing_required_split(self, tmp_path):
        manifest_path, _ = write_synthetic_manifest(
            tmp_path, splits=("train", "val"))
        manifest = qc.V7CacheManifest(manifest_path)
        with pytest.raises(ValueError, match="missing from the manifest"):
            manifest.validate_runtime_contract(
                required=("train", "val", "test"))

    def test_fail_closed_when_sidecar_tampered(self, tmp_path):
        manifest_path, manifest_json = write_synthetic_manifest(tmp_path)
        manifest_path = Path(manifest_path)
        # tamper: manifest contract_hash no longer matches split metadata
        manifest_json["tasks"]["ImageNet-R"]["train"]["contract_hash"] = "f" * 64
        manifest_path.write_text(json.dumps(manifest_json), encoding="utf-8")
        with pytest.raises(ValueError, match="contract_hash"):
            qc.V7CacheManifest(str(manifest_path)).validate_runtime_contract(
                required=("train",))
        # tamper: split metadata count disagrees with the manifest
        manifest_path, _ = write_synthetic_manifest(tmp_path)
        manifest_path = Path(manifest_path)
        metadata_path = tmp_path / "query_cache" / "task0" / "train" / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["num_declared_samples"] = 9
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(ValueError, match="sample count"):
            qc.V7CacheManifest(str(manifest_path)).validate_runtime_contract(
                required=("train",))

    def test_task_center_loader_alias(self):
        assert qc.get_task_center is qc.load_task_center
