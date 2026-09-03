"""Phase A independent re-verification of the V7 fixed-query cache (0903 spec §2-3).

Read-only and CPU-only: trusts nothing from earlier reports or report text.
For every manifest split it recomputes, from the *live* declared source files,
the on-disk cache binaries and the current runtime module constants / live
CLIP backbone provenance:

  - declared == saved == unique sample counts, duplicates = 0;
  - cache id sequence == declared id sequence (missing / extra / reordered);
  - tensor dim 1536 fp32, all finite, L2 norms inside tolerance;
  - sample_id_set_hash and query_tensor_hash recomputed == metadata == manifest;
  - source_dataset_sha256 and source_content_hash recomputed from the live
    declared file == metadata contract;
  - runtime binding (query mode / impl hash / backbone content hash / schema /
    dim / dtype) recomputed from live module constants + live backbone dir;
  - producer-vs-runtime git pair recorded (drift alone is reported, never fatal:
    cache is content-bound, decision record in the adaptation report §3).

Task centers are recomputed from the full cached train rows and compared to
the stored task_center.pt payloads (num_queries_used == full train count,
center max-abs-diff / cosine within fp32 accumulation tolerance).

Emits the spec §3 "QUERY CACHE PRECHECK" block and writes machine-readable
evidence to artifacts/v7_query_cache/reverification_<ts>.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compose"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import compose.v7.query_cache as qc  # noqa: E402
from compose.eval.query_features import query_backbone_provenance  # noqa: E402
from compose.eval.precompute_v7_queries import git_head, git_branch  # noqa: E402
from compose.v7.query import FixedQueryProvenance, full_train_task_center  # noqa: E402

EVIDENCE_DIR = ROOT / "artifacts" / "v7_query_cache"


def _records_of(path: str) -> List[Dict[str, object]]:
    with open(path, encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("declared split is empty or invalid: {}".format(path))
    return records


def verify_split(
    manifest: qc.V7CacheManifest, task_name: str, split: str, info: Dict[str, object]
) -> Dict[str, object]:
    result: Dict[str, object] = {"task": task_name, "split": split}
    reader = manifest.reader(manifest.task_index(task_name), split)
    meta = reader.metadata
    contract = meta["contract"]

    # 1. counts: declared == saved == unique
    declared_path = str(contract["source_dataset_path"])
    records = _records_of(declared_path)
    declared_ids = [qc.sample_id_of(record) for record in records]
    declared_unique = len(set(declared_ids))
    saved = int(meta["num_saved_queries"])
    unique = int(meta["num_unique_sample_ids"])
    result["declared_sample_count"] = len(records)
    result["saved_query_count"] = saved
    result["unique_sample_id_count"] = unique
    result["declared_unique_sample_id_count"] = declared_unique
    result["count_equality"] = bool(
        len(records) == saved == unique == declared_unique
        == int(info["sample_count"]) == reader.n
    )
    result["duplicate_declared_ids"] = int(len(declared_ids) - declared_unique)

    # 2. cache id sequence == declared id sequence
    cache_ids = list(meta["ordered_sample_ids"])
    result["missing_sample_ids"] = int(
        sum(1 for sid in declared_ids if sid not in reader._index)
    )
    result["unknown_sample_ids"] = int(
        sum(1 for sid in cache_ids if sid not in set(declared_ids))
    )
    result["cache_ids_match_declared_order"] = bool(cache_ids == declared_ids)

    # 3. tensor: dim/dtype/finite/norms
    tensor = reader.queries
    result["query_dim"] = int(tensor.shape[1])
    result["query_dtype"] = str(tensor.dtype).split(".")[-1]
    result["all_finite"] = bool(torch.isfinite(tensor).all())
    stats = qc.norm_stats(tensor.detach().cpu())
    result["norm_min"] = stats["norm_min"]
    result["norm_max"] = stats["norm_max"]
    result["norm_in_tolerance"] = bool(
        (1.0 - stats["norm_min"]) <= 1e-6 and (stats["norm_max"] - 1.0) <= 1e-6
    )

    # 4. rehash: ids + tensor bytes
    recomputed = reader.rehash()
    result["sample_id_set_hash_matches"] = bool(
        recomputed["sample_id_set_hash"] == meta["sample_id_set_hash"]
    )
    result["query_tensor_hash_matches_metadata"] = bool(
        recomputed["query_tensor_hash"] == meta["query_tensor_hash"]
    )
    result["query_hash_matches_manifest"] = bool(
        recomputed["query_tensor_hash"] == info["query_hash"]
    )

    # 5. live source binding
    result["source_dataset_sha256_matches"] = bool(
        qc.sha256_file(declared_path) == contract["source_dataset_sha256"]
    )
    result["source_content_hash_matches"] = bool(
        qc.split_content_hash(records) == contract["source_content_hash"]
    )

    # 6. runtime binding (content fields only)
    backbone = query_backbone_provenance(contract["query_backbone_path"])
    result["backbone_path_resolves"] = str(
        Path(contract["query_backbone_path"]).expanduser().resolve()
    ) == str(backbone["resolved_path"])
    result["backbone_hash_matches_live"] = bool(
        backbone["backbone_hash"] == contract["query_backbone_hash"]
    )
    result["impl_hash_matches_live"] = bool(
        FixedQueryProvenance().module_hash == contract["query_impl_hash"]
    )
    result["mode_matches_runtime"] = bool(
        contract["query_mode"] == qc.QUERY_MODE
        and contract["schema_version"] == qc.QUERY_SCHEMA_VERSION
        and contract["query_dim"] == qc.QUERY_DIM
        and contract["query_dtype"] == qc.QUERY_DTYPE
    )
    result["content_bound"] = bool(
        result["source_dataset_sha256_matches"]
        and result["source_content_hash_matches"]
        and result["backbone_hash_matches_live"]
        and result["impl_hash_matches_live"]
        and result["mode_matches_runtime"]
    )
    result["PASS"] = bool(
        result["count_equality"]
        and result["cache_ids_match_declared_order"]
        and result["missing_sample_ids"] == 0
        and result["unknown_sample_ids"] == 0
        and result["query_dim"] == qc.QUERY_DIM
        and result["query_dtype"] == "float32"
        and result["all_finite"]
        and result["norm_in_tolerance"]
        and result["sample_id_set_hash_matches"]
        and result["query_tensor_hash_matches_metadata"]
        and result["query_hash_matches_manifest"]
        and result["content_bound"]
    )
    return result


def verify_task_center(manifest: qc.V7CacheManifest, task_name: str) -> Dict[str, object]:
    task_index = manifest.task_index(task_name)
    train_meta = manifest.reader(task_index, "train").metadata
    center_path = str(train_meta["task_center"]["path"])
    payload = qc.load_task_center(center_path)
    center = payload["center"]
    declared = int(payload["num_declared_train_samples"])
    used = int(payload["num_queries_used_for_center"])
    # recompute the center from the full cached train rows
    reader = manifest.reader(task_index, "train")
    recomputed, coverage = full_train_task_center(
        reader.queries.detach().cpu(), reader.n
    )
    diff = (center - recomputed).abs()
    cos = torch.nn.functional.cosine_similarity(center, recomputed, dim=0).item()
    result = {
        "task": task_name,
        "center_path": center_path,
        "num_queries_used_for_center": used,
        "num_declared_train_samples": declared,
        "recomputed_num_queries": int(coverage["num_queries_used_for_center"]),
        "counts_match_full_train": bool(declared == reader.n == used),
        "center_max_abs_diff": float(diff.max().item()),
        "center_mean_abs_diff": float(diff.mean().item()),
        "center_cosine": float(cos),
        "PASS": bool(
            declared == reader.n == used
            and diff.max().item() < 1e-5
            and cos > 1.0 - 1e-5
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(EVIDENCE_DIR / "query_cache_manifest.json"),
        help="path to query_cache_manifest.json",
    )
    parser.add_argument(
        "--evidence-out",
        default=None,
        help="json evidence output path (default: artifacts/v7_query_cache/reverification_<ts>.json)",
    )
    args = parser.parse_args()

    started = time.time()
    manifest = qc.V7CacheManifest(args.manifest)
    runtime_git_sha = git_head()
    runtime_git_branch = git_branch()
    manifest_sha = manifest.manifest_sha256()

    # manifest-level runtime contract fields
    top_level = {
        "kind": manifest.query_mode == qc.QUERY_MODE,
        "schema_version": manifest.schema_version == qc.QUERY_SCHEMA_VERSION,
        "query_dim": manifest.query_dim == qc.QUERY_DIM,
        "dtype": manifest.dtype == qc.QUERY_DTYPE,
    }

    split_results: List[Dict[str, object]] = []
    center_results: List[Dict[str, object]] = []
    all_pass = True
    for task_name in sorted(manifest.payload["tasks"]):
        for split in ("train", "val", "test"):
            info = manifest.split_info(manifest.task_index(task_name), split)
            res = verify_split(manifest, task_name, split, info)
            split_results.append(res)
            all_pass &= bool(res["PASS"])
        center_results.append(verify_task_center(manifest, task_name))
        all_pass &= bool(center_results[-1]["PASS"])

    task_pass: Dict[str, bool] = {}
    for task_name in sorted(manifest.payload["tasks"]):
        task_pass[task_name] = all(
            r["PASS"]
            for r in split_results
            if r["task"] == task_name and r["split"] in ("train", "val", "test")
        )

    evidence = {
        "kind": "v7_query_cache_phase_a_reverification",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "manifest_path": str(Path(args.manifest).resolve()),
        "manifest_sha256": manifest_sha,
        "producer_git_sha": manifest.payload.get("git_sha"),
        "producer_git_branch": manifest.payload.get("git_branch"),
        "runtime_git_sha": runtime_git_sha,
        "runtime_git_branch": runtime_git_branch,
        "git_drift": bool(runtime_git_sha != manifest.payload.get("git_sha")),
        "manifest_top_level_contract": top_level,
        "total_split_count": len(split_results),
        "splits": split_results,
        "task_centers": center_results,
        "ALL_SPLITS_PASS": bool(all_pass),
        "wall_seconds": round(time.time() - started, 2),
    }

    out_path = Path(
        args.evidence_out
        or str(EVIDENCE_DIR / ("reverification_%s.json" % time.strftime("%Y%m%d_%H%M%S")))
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---- spec §3 precheck printout ---------------------------------------
    print("=" * 72)
    print("QUERY CACHE PRECHECK")
    print("=" * 72)
    print("manifest            :", manifest.path)
    print("manifest sha256     :", manifest_sha)
    print("producer git        :", manifest.payload.get("git_sha"))
    print("runtime git         :", runtime_git_sha)
    print("git drift (recorded):", evidence["git_drift"])
    print("content-bound fields:", top_level)
    total = 0
    for res in split_results:
        total += int(res["saved_query_count"])
        print(
            "{:<10} {:<5}: {}  declared={} saved={} unique={} miss={} unk={} "
            "dim={} finite={} norm={} idhash={} qhash={} srchash={} cthash={}".format(
                res["task"],
                res["split"],
                "PASS" if res["PASS"] else "FAIL",
                res["declared_sample_count"],
                res["saved_query_count"],
                res["unique_sample_id_count"],
                res["missing_sample_ids"],
                res["unknown_sample_ids"],
                res["query_dim"],
                res["all_finite"],
                res["norm_in_tolerance"],
                res["sample_id_set_hash_matches"],
                res["query_tensor_hash_matches_metadata"],
                res["source_content_hash_matches"],
                res["content_bound"],
            )
        )
    print("-" * 72)
    for res in center_results:
        print(
            "{:<10} center: {}  used={}/{} recomputed={} max_abs_diff={:.2e} "
            "cosine={:.9f}".format(
                res["task"],
                "PASS" if res["PASS"] else "FAIL",
                res["num_queries_used_for_center"],
                res["num_declared_train_samples"],
                res["recomputed_num_queries"],
                res["center_max_abs_diff"],
                res["center_cosine"],
            )
        )
    print("-" * 72)
    coverage = all(
        r["PASS"] for r in split_results
    ) and all(r["PASS"] for r in center_results)
    print("FULL_SAMPLE_QUERY_COVERAGE =", "YES" if coverage else "NO",
          "(total saved queries %d)" % total)
    print("QUERY_CONTRACT_MATCH =", "YES" if coverage else "NO")
    print("QUERY_CACHE_READY =", "YES" if coverage else "NO")
    print("evidence            :", out_path)
    print("wall seconds        :", evidence["wall_seconds"])
    return 0 if coverage else 1


if __name__ == "__main__":
    sys.exit(main())
