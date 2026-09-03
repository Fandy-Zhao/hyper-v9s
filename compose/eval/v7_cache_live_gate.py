"""Phase B cache-vs-live equivalence gates (0903 spec §19-21).

Bounded real-data gates that compare the precomputed fixed-query cache
(``v7_fixed_query_cache_gpu01_20260903``) against a fresh *live* CLIP
forward computed on the same physical GPU with the producer's own
reference pipeline (``precompute_v7_queries._worker_forward`` - the exact
code and device math that produced the cache):

* QUERY_NUMERICAL_EQUIVALENCE  - per-sample fp32 rows: cache split rows vs
  live rows for the same declared sample_ids.  Verdict PASS iff the id
  sequences agree exactly and ``cosine_min >= 1 - 1e-6`` (the Phase A
  bounded-gate tolerance; bit equality is recorded but fp16 CLIP forward
  noise makes cosine the verdict bound).
* TOP2_ROUTING_EQUIVALENCE     - Global Top-2 over one expert pool
  (default: the dormant 2-GPU formal run's S2 candidate keys, the Phase A
  reference pool) fed with cache rows vs live rows.  Verdict PASS iff the
  per-sample top-2 expert ids agree on 100% of the bounded sample.

Both gates are content-bound fail closed: the live backbone and the
fixed-query implementation hash must match the cache's runtime contract
before any forward happens.  A FAIL exits non-zero and nothing is
recorded as passed.

Run pinned to a physical GPU through the worker-GPU env pattern
(``CUDA_VISIBLE_DEVICES=<physical>`` + ``--device cuda:0``), exactly like
``cached_selections`` and the S6 worker subprocesses.

Usage::

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=repo \\
      python -m compose.eval.v7_cache_live_gate \\
        --formal-config configs/v7_ucit_formal.yaml \\
        --method-config configs/v7_global_coevolution.yaml \\
        --cache-manifest artifacts/v7_query_cache/query_cache_manifest.json \\
        --task-index 0 --split train --limit 128 \\
        --pool-state /data/.../v7_ucit_formal_2gpu_b2a16_w4_seed42_20260903/task0/state/candidate_keys.pt \\
        --audit-output <run_root>/gate_task0_train.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import yaml

from compose.eval import precompute_v7_queries as prod
from compose.eval.query_features import query_backbone_provenance
from compose.v7 import query_cache as qc
from compose.v7.query import FixedMultimodalQuery, FixedQueryProvenance

# Phase A bounded-gate tolerance (report §13): fp16 CLIP forward noise is
# ~2.4e-7 in cosine; the single-vs-dual gate passed at cosine_min
# 0.99999976 >= 1 - 1e-6.  Bit equality is recorded, not required.
COSINE_MIN_TOLERANCE = 1.0 - 1.0e-6


def _method_query_path(method_config: str) -> str:
    with open(method_config, "r", encoding="utf-8") as handle:
        method = yaml.safe_load(handle)
    return str(method["query"]["path"])


def _physical_gpu() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if "," not in visible and visible.strip():
        return int(visible.strip())
    return -1


def _atomic_write_json(payload: Dict[str, object], path: str) -> None:
    directory = Path(path).parent
    directory.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    fd, tmp = __import__("tempfile").mkstemp(
        prefix=".audit-", suffix=".tmp", dir=str(directory)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def run_gate(
    *,
    formal_config: str,
    method_config: str,
    cache_manifest: str,
    task_index: int,
    split: str,
    limit: int,
    pool_state: Optional[str],
    device: str,
    batch_size: int,
    prefetch: int,
    decode_workers: int,
) -> Dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("gate requires a CUDA device (worker-GPU env)")
    formal = prod._load_formal_tasks(formal_config)
    if task_index < 0 or task_index >= len(formal["tasks"]):
        raise ValueError("task index {} out of formal range".format(task_index))
    task = formal["tasks"][task_index]
    if split not in task["splits"]:
        raise ValueError("split {} not declared for task{}".format(split, task_index))
    clip_path = _method_query_path(method_config)

    started = time.monotonic()
    # Content-bound precheck (fail closed before any GPU forward).
    manifest = qc.V7CacheManifest.locate(cache_manifest)
    manifest.validate_runtime_contract(
        required=(split,),
        backbone_hash=query_backbone_provenance(clip_path)["backbone_hash"],
        impl_hash=FixedQueryProvenance().module_hash,
    )

    records = prod._records_of(task["splits"][split])[: int(limit)]
    sample_ids = prod._sample_ids(records)
    if len(sample_ids) != int(limit):
        raise ValueError("declared split shorter than the bounded limit")
    reader = manifest.reader(task_index, split)
    if list(reader.sample_ids[: len(sample_ids)]) != sample_ids:
        raise ValueError(
            "cache split id sequence does not match the declared file for "
            "task{}.{}".format(task_index, split)
        )
    cache_queries = reader.get_batch(sample_ids, device=device)
    if cache_queries.shape != (len(sample_ids), qc.QUERY_DIM):
        raise ValueError("cache batch shape mismatch")

    clip, processor = prod._load_clip(clip_path, device)
    encoder = FixedMultimodalQuery().to(device).eval()
    live_queries, timing = prod._worker_forward(
        records,
        formal["image_folder"],
        device,
        clip,
        processor,
        encoder,
        batch_size,
        prefetch,
        decode_workers,
    )
    del clip, processor, encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    live_queries = live_queries.to(device)

    numerical = qc.compare_by_sample_id(
        sample_ids, cache_queries, sample_ids, live_queries
    )
    numerical["norm"] = qc.norm_stats(cache_queries)
    numerical_pass = numerical["cosine_min"] >= COSINE_MIN_TOLERANCE

    route: Dict[str, object] = {}
    route_pass = True
    if pool_state is not None:
        if not Path(pool_state).is_file():
            raise ValueError("pool state missing: {}".format(pool_state))
        route = qc.verify_route_agreement(
            pool_state, sample_ids, cache_queries, sample_ids, live_queries
        )
        route_pass = bool(
            route["agreement_exact"] and route["top2_agreement_rate"] == 1.0
        )

    verdict = bool(numerical_pass and route_pass)
    audit = {
        "gate_kind": "v7_phaseB_cache_vs_live_equivalence",
        "verdict": "PASS" if verdict else "FAIL",
        "task_index": int(task_index),
        "task_name": str(task["name"]),
        "split": str(split),
        "samples": int(len(sample_ids)),
        "tolerance": {"cosine_min": ">= {}".format(COSINE_MIN_TOLERANCE)},
        "manifest_sha256": manifest.manifest_sha256(),
        "cache_n": int(reader.n),
        "physical_gpu": _physical_gpu(),
        "device": str(device),
        "git_head": prod.git_head(),
        "branch": prod.git_branch(),
        "numerical": numerical,
        "route": route,
        "live_forward": timing,
        "wall_time_s": round(time.monotonic() - started, 3),
        "queries": {
            "QUERY_NUMERICAL_EQUIVALENCE": "PASS"
            if numerical_pass
            else "FAIL",
            "TOP2_ROUTING_EQUIVALENCE": "PASS"
            if route_pass
            else "FAIL",
        },
    }
    return audit


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--formal-config", required=True)
    parser.add_argument("--method-config", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--pool-state", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args(argv)

    audit = run_gate(
        formal_config=args.formal_config,
        method_config=args.method_config,
        cache_manifest=args.cache_manifest,
        task_index=args.task_index,
        split=args.split,
        limit=args.limit,
        pool_state=args.pool_state,
        device=args.device,
        batch_size=args.batch_size,
        prefetch=args.prefetch,
        decode_workers=args.decode_workers,
    )
    _atomic_write_json(audit, args.audit_output)
    print(json.dumps({key: audit[key] for key in (
        "gate_kind", "verdict", "task_name", "split", "samples", "physical_gpu",
    )}, sort_keys=True))
    print("QUERY_NUMERICAL_EQUIVALENCE: {}".format(
        audit["queries"]["QUERY_NUMERICAL_EQUIVALENCE"]
    ))
    numerical = audit["numerical"]
    print(
        "  task{}.{} cosine_min={} cosine_mean={} exact_bit_equal={} "
        "max_abs_diff={}".format(
            audit["task_index"], audit["split"], numerical["cosine_min"],
            numerical["cosine_mean"], numerical["exact_bit_equal"],
            numerical["max_abs_diff"],
        )
    )
    if audit["route"]:
        print("TOP2_ROUTING_EQUIVALENCE: {}".format(
            audit["queries"]["TOP2_ROUTING_EQUIVALENCE"]
        ))
        route = audit["route"]
        print(
            "  rate={} exact={} max_abs_score_diff={} visible_experts={}".format(
                route["top2_agreement_rate"], route["agreement_exact"],
                route["max_abs_score_diff"], route["visible_expert_ids"],
            )
        )
    print("audit written: {}".format(args.audit_output))
    return 0 if audit["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
