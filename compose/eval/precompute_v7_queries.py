"""GPU0/GPU1 fixed-query precompute orchestrator and workers (V7, 2026-09-03).

Standalone experiment entry point.  Does NOT touch the V7 task orchestrator,
training, RMS, pruning or evaluation main flows.  It produces a reusable,
audited binary Fixed Query cache over the *declared* UCIT split files.

The query is computed with the exact runtime contract of the live V7 routing
path (compose.eval.eval_task / compose.eval.query_features) and the math is
owned by ``compose.v7.query.FixedMultimodalQuery`` (never redefined here):

    q_i = L2Norm(concat(LN(z_visual), LN(z_text)))        # 1536-D, fp32
    z_visual = L2Norm(CLIP image embeds)   # fp16 CLIP forward -> fp32
    z_text   = L2Norm(CLIP text embeds)    # text = question_text(record)
    image    = PIL RGB + CLIPProcessor (336, padding/truncation like eval)

Cache artifact layout under ``--out-root``::

    query_cache/task{t}/{split}/{queries.pt, metadata.json}
    query_cache/task{t}/task_center.pt
    tmp/<task>.<split>.rank{r}.pt          (worker partials; deleted on merge)
    metrics/precompute_stats.json

Modes
-----
``precompute`` : orchestrator.  For every (task, split): spawns world_size
                 workers, one per physical GPU, each computing the
                 deterministic ``index % world_size`` shard of the declared
                 split; then merges, audits and atomically writes the cache.
``worker``     : child of the orchestrator; owns exactly one GPU.
``gate``       : bounded real-data gate: world-1 (GPU0) vs world-2
                 (GPU0+GPU1), per-sample numerical equivalence, task-center
                 equivalence and (optionally) Global Top-2 route smoke.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import yaml

from compose.eval.query_features import query_backbone_provenance
from compose.v7 import query_cache as qc
from compose.v7.query import FixedMultimodalQuery, FixedQueryProvenance


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def git_branch() -> str:
    return subprocess.check_output(
        ["git", "branch", "--show-current"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _load_formal_tasks(formal_config: str) -> Dict[str, object]:
    with open(formal_config, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    tasks = payload["tasks"]
    if len(tasks) != 6 or [int(t["task_index"]) for t in tasks] != list(range(6)):
        raise ValueError("formal config must declare exactly tasks 0..5 in order")
    with open(payload["method_config"], "r", encoding="utf-8") as handle:
        method = yaml.safe_load(handle)
    image_folder = payload["data"]["image_folder"]
    query = method["query"]
    return {
        "image_folder": image_folder,
        "backbone_name": query["backbone"],
        "backbone_path": query["path"],
        "query_dim": int(query["query_dim"]),
        "tasks": [
            {
                "task_index": int(entry["task_index"]),
                "name": str(entry["name"]),
                "splits": {
                    "train": str(entry["train_file"]),
                    "val": str(entry["validation_file"]),
                    "test": str(entry["test_file"]),
                },
            }
            for entry in tasks
        ],
    }


def _records_of(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("declared split is empty or invalid: {}".format(path))
    return records


def _sample_ids(records: Sequence[Dict[str, object]]) -> List[str]:
    return [qc.sample_id_of(record) for record in records]


def _partial_name(task_index: int, split: str, rank: int) -> str:
    return "task{}.{}.rank{}.pt".format(task_index, split, rank)


def _load_clip(backbone_path: str, device: str):
    from transformers import CLIPModel, CLIPProcessor

    clip = CLIPModel.from_pretrained(
        backbone_path, torch_dtype=torch.float16
    ).to(device).eval()
    processor = CLIPProcessor.from_pretrained(backbone_path)
    return clip, processor


def _make_contract(
    args, task: Dict[str, object], split: str, split_file: str,
    records: Sequence[Dict[str, object]], world_size: int,
) -> Dict[str, object]:
    """Full SplitContract + hash (device-independent bind of the cache)."""
    backbone = query_backbone_provenance(args.backbone_path)
    contract = qc.build_split_contract(
        git_sha=git_head(),
        git_branch=git_branch(),
        task_index=int(task["task_index"]),
        task_name=str(task["name"]),
        split=str(split),
        source_dataset_path=split_file,
        records=records,
        image_folder=args.image_folder,
        backbone_name=args.backbone_name,
        backbone_path=args.backbone_path,
        backbone_hash=backbone["backbone_hash"],
        query_impl_hash=FixedQueryProvenance().module_hash,
        world_size=int(world_size),
        batch_size=args.batch_size,
    )
    return {"contract": contract, "contract_hash": contract.contract_hash()}


# ---------------------------------------------------------------------------
# GPU utilization sampler (physical GPUs, orchestrator side).
# ---------------------------------------------------------------------------

class GpuSampler:
    """Samples utilization of the physical GPUs with nvidia-smi (~0.5 s)."""

    def __init__(self, physical_ids: Sequence[int]) -> None:
        self.physical_ids = [int(value) for value in physical_ids]
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._samples: Dict[int, List[int]] = {value: [] for value in self.physical_ids}
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu",
                        "--format=csv,noheader,nounits",
                        "--id={}".format(",".join(str(v) for v in self.physical_ids)),
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                time.sleep(0.5)
                continue
            with self._lock:
                for line in result.strip().splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) == 2:
                        try:
                            self._samples[int(parts[0])].append(int(parts[1]))
                        except (ValueError, KeyError):
                            pass
            self._stop.wait(0.5)

    def start(self) -> None:
        self._thread.start()

    def summary(self) -> Dict[int, Dict[str, float]]:
        with self._lock:
            return {
                value: {
                    "avg_util_percent": (sum(parts) / len(parts) if parts else None),
                    "samples": len(parts),
                }
                for value, parts in self._samples.items()
            }

    def stop(self) -> Dict[int, Dict[str, float]]:
        self._stop.set()
        self._thread.join(timeout=5.0)
        return self.summary()


def sample_util_while(physical_ids: Sequence[int], fn):
    """Run fn and report per-GPU average utilization over its window."""
    sampler = GpuSampler(physical_ids)
    sampler.start()
    try:
        result = fn()
    finally:
        util = sampler.stop()
    return result, util


# ---------------------------------------------------------------------------
# Worker: frozen CLIP + FixedMultimodalQuery over one deterministic shard.
# ---------------------------------------------------------------------------

def _worker_forward(
    records: Sequence[Dict[str, object]],
    image_folder: str,
    device: str,
    clip,
    processor,
    encoder: FixedMultimodalQuery,
    batch_size: int,
    prefetch: int,
    decode_workers: int,
):
    """Reference feature pipeline (identical calls to compose.eval.query_features).

    Images are PIL-decoded by worker threads *ahead* of the GPU batch that
    needs them; every batch still calls CLIPProcessor with the same
    padding=True / truncation=True composition as the reference path.
    Returns (queries [N,1536] fp32 in record order, timing stats).
    """
    from PIL import Image
    from compose.data.records import question_text

    n = len(records)
    if prefetch < 1 or decode_workers < 1:
        prefetch = 0
    window = prefetch * batch_size if prefetch else batch_size
    # The CLIP processor resizes every image to the configured shortest edge
    # with BICUBIC, serially in the main thread (dominant CPU cost for large
    # images).  Pre-resizing in the parallel decode threads with the *same*
    # rule (shortest-edge ``target``, truncating aspect math, same PIL
    # resample) leaves the later processor call a same-size copy, so the
    # produced tensors are bit-identical to the reference pipeline while the
    # GPU stays fed.  Disabled whenever the processor config says otherwise.
    try:
        image_processor = processor.image_processor
        resize_target = int(image_processor.size["shortest_edge"])
        resize_enabled = bool(
            getattr(image_processor, "do_resize", True)
            and getattr(image_processor, "do_convert_rgb", True)
        )
        from PIL.Image import Resampling
        resize_filter = getattr(
            image_processor, "resample",
            Resampling.BICUBIC) or Resampling.BICUBIC
    except Exception:
        resize_target, resize_enabled, resize_filter = 336, False, None

    def decode(index: int):
        record = records[index]
        started = time.monotonic()
        image_path = os.path.join(image_folder, str(record["image"]))
        if not os.path.isfile(image_path):
            raise ValueError("missing image: {}".format(image_path))
        image = Image.open(image_path).convert("RGB")
        if resize_enabled:
            width, height = image.size
            if width <= height:
                if width != resize_target:
                    image = image.resize(
                        (resize_target, int(resize_target * height / width)),
                        resample=resize_filter)
            elif height != resize_target:
                image = image.resize(
                    (int(resize_target * width / height), resize_target),
                    resample=resize_filter)
        text = question_text(record)
        return index, image, text, time.monotonic() - started

    executor = ThreadPoolExecutor(max_workers=decode_workers) if prefetch else None
    future_map: Dict[int, object] = {}
    submitted = 0
    decode_time = 0.0
    data_time = 0.0
    forward_time = 0.0
    batch_tensors = []
    try:
        with torch.inference_mode():
            consumed = 0
            while consumed < n:
                batch_end = min(consumed + batch_size, n)
                # Keep decode futures for at least `window` records in flight.
                if executor is not None:
                    submit_limit = min(n, batch_end + window)
                    while submitted < submit_limit:
                        future_map[submitted] = executor.submit(decode, submitted)
                        submitted += 1
                gathered = []
                for index in range(consumed, batch_end):
                    if executor is not None:
                        _, image, text, elapsed = future_map.pop(index).result()
                    else:
                        _, image, text, elapsed = decode(index)
                    decode_time += elapsed
                    gathered.append((image, text))
                t_prep = time.monotonic()
                inputs = processor(
                    text=[text for _, text in gathered],
                    images=[image for image, _ in gathered],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)
                data_time += time.monotonic() - t_prep
                t_fwd = time.monotonic()
                outputs = clip(**inputs)
                z_v = torch.nn.functional.normalize(outputs.image_embeds.float(), dim=-1)
                z_s = torch.nn.functional.normalize(outputs.text_embeds.float(), dim=-1)
                batch_tensors.append(encoder(z_v, z_s).detach().cpu())
                forward_time += time.monotonic() - t_fwd
                consumed = batch_end
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    queries = torch.cat(batch_tensors, dim=0).to(torch.float32).contiguous()
    if queries.shape != (n, qc.QUERY_DIM):
        raise ValueError("worker produced shape {} for {} records".format(queries.shape, n))
    stats = {
        "decode_time_s": round(decode_time, 3),
        "prep_time_s": round(data_time, 3),
        "forward_time_s": round(forward_time, 3),
        "gpu_max_memory_gib": round(
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 3), 3
        )
        if torch.cuda.is_available() else 0.0,
    }
    return queries, stats


def _worker_main(args) -> int:
    device = args.device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("worker rank {}: cuda unavailable".format(device))
        if args.rank >= torch.cuda.device_count():
            raise RuntimeError(
                "worker rank {} needs local cuda:{} but only {} visible "
                "(CUDA_VISIBLE_DEVICES mapping error)".format(
                    args.rank, args.rank, torch.cuda.device_count()
                )
            )
    started = time.monotonic()
    all_records = _records_of(args.split_file)
    if args.limit is not None:
        all_records = all_records[: int(args.limit)]
    all_ids = _sample_ids(all_records)
    if args.world_size == 1:
        slice_ids = list(all_ids)
    else:
        slice_ids = qc.rank_sample_slice(all_ids, args.world_size, args.rank)
    positions = {value: index for index, value in enumerate(all_ids)}
    records = [all_records[positions[value]] for value in slice_ids]

    clip, processor = _load_clip(args.backbone_path, device)
    model_load_s = time.monotonic() - started
    encoder = FixedMultimodalQuery().to(device).eval()
    queries, timing = _worker_forward(
        records,
        args.image_folder,
        device,
        clip,
        processor,
        encoder,
        args.batch_size,
        args.prefetch,
        args.decode_workers,
    )
    timing["model_load_time_s"] = round(model_load_s, 3)
    timing["wall_time_s"] = round(time.monotonic() - started, 3)
    timing["samples"] = int(queries.shape[0])
    timing["rank"] = int(args.rank)
    timing["world_size"] = int(args.world_size)
    timing["physical_gpu"] = int(args.physical_gpu)
    timing["local_cuda"] = device
    partial_path = str(
        Path(args.partial) / _partial_name(args.task_index, args.split, args.rank)
    )
    qc.write_partial(
        partial_path,
        rank=args.rank,
        world_size=args.world_size,
        contract_hash=args.contract_hash,
        task_index=args.task_index,
        task_name=args.task_name,
        split=args.split,
        sample_ids=slice_ids,
        queries=queries,
        stats=timing,
    )
    print(
        "worker rank{} physical GPU{} local {}: {}.{} {} samples in {:.1f}s "
        "(forward {:.1f}s, decode {:.1f}s) -> {}".format(
            args.rank, args.physical_gpu, device, args.task_name, args.split,
            queries.shape[0], timing["wall_time_s"], timing["forward_time_s"],
            timing["decode_time_s"], partial_path,
        ),
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# Orchestrator.
# ---------------------------------------------------------------------------

def _physical_gpus_free(physical_ids: Sequence[int]) -> List[str]:
    """Foreign compute processes already bound to the requested physical GPUs."""
    my_pid = str(os.getpid())
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True, stderr=subprocess.DEVNULL,
    )
    mapping = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True, stderr=subprocess.DEVNULL,
    )
    uuid_to_index = {}
    for line in mapping.strip().splitlines():
        index, uuid_ = [part.strip() for part in line.split(",")]
        uuid_to_index[uuid_] = int(index)
    requested = [int(value) for value in physical_ids]
    busy = []
    for line in apps.strip().splitlines():
        if not line.strip():
            continue
        uuid_, pid = [part.strip() for part in line.split(",")]
        index = uuid_to_index.get(uuid_)
        if index in requested and pid != my_pid:
            busy.append("GPU{} (pid {})".format(index, pid))
    return busy


def _spawn_workers(
    args, task, split, split_file, contract_hash, tmp_dir, log_dir, run_label,
) -> List[Path]:
    """Spawn (and wait for) the missing shard workers, one per physical GPU.

    Returns the list of partial paths for every rank 0..world_size-1.
    """
    records = _records_of(split_file)
    ids = _sample_ids(records)
    gpus = args.gpus
    gpus_csv = ",".join(str(value) for value in gpus)
    python = args.python
    worker_env = dict(os.environ)
    worker_env["CUDA_VISIBLE_DEVICES"] = gpus_csv
    repo_root = str(Path(__file__).resolve().parents[2])
    worker_env["PYTHONPATH"] = repo_root + os.pathsep + worker_env.get("PYTHONPATH", "")
    jobs = []
    partials = []
    for rank in range(args.world_size):
        partial = tmp_dir / _partial_name(args.task_index, split, rank)
        partials.append(partial)
        expected = qc.rank_sample_slice(ids, args.world_size, rank)
        if not args.force and qc.partial_valid_for(str(partial), contract_hash, expected):
            print("  [{}] reuse rank{} partial {}".format(run_label, rank, partial),
                  flush=True)
            continue
        command = [
            python, "-m", "compose.eval.precompute_v7_queries",
            "--mode", "worker",
            "--task-index", str(args.task_index),
            "--task-name", args.task_name,
            "--split", split,
            "--split-file", str(split_file),
            "--image-folder", str(args.image_folder),
            "--backbone-path", str(args.backbone_path),
            "--backbone-name", str(args.backbone_name),
            "--batch-size", str(args.batch_size),
            "--prefetch", str(args.prefetch),
            "--decode-workers", str(args.decode_workers),
            "--world-size", str(args.world_size),
            "--rank", str(rank),
            "--device", "cuda:{}".format(rank),
            "--partial", str(tmp_dir),
            "--contract-hash", str(contract_hash),
            "--physical-gpu", str(gpus[rank]),
            "--out-root", str(args.out_root),
        ]
        if args.limit is not None:
            command += ["--limit", str(args.limit)]
        log_path = log_dir / ("{}.{}.rank{}.log".format(
            task["name"].lower(), split, rank))
        handle = open(log_path, "w", encoding="utf-8")
        jobs.append((command, handle, log_path))
    if jobs:
        processes = []
        for command, handle, log_path in jobs:
            proc = subprocess.Popen(
                command, env=worker_env, stdout=handle, stderr=subprocess.STDOUT
            )
            processes.append((proc, handle, log_path))
        for proc, handle, log_path in processes:
            code = proc.wait()
            handle.close()
            if code != 0:
                tail = "\n".join(
                    Path(log_path).read_text(encoding="utf-8").splitlines()[-30:]
                )
                raise RuntimeError(
                    "worker failed ({}) for {}:\n{}".format(code, run_label, tail)
                )
    return partials


def _run_split(args, task, split, out_root, log_dir) -> Dict[str, object]:
    """Compute one declared split across world_size GPUs, merge, audit, save."""
    task_index = int(task["task_index"])
    # The shared Namespace is reused sequentially over tasks/splits: bind the
    # per-task worker context here (spawn + partial naming read these).
    args.task_index = task_index
    args.task_name = str(task["name"])
    split_file = task["splits"][split]
    records = _records_of(split_file)
    ids = _sample_ids(records)
    declared_count = len(ids)
    if len(set(ids)) != declared_count:
        raise ValueError("declared split has non-unique sample ids: {}".format(split_file))
    contract_info = _make_contract(args, task, split, split_file, records, args.world_size)
    contract_hash = contract_info["contract_hash"]
    split_dir = out_root / "query_cache" / "task{}".format(task_index) / split
    tmp_dir = out_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if not args.force and qc.split_cache_valid(str(split_dir), contract_hash):
        print("  [{}] official cache present and contract-current; skip".format(
            task["name"]), flush=True)
        return {"task_index": task_index, "split": split, "skipped": True}

    run_label = "task{}.{}.{}".format(task["name"], split, args.world_size)
    started = time.monotonic()
    worker_result, gpu_util = sample_util_while(args.gpus, lambda: _spawn_workers(
        args, task, split, split_file, contract_hash, tmp_dir, log_dir, run_label))
    partials = worker_result
    worker_wall_s = time.monotonic() - started
    merge_started = time.monotonic()
    merged_ids, queries, audit = qc.merge_partials(
        [str(path) for path in partials],
        contract_hash=contract_hash,
        declared_ids=ids,
        declared_count=declared_count,
    )
    audit["sample_id_set_hash_matches_declared"] = bool(
        qc.sample_id_set_hash(merged_ids) == qc.sample_id_set_hash(ids)
    )
    audit["norm_stats"] = qc.norm_stats(queries)
    audit["ordered_id_hash_matches_declared"] = bool(
        qc.ordered_id_hash(merged_ids) == qc.ordered_id_hash(ids)
    )
    worker_stats = {}
    for rank in range(args.world_size):
        partial = tmp_dir / _partial_name(task_index, split, rank)
        if partial.is_file():
            try:
                _, _, meta = qc.load_partial(str(partial))
                worker_stats["rank{}".format(rank)] = meta["stats"]
            except Exception:
                pass
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    queries_path, metadata = qc.write_split_cache(
        str(split_dir),
        contract=contract_info["contract"],
        sample_ids=merged_ids,
        queries=queries,
        merge_audit=audit,
        worker_stats=worker_stats,
        runtime={
            "physical_gpus": list(args.gpus),
            "local_cuda_mapping": {
                "cuda:{}".format(rank): gpu
                for rank, gpu in enumerate(args.gpus)
            },
            "gpu_util_percent": {
                "GPU{}".format(gpu): util["avg_util_percent"]
                for gpu, util in gpu_util.items()
            },
            "world_size": args.world_size,
        },
        created_at=created_at,
    )
    audit_time_s = time.monotonic() - merge_started
    stats_payload = {
        "task_index": task_index,
        "task_name": task["name"],
        "split": split,
        "declared_count": declared_count,
        "unique_declared_ids": len(set(ids)),
        "saved_count": int(queries.shape[0]),
        "world_size": args.world_size,
        "physical_gpus": list(args.gpus),
        "worker_wall_time_s": round(worker_wall_s, 3),
        "merge_audit_time_s": round(audit_time_s, 3),
        "total_time_s": round(time.monotonic() - started, 3),
        "samples_per_second": round(declared_count / max(worker_wall_s, 1e-6), 2),
        "rank0_samples": int(audit.get("rank0_sample_count", 0)),
        "rank1_samples": int(audit.get("rank1_sample_count", 0)),
        "query_tensor_hash": qc.query_tensor_hash(merged_ids, queries),
        "sample_id_set_hash": qc.sample_id_set_hash(merged_ids),
        "queries_file": str(queries_path.resolve()),
        "metadata_file": str((split_dir / "metadata.json").resolve()),
        "gpu_util": gpu_util,
        "worker_stats": worker_stats,
        "skipped": False,
    }
    # Clean up the per-rank partials now that the official cache is written
    # and fully audited.
    for rank in range(args.world_size):
        partial = tmp_dir / _partial_name(task_index, split, rank)
        for suffix in ("", ".json"):
            candidate = Path(str(partial) + suffix)
            if candidate.is_file():
                candidate.unlink()
    print(
        "  [{}] merged {} samples in {:.1f}s ({:.1f} samples/s) -> {}".format(
            run_label, queries.shape[0], worker_wall_s,
            declared_count / max(worker_wall_s, 1e-6), queries_path,
        )
    )
    return stats_payload


def _write_task_center(args, task, out_root) -> Dict[str, object]:
    task_index = int(task["task_index"])
    train_dir = out_root / "query_cache" / "task{}".format(task_index) / "train"
    ids, queries = qc.read_split_cache(str(train_dir))
    split_file = task["splits"]["train"]
    records = _records_of(split_file)
    contract_info = _make_contract(args, task, "train", split_file, records, args.world_size)
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    center_path, payload = qc.write_task_center(
        str(out_root / "query_cache" / "task{}".format(task_index)),
        contract=contract_info["contract"],
        queries=queries,
        source_query_cache_hash=qc.sha256_file(str(train_dir / "queries.pt")),
        created_at=created_at,
    )
    metadata_path = train_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["task_center"] = {
        "path": str(center_path.resolve()),
        "num_queries_used_for_center": int(payload["num_queries_used_for_center"]),
        "num_declared_train_samples": int(payload["num_declared_train_samples"]),
        "center_counts_equal": bool(
            payload["num_queries_used_for_center"] == payload["num_declared_train_samples"]
        ),
        "source_query_cache_hash": payload["source_query_cache_hash"],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("  [{}] full-train task center saved -> {}".format(task["name"], center_path),
          flush=True)
    return {
        "num_queries_used_for_center": int(payload["num_queries_used_for_center"]),
        "num_declared_train_samples": int(payload["num_declared_train_samples"]),
        "center_counts_equal": bool(
            payload["num_queries_used_for_center"] == payload["num_declared_train_samples"]
        ),
        "path": str(center_path.resolve()),
    }


def _build_manifest(args, out_root: Path) -> Dict[str, object]:
    manifest = {
        "schema_version": 1,
        "kind": "v7_fixed_query_cache_manifest",
        "git_sha": git_head(),
        "git_branch": git_branch(),
        "query_dim": 1536,
        "dtype": "float32",
        "query_mode": "v7_fixed",
        "cache_root": str((out_root / "query_cache").resolve()),
        "physical_gpus": list(args.gpus),
        "local_cuda_mapping": {
            "cuda:{}".format(rank): gpu for rank, gpu in enumerate(args.gpus)
        },
        "tasks": {},
    }
    for folder in sorted((out_root / "query_cache").glob("task*"), key=lambda p: p.name):
        for split in ("train", "val", "test"):
            metadata_path = folder / split / "metadata.json"
            queries_path = folder / split / "queries.pt"
            if not metadata_path.is_file() or not queries_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            task_name = str(metadata["contract"]["task_name"])
            manifest["tasks"].setdefault(task_name, {})[split] = {
                "path": str(queries_path.resolve()),
                "metadata_path": str(metadata_path.resolve()),
                "sample_count": int(metadata["num_saved_queries"]),
                "sample_id_hash": metadata["sample_id_set_hash"],
                "query_hash": metadata["query_tensor_hash"],
                "source_dataset_sha256": metadata["contract"]["source_dataset_sha256"],
                "contract_hash": metadata["contract_hash"],
            }
    (out_root / "query_cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _orchestrate(args, formal) -> None:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log_dir = out_root / "logs"
    table = {entry["name"]: entry for entry in formal["tasks"]}
    requested = args.task_names or [entry["name"] for entry in formal["tasks"]]
    for name in requested:
        if name not in table:
            raise ValueError("unknown task {!r}; choose from {}".format(
                name, sorted(table)))
    args.gpus = [int(value) for value in args.gpus.split(",")]
    if len(args.gpus) != args.world_size or len(set(args.gpus)) != args.world_size:
        raise ValueError("--gpus must list {} distinct physical GPUs".format(args.world_size))
    busy = _physical_gpus_free(args.gpus)
    if busy and not args.allow_busy:
        raise RuntimeError(
            "refusing to start: physical GPUs busy: {}. Run again after they "
            "are free or pass --allow-busy (not recommended).".format(busy))
    summary = []
    for task in table.values():
        if task["name"] not in requested:
            continue
        for split in args.splits:
            split_file = task["splits"].get(split)
            if split_file is None:
                raise ValueError("task {} has no {} split declared".format(task["name"], split))
            if not Path(split_file).is_file():
                raise FileNotFoundError(split_file)
            print("== {}.{} (declared {}) ==".format(task["name"], split, split_file),
                  flush=True)
            stats = _run_split(args, task, split, out_root, log_dir)
            if not stats.get("skipped"):
                summary.append(stats)
    for task in table.values():
        if task["name"] not in requested:
            continue
        if "train" not in args.splits:
            continue
        train_dir = out_root / "query_cache" / "task{}".format(task["task_index"]) / "train"
        if not (train_dir / "queries.pt").is_file():
            continue
        center_stats = _write_task_center(args, task, out_root)
        summary.append({
            "task_index": task["task_index"],
            "task_name": task["name"],
            "split": "task_center",
            **center_stats,
        })
    metrics_dir = out_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "precompute_stats.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = _build_manifest(args, out_root)
    (out_root / "metrics" / "manifest_summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("precompute done: {} splits + centers; manifest -> {}".format(
        len(summary), out_root / "query_cache_manifest.json"), flush=True)


# ---------------------------------------------------------------------------
# Bounded gate: world-1 vs world-2 numerical and routing equivalence.
# ---------------------------------------------------------------------------

def _gate_phase(args, task, split, world_size, gpus, prefix):
    """Run bounded queries with one world configuration; returns artifacts.

    The gate is a verification run, so any partial from an earlier gate is
    deleted first (the orchestrator would otherwise reuse contract-valid
    shards and mask a regression).
    """
    split_file = task["splits"][split]
    gate_root = Path(args.out_root) / qc_gate_dir(prefix)
    tmp_dir = gate_root / "tmp"
    log_dir = gate_root / "logs"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    sub_args = argparse.Namespace(**vars(args))
    sub_args.gpus = gpus
    sub_args.world_size = world_size
    sub_args.task_index = int(task["task_index"])
    sub_args.task_name = task["name"]
    sub_args.split = split
    sub_args.split_file = split_file
    sub_args.image_folder = args.image_folder
    sub_args.limit = int(args.bounded)
    sub_args.contract_hash = _make_contract(
        args, task, split, split_file,
        _records_of(split_file)[: int(args.bounded)], world_size,
    )["contract_hash"]

    def spawn():
        return _spawn_workers(
            sub_args, task, split, split_file, sub_args.contract_hash,
            tmp_dir, log_dir, "gate-{}".format(prefix),
        )

    started = time.monotonic()
    partials, util = sample_util_while(gpus, spawn)
    wall_s = time.monotonic() - started
    declared_ids = _sample_ids(_records_of(split_file)[: int(args.bounded)])
    merged_ids, queries, audit = qc.merge_partials(
        [str(path) for path in partials],
        contract_hash=sub_args.contract_hash,
        declared_ids=declared_ids,
        declared_count=len(declared_ids),
    )
    return {
        "world_size": world_size,
        "gpus": list(gpus),
        "wall_time_s": round(wall_s, 3),
        "samples_per_second": round(len(declared_ids) / max(wall_s, 1e-6), 2),
        "ids": merged_ids,
        "queries": queries,
        "audit": audit,
        "gpu_util": util,
    }


def qc_gate_dir(prefix: str) -> str:
    return "_gate/{}".format(prefix)


def _gate_main(args, formal) -> None:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.task_index is None:
        raise ValueError("gate mode requires --task-index (0..5)")
    task = formal["tasks"][int(args.task_index)]
    gpus = [int(value) for value in args.gpus.split(",")]
    if len(gpus) < 2 or len(set(gpus)) != 2:
        raise ValueError("gate requires two distinct physical GPUs")
    busy = _physical_gpus_free(gpus[:2])
    if busy and not args.allow_busy:
        raise RuntimeError("gate GPUs busy: {}".format(busy))
    split = args.split
    records = _records_of(task["splits"][split])[: int(args.bounded)]
    if len(records) < 2:
        raise ValueError("bounded gate needs at least 2 samples")
    print("gate: {} {} bounded={} ({} declared first records)".format(
        task["name"], split, len(records), args.bounded), flush=True)
    single = _gate_phase(args, task, split, 1, [gpus[0]], "single")
    multi = _gate_phase(args, task, split, 2, gpus[:2], "multi")

    comparison = qc.compare_by_sample_id(
        single["ids"], single["queries"], multi["ids"], multi["queries"])
    from compose.v7.query import full_train_task_center
    center_a, coverage_a = full_train_task_center(single["queries"], single["queries"].shape[0])
    center_b, coverage_b = full_train_task_center(multi["queries"], multi["queries"].shape[0])
    center_compare = qc.compare_by_sample_id(
        ["mu"], center_a.unsqueeze(0), ["mu"], center_b.unsqueeze(0))
    route = None
    if args.pool:
        route = qc.verify_route_agreement(
            args.pool, single["ids"], single["queries"], multi["ids"], multi["queries"])
    verdict = "PASS" if comparison["cosine_min"] >= 1.0 - 1.0e-6 else "FAIL"
    report = {
        "gate_kind": "bounded_real_data_single_vs_dual",
        "schema_version": 1,
        "git_sha": git_head(),
        "git_branch": git_branch(),
        "task_name": task["name"],
        "task_index": int(task["task_index"]),
        "split": split,
        "bounded_samples": len(records),
        "source_split_file": str(task["splits"][split]),
        "single": {key: single[key] for key in ("world_size", "gpus", "wall_time_s",
                                                 "samples_per_second", "audit", "gpu_util")},
        "multi": {key: multi[key] for key in ("world_size", "gpus", "wall_time_s",
                                              "samples_per_second", "audit", "gpu_util")},
        "speedup_x": round(single["wall_time_s"] / max(multi["wall_time_s"], 1e-9), 3),
        "comparison": comparison,
        "center": {"coverage_single": coverage_a, "coverage_multi": coverage_b,
                   "comparison": center_compare},
        "route_top2": route,
        "verdict": verdict,
        "verdict_tolerance": "cosine_min >= 1 - 1e-6",
    }
    (out_root / "_gate" / "gate_result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print("GATE VERDICT: {}".format(verdict), flush=True)
    if verdict != "PASS":
        raise SystemExit(1)


# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="V7 fixed-query GPU0/GPU1 precompute orchestrator "
                    "(standalone experiment; no training code is touched)")
    parser.add_argument("--mode", choices=("precompute", "worker", "gate"), required=True)
    parser.add_argument("--config", default="configs/v7_ucit_formal.yaml")
    parser.add_argument("--out-root", required=True,
                        help="experiment root (cache lives under <root>/query_cache)")
    parser.add_argument("--gpus", default="0,1",
                        help="comma-separated *physical* GPU ids, one per worker")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--task-names", nargs="*",
                        help="subset of UCIT task names (default: all six)")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"],
                        help="split names to compute (declared train/val/test)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="worker: only the first N declared records (gate)")
    parser.add_argument("--force", action="store_true",
                        help="ignore existing contract-valid official caches/partials")
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    # worker-only
    parser.add_argument("--task-name")
    parser.add_argument("--split")
    parser.add_argument("--split-file")
    parser.add_argument("--image-folder")
    parser.add_argument("--backbone-path")
    parser.add_argument("--backbone-name")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--partial")
    parser.add_argument("--contract-hash")
    parser.add_argument("--run-id")
    # gate-only
    parser.add_argument("--bounded", type=int, default=128)
    parser.add_argument("--pool", default=None,
                        help="V7 key state for the Top-2 routing smoke (gate)")
    args = parser.parse_args(argv)
    if args.mode == "worker":
        for flag in ("task-name", "split", "split-file", "image-folder",
                     "backbone-path", "partial", "contract-hash"):
            if getattr(args, flag.replace("-", "_")) is None:
                parser.error("worker mode requires --{}".format(flag))
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    formal = _load_formal_tasks(args.config)
    # The formal config is the single source for the data/backbone contract;
    # explicit CLI values (e.g. worker spawn args) always win.
    args.image_folder = args.image_folder or formal["image_folder"]
    args.backbone_name = args.backbone_name or formal["backbone_name"]
    args.backbone_path = args.backbone_path or formal["backbone_path"]
    if args.mode == "precompute":
        if len(args.splits) != len(set(args.splits)):
            raise ValueError("duplicate split in --splits")
        _orchestrate(args, formal)
        return 0
    if args.mode == "worker":
        return _worker_main(args)
    if args.mode == "gate":
        _gate_main(args, formal)
        return 0
    raise ValueError("unknown mode")


if __name__ == "__main__":
    sys.exit(main())
