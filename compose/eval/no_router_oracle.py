"""Resumable no-router oracle evaluation for the frozen V6.2 UCIT run.

The only inference-time intervention in this module is an explicit empty,
single, or pair expert selection.  It never imports or invokes ComposeRouter.
Generation otherwise mirrors :mod:`compose.eval.eval_task`, while scoring is
delegated to the original Hyper-LLaVA UCIT evaluator modules.
"""

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images, tokenizer_image_token

from compose.data.records import answer_text, question_text
from compose.eval.eval_task import _prompt
from compose.eval.load_compose import load_compose_model
from compose.oracle.candidate_sets import CandidateSet, build_candidate_sets


PYTHON = "/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"
JAVA_BIN = "/home/zhaozhuofan/miniconda3/envs/hyper/lib/jvm/bin"
BASE_MODEL = "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
VISION_TOWER = "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
PROJECTOR_PATH = os.path.join(BASE_MODEL, "mm_projector.bin")
IMAGE_FOLDER = "/data/dataset/zhaozhuofan/UCIT/datasets"
FORMAL_COMMIT = "fb5e9083d52cc996cdc2ab5a58634a8c489c5b6b"
TASK_NAMES = ["ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR", "Flickr30k"]
TEST_FILES = [
    "/data/dataset/zhaozhuofan/UCIT/instructions/ImageNet-R/test_3000.json",
    "/data/dataset/zhaozhuofan/UCIT/instructions/ArxivQA/test_3000.json",
    "/data/dataset/zhaozhuofan/UCIT/instructions/VizWiz/test_3000.json",
    "/data/dataset/zhaozhuofan/UCIT/instructions/IconQA/test_3000.json",
    "/data/dataset/zhaozhuofan/UCIT/instructions/CLEVR/test_3000.json",
    "/data/dataset/zhaozhuofan/UCIT/instructions/Flickr30k/test_3000.json",
]
CAPTION_ANNOTATIONS = {
    2: "/data/dataset/zhaozhuofan/UCIT/instructions/VizWiz/val_coco_type_3000.json",
    5: "/data/dataset/zhaozhuofan/UCIT/instructions/Flickr30k/val_coco_type_3000.json",
}
ROUTED_FINAL = [19.83, 88.53, 54.19, 26.37, 41.03, 51.41]
SCORE_RE = re.compile(
    r"^\s*(?:Accuracy|Average)\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*%?\s*$",
    re.IGNORECASE,
)


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def _record_id(record: Dict[str, object], fallback: int) -> str:
    return str(record.get("question_id", record.get("id", fallback)))


def _bounds(length: int, count: int, index: int) -> Tuple[int, int]:
    if count <= 0 or not 0 <= index < count:
        raise ValueError("invalid chunk selection {}/{}".format(index, count))
    size = int(math.ceil(length / count))
    return min(index * size, length), min((index + 1) * size, length)


def _read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _candidate_dir(root: Path, task: int, split: str, candidate: CandidateSet) -> Path:
    return root / "fixed" / "final_pool" / "task{}".format(task) / split / "candidate_{:02d}".format(candidate.index)


def _final_checkpoint(formal_root: Path) -> Path:
    manifest = json.loads(
        (formal_root / "task5" / "snapshots" / "task5" / "manifest.json").read_text(encoding="utf-8")
    )
    checkpoint = Path(manifest["pool_checkpoint_dir"])
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if manifest.get("git_commit") != FORMAL_COMMIT:
        raise RuntimeError("formal snapshot commit mismatch")
    if manifest.get("active_expert_ids") != list(range(10)):
        raise RuntimeError("formal final pool is not exactly experts 0..9")
    return checkpoint


def _fingerprint(root: Path) -> Dict[str, dict]:
    result = {}
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = str(path.relative_to(root))
        result[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return result


def _prepare_validation(formal_root: Path, output_root: Path, task: int) -> Dict[str, str]:
    source = formal_root / "task{}".format(task) / "data" / "teacher_val.json"
    records = json.loads(source.read_text(encoding="utf-8"))
    if len(records) != 200:
        raise RuntimeError("task {} formal teacher-validation size is not 200".format(task))
    normalized = []
    for index, record in enumerate(records):
        row = copy.deepcopy(record)
        row["question_id"] = str(index)
        row["answer"] = answer_text(record)
        normalized.append(row)
    question_file = output_root / "inputs" / "validation" / "task{}.json".format(task)
    _write_json(question_file, normalized)
    if task in CAPTION_ANNOTATIONS:
        annotation = {
            "info": {},
            "licenses": [],
            "categories": [{"id": 1, "name": "captioning"}],
            "images": [{"id": index + 1} for index in range(len(normalized))],
            "annotations": [
                {
                    "id": index + 1,
                    "image_id": index + 1,
                    "category_id": 1,
                    "caption": row["answer"],
                }
                for index, row in enumerate(normalized)
            ],
        }
        annotation_file = output_root / "inputs" / "validation" / "task{}_coco.json".format(task)
        _write_json(annotation_file, annotation)
    else:
        annotation_file = question_file
    return {"questions": str(question_file), "annotation": str(annotation_file)}


def prepare(args) -> None:
    formal_root = Path(args.formal_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not (formal_root / "FORMAL_COMPLETE").is_file():
        raise RuntimeError("formal run is incomplete")
    if (formal_root / "FORMAL_RUN_COMMIT").read_text().strip() != FORMAL_COMMIT:
        raise RuntimeError("FORMAL_RUN_COMMIT mismatch")
    checkpoint = _final_checkpoint(formal_root)
    validation = [_prepare_validation(formal_root, output_root, task) for task in range(6)]
    fingerprint_path = output_root / "formal_seed42_fingerprint_before.json"
    if not fingerprint_path.is_file():
        print("Hashing frozen formal artifacts...", flush=True)
        _write_json(fingerprint_path, _fingerprint(formal_root))
    current_commit = _git_commit()
    manifest_path = output_root / "run_manifest.json"
    if manifest_path.is_file():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("oracle_code_commit") != current_commit:
            raise RuntimeError(
                "output root is locked to oracle code commit {}; current is {}"
                .format(previous_manifest.get("oracle_code_commit"), current_commit)
            )
    manifest = {
        "schema_version": 1,
        "experiment": "no_router_oracle_v62_seed42",
        "formal_root": str(formal_root),
        "formal_run_commit": FORMAL_COMMIT,
        "oracle_code_commit": current_commit,
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "seed": 42,
        "physical_gpus_allowed": [4, 5, 6, 7],
        "final_checkpoint": str(checkpoint),
        "expert_ids": list(range(10)),
        "candidate_count": 56,
        "router_enabled": False,
        "router_call_count": 0,
        "rms_policy": "load_commit_frozen_only",
        "validation_policy": "frozen formal teacher_val records 2000:2200",
        "validation": validation,
        "test_questions": TEST_FILES,
        "routed_final": dict(zip(TASK_NAMES, ROUTED_FINAL)),
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _load_bundle(checkpoint: str, device: str):
    bundle = load_compose_model(
        model_path=BASE_MODEL,
        checkpoint_dir=checkpoint,
        vision_tower=VISION_TOWER,
        projector_path=PROJECTOR_PATH,
        expert_id=None,
        device=device,
        dtype=torch.bfloat16,
        model_max_length=2048,
    )
    calibration = bundle.load_summary.get("rms_calibration")
    if not calibration:
        raise RuntimeError("frozen checkpoint has no RMS calibration")
    from compose.lora.rms import apply_kappa_calibration

    apply_kappa_calibration(bundle.model, calibration)
    bundle.load_summary["rms_calibration_applied"] = True
    return bundle


def _set_candidate(bundle, candidate: CandidateSet) -> None:
    manager = bundle.expert_pool.manager
    if candidate.expert_ids:
        manager.set_default_selection(
            candidate.expert_ids, candidate.gates, normalization=candidate.normalization
        )
    else:
        manager.clear_default_selection()


def _prepare_generation_inputs(bundle, record: dict, device: str):
    prompt = _prompt(record, bundle.model.config, "vicuna_v1")
    input_ids = tokenizer_image_token(
        prompt, bundle.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    image_path = os.path.join(IMAGE_FOLDER, str(record["image"]))
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    image_tensor = process_images([image], bundle.image_processor, bundle.model.config)
    image_tensor = image_tensor[0].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    return input_ids, image_tensor


def _generate_with_inputs(bundle, input_ids, image_tensor, max_new_tokens: int) -> str:
    with torch.inference_mode():
        output_ids = bundle.model.generate(
            input_ids=input_ids,
            images=image_tensor,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    return bundle.tokenizer.batch_decode(
        output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )[0].strip()


def fixed_worker(args) -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = args.device
    bundle = _load_bundle(args.checkpoint, device)
    all_candidates = build_candidate_sets(bundle.expert_pool.expert_ids())
    if len(all_candidates) != args.expected_candidates:
        raise RuntimeError("candidate count mismatch: {}".format(len(all_candidates)))
    requested = (
        {int(value) for value in args.candidate_indices.split(",") if value.strip()}
        if args.candidate_indices
        else None
    )
    candidates = [
        candidate for candidate in all_candidates
        if requested is None or candidate.index in requested
    ]
    if requested is not None and {candidate.index for candidate in candidates} != requested:
        raise ValueError("candidate-indices contains an unknown candidate")
    records_all = json.loads(Path(args.question_file).read_text(encoding="utf-8"))
    if args.max_samples is not None:
        records_all = records_all[: args.max_samples]
    start, end = _bounds(len(records_all), args.num_chunks, args.chunk_idx)
    records = records_all[start:end]
    root = Path(args.output_root)
    paths = {
        candidate.index: _candidate_dir(root, args.task, args.split, candidate)
        / "chunks" / "chunk_{}_{}.jsonl".format(args.num_chunks, args.chunk_idx)
        for candidate in candidates
    }
    commit = _git_commit()
    existing = {}
    for candidate in candidates:
        path = paths[candidate.index]
        rows = _read_jsonl(path)
        if len(rows) > len(records):
            raise RuntimeError("resume file has too many rows: {}".format(path))
        expected_ids = [_record_id(record, start + i) for i, record in enumerate(records[: len(rows)])]
        actual_ids = [str(row["question_id"]) for row in rows]
        if actual_ids != expected_ids:
            raise RuntimeError("resume prefix mismatch: {}".format(path))
        if any(row.get("metadata", {}).get("git_commit") != commit for row in rows):
            raise RuntimeError("resume commit mismatch: {}".format(path))
        existing[candidate.index] = len(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
    handles = {index: paths[index].open("a", encoding="utf-8") for index in paths}
    started = time.time()
    try:
        for local_index, record in enumerate(tqdm(records, desc="fixed combinations")):
            pending = [candidate for candidate in candidates if existing[candidate.index] <= local_index]
            if not pending:
                continue
            input_ids, image_tensor = _prepare_generation_inputs(bundle, record, device)
            for candidate in pending:
                if existing[candidate.index] != local_index:
                    raise RuntimeError("non-contiguous candidate resume state")
                _set_candidate(bundle, candidate)
                text = _generate_with_inputs(
                    bundle, input_ids, image_tensor, args.max_new_tokens
                )
                row = {
                    "question_id": _record_id(record, start + local_index),
                    "prompt": question_text(record),
                    "text": text,
                    "model_id": "compose-no-router-oracle",
                    "metadata": {
                        "checkpoint": args.checkpoint,
                        "git_commit": commit,
                        "selection": {
                            "expert_ids": list(candidate.expert_ids),
                            "gates": list(candidate.gates),
                            "normalization": candidate.normalization,
                            "selection_source": "explicit_no_router_fixed",
                            "router_called": False,
                        },
                    },
                }
                handle = handles[candidate.index]
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                existing[candidate.index] += 1
    finally:
        for handle in handles.values():
            handle.close()
    if any(count != len(records) for count in existing.values()):
        raise RuntimeError("worker did not complete every candidate")
    summary = {
        "task": args.task,
        "split": args.split,
        "chunk_idx": args.chunk_idx,
        "num_chunks": args.num_chunks,
        "record_start": start,
        "record_end": end,
        "records": len(records),
        "candidate_count": len(candidates),
        "router_call_count": 0,
        "checkpoint": args.checkpoint,
        "git_commit": commit,
        "duration_seconds": time.time() - started,
    }
    _write_json(root / "workers" / "fixed_{}_task{}_chunk{}.json".format(args.split, args.task, args.chunk_idx), summary)
    print(json.dumps(summary, sort_keys=True))


def _annotation_for(output_root: Path, task: int, split: str) -> Path:
    if split == "val":
        suffix = "_coco" if task in CAPTION_ANNOTATIONS else ""
        return output_root / "inputs" / "validation" / "task{}{}.json".format(task, suffix)
    return Path(CAPTION_ANNOTATIONS.get(task, TEST_FILES[task]))


def _score(task: int, annotation: Path, answers: Path, score_dir: Path) -> dict:
    score_dir.mkdir(parents=True, exist_ok=True)
    module = "llava.eval.eval_caption" if task in CAPTION_ANNOTATIONS else "llava.eval.eval_deepseek_r1"
    metric_name = "Average" if task in CAPTION_ANNOTATIONS else "Accuracy"
    scorer_env = dict(os.environ, PYTHONPATH="/home/zhaozhuofan/Hyper-LlaVA")
    scorer_env["PATH"] = JAVA_BIN + os.pathsep + scorer_env.get("PATH", "")
    result = subprocess.run(
        [
            PYTHON,
            "-m",
            module,
            "--annotation-file",
            str(annotation),
            "--result-file",
            str(answers),
            "--output-dir",
            str(score_dir),
        ],
        text=True,
        capture_output=True,
        env=scorer_env,
    )
    if result.returncode != 0:
        raise RuntimeError("{} failed: {}".format(module, result.stderr[-4000:]))
    result_text = score_dir / "Result.text"
    value = None
    for line in result_text.read_text(encoding="utf-8").splitlines():
        match = SCORE_RE.match(line)
        if match:
            value = float(match.group(1))
            break
    if value is None or not math.isfinite(value):
        raise RuntimeError("no finite score in {}".format(result_text))
    metric = {
        "task": task,
        "task_name": TASK_NAMES[task],
        "metric": metric_name,
        "value": value,
        "score_unit": "percentage_points",
        "scorer": module,
        "annotation_file": str(annotation),
        "answers_file": str(answers),
        "answers_sha256": _sha256(answers),
        "result_text_sha256": _sha256(result_text),
    }
    _write_json(score_dir / "metric.json", metric)
    return metric


def merge_and_score(output_root: Path, task: int, split: str, question_file: Path, num_chunks: int) -> None:
    records = json.loads(question_file.read_text(encoding="utf-8"))
    candidates = build_candidate_sets(range(10))
    annotation = _annotation_for(output_root, task, split)
    for candidate in candidates:
        candidate_dir = _candidate_dir(output_root, task, split, candidate)
        answers = candidate_dir / "answers.jsonl"
        chunks = [candidate_dir / "chunks" / "chunk_{}_{}.jsonl".format(num_chunks, idx) for idx in range(num_chunks)]
        rows = []
        for index, chunk in enumerate(chunks):
            chunk_rows = _read_jsonl(chunk)
            begin, end = _bounds(len(records), num_chunks, index)
            if len(chunk_rows) != end - begin:
                raise RuntimeError("incomplete chunk {}".format(chunk))
            rows.extend(chunk_rows)
        expected_ids = [_record_id(record, index) for index, record in enumerate(records)]
        if [str(row["question_id"]) for row in rows] != expected_ids:
            raise RuntimeError("merged record order mismatch for candidate {}".format(candidate.index))
        with answers.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        metric_file = candidate_dir / "score" / "metric.json"
        if metric_file.is_file():
            previous = json.loads(metric_file.read_text(encoding="utf-8"))
            if previous.get("answers_sha256") == _sha256(answers):
                continue
        metric = _score(task, annotation, answers, candidate_dir / "score")
        metric["candidate"] = candidate.to_dict()
        _write_json(metric_file, metric)


def _validate_gpus(value: str) -> List[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus or any(item not in {"4", "5", "6", "7"} for item in gpus):
        raise ValueError("only physical GPUs 4,5,6,7 are permitted")
    if len(set(gpus)) != len(gpus):
        raise ValueError("duplicate GPU ids")
    return gpus


def run_fixed(args) -> None:
    output_root = Path(args.output_root).resolve()
    formal_root = Path(args.formal_root).resolve()
    checkpoint = _final_checkpoint(formal_root)
    gpus = _validate_gpus(args.gpus)
    question_file = (
        output_root / "inputs" / "validation" / "task{}.json".format(args.task)
        if args.split == "val"
        else Path(TEST_FILES[args.task])
    )
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for index, gpu in enumerate(gpus):
        log_path = logs / "fixed_{}_task{}_chunk{}.log".format(args.split, args.task, index)
        handle = log_path.open("a", encoding="utf-8")
        command = [
            PYTHON,
            "-m",
            "compose.eval.no_router_oracle",
            "fixed-worker",
            "--checkpoint",
            str(checkpoint),
            "--question-file",
            str(question_file),
            "--output-root",
            str(output_root),
            "--task",
            str(args.task),
            "--split",
            args.split,
            "--num-chunks",
            str(len(gpus)),
            "--chunk-idx",
            str(index),
            "--device",
            "cuda:0",
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--expected-candidates",
            "56",
        ]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        processes.append((index, gpu, handle, subprocess.Popen(command, env=environment, stdout=handle, stderr=subprocess.STDOUT)))
    failures = []
    for index, gpu, handle, process in processes:
        code = process.wait()
        handle.close()
        if code:
            failures.append("chunk {} gpu {} exit {}".format(index, gpu, code))
    if failures:
        raise RuntimeError("; ".join(failures))
    merge_and_score(output_root, args.task, args.split, question_file, len(gpus))
    marker = output_root / "markers" / "fixed_{}_task{}.done".format(args.split, args.task)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(_git_commit() + "\n", encoding="utf-8")
    print("fixed {} task {} complete".format(args.split, args.task))


def sample_nll_worker(args) -> None:
    from compose.oracle.evaluator import _candidate_nll, _collate, _prepare_multimodal_batch
    from compose.oracle.metrics import compute_oracle_record, summarize_oracle_records

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    bundle = _load_bundle(args.checkpoint, args.device)
    candidates = build_candidate_sets(bundle.expert_pool.expert_ids())
    records_all = json.loads(Path(args.question_file).read_text(encoding="utf-8"))
    start, end = _bounds(len(records_all), args.num_chunks, args.chunk_idx)
    records = records_all[start:end]
    output = Path(args.output_file)
    previous = _read_jsonl(output)
    if len(previous) > len(records):
        raise RuntimeError("sample NLL resume file too long")
    commit = _git_commit()
    expected = [_record_id(record, start + index) for index, record in enumerate(records[: len(previous)])]
    if [str(row["sample_id"]) for row in previous] != expected:
        raise RuntimeError("sample NLL resume prefix mismatch")
    if any(row.get("model_commit") != commit for row in previous):
        raise RuntimeError("sample NLL resume commit mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with output.open("a", encoding="utf-8") as handle:
        for offset in tqdm(range(len(previous), len(records), args.batch_size), desc="sample oracle NLL"):
            batch_records = records[offset : offset + args.batch_size]
            raw = _collate(batch_records, bundle, IMAGE_FOLDER, args.device)
            batch = _prepare_multimodal_batch(bundle, raw)
            losses_by_set = []
            token_counts = None
            for candidate in candidates:
                losses, counts = _candidate_nll(bundle, candidate, batch)
                losses_by_set.append(losses)
                if token_counts is None:
                    token_counts = counts
                elif not torch.equal(token_counts, counts):
                    raise AssertionError("target token count changed across candidates")
            matrix = torch.stack(losses_by_set, dim=1)
            for row_index, record in enumerate(batch_records):
                losses = [float(value) for value in matrix[row_index].tolist()]
                metrics = compute_oracle_record(losses, candidates)
                row = {
                    "sample_id": _record_id(record, start + offset + row_index),
                    "task": args.task,
                    "candidate_expert_ids": [list(value.expert_ids) for value in candidates],
                    "set_ids": [value.index for value in candidates],
                    "set_gates": [list(value.gates) for value in candidates],
                    "set_normalization": [value.normalization for value in candidates],
                    "set_nll": losses,
                    "target_token_count": int(token_counts[row_index]),
                    "checkpoint_ids": bundle.expert_pool.expert_ids(),
                    "model_commit": commit,
                    "router_called": False,
                    "rms_calibration_applied": True,
                    **metrics,
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
    rows = _read_jsonl(output)
    summary = {
        **summarize_oracle_records(rows),
        "task": args.task,
        "chunk_idx": args.chunk_idx,
        "records": len(rows),
        "candidate_count": len(candidates),
        "router_call_count": 0,
        "rms_calibration_applied": True,
        "duration_seconds": time.time() - started,
    }
    _write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, sort_keys=True))


def sample_generate_worker(args) -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    bundle = _load_bundle(args.checkpoint, args.device)
    candidates = build_candidate_sets(bundle.expert_pool.expert_ids())
    records_all = json.loads(Path(args.question_file).read_text(encoding="utf-8"))
    oracle_all = _read_jsonl(Path(args.selection_file))
    if len(oracle_all) != len(records_all):
        raise RuntimeError("sample selection count mismatch")
    start, end = _bounds(len(records_all), args.num_chunks, args.chunk_idx)
    records = records_all[start:end]
    oracle = oracle_all[start:end]
    output = Path(args.output_file)
    previous = _read_jsonl(output)
    commit = _git_commit()
    expected = [_record_id(record, start + index) for index, record in enumerate(records[: len(previous)])]
    if [str(row["question_id"]) for row in previous] != expected:
        raise RuntimeError("sample generation resume prefix mismatch")
    if any(row.get("metadata", {}).get("git_commit") != commit for row in previous):
        raise RuntimeError("sample generation resume commit mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for index in tqdm(range(len(previous), len(records)), desc="sample oracle generation"):
            selection_row = oracle[index]
            candidate = candidates[int(selection_row["best_overall_index"])]
            if list(candidate.expert_ids) != selection_row["candidate_expert_ids"][candidate.index]:
                raise RuntimeError("sample oracle candidate mapping mismatch")
            _set_candidate(bundle, candidate)
            input_ids, image_tensor = _prepare_generation_inputs(
                bundle, records[index], args.device
            )
            text = _generate_with_inputs(
                bundle, input_ids, image_tensor, args.max_new_tokens
            )
            row = {
                "question_id": _record_id(records[index], start + index),
                "prompt": question_text(records[index]),
                "text": text,
                "model_id": "compose-sample-oracle-upper-bound",
                "metadata": {
                    "checkpoint": args.checkpoint,
                    "git_commit": commit,
                    "selection": {
                        "expert_ids": list(candidate.expert_ids),
                        "gates": list(candidate.gates),
                        "normalization": candidate.normalization,
                        "selection_source": "target_answer_teacher_forcing_oracle",
                        "uses_target_answer": True,
                        "router_called": False,
                    },
                    "oracle_teacher_forcing_loss": selection_row["set_nll"][candidate.index],
                    "best_single_loss": selection_row["best_single_loss"],
                    "best_pair_loss": selection_row["best_pair_loss"],
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
    _write_json(
        output.with_suffix(".summary.json"),
        {"task": args.task, "chunk_idx": args.chunk_idx, "records": len(records), "router_call_count": 0},
    )


def _run_parallel(command_builder, gpus: List[str], logs: List[Path]) -> None:
    processes = []
    for index, (gpu, log) in enumerate(zip(gpus, logs)):
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command_builder(index),
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((index, gpu, handle, process))
    failures = []
    for index, gpu, handle, process in processes:
        code = process.wait()
        handle.close()
        if code:
            failures.append("chunk {} gpu {} exit {}".format(index, gpu, code))
    if failures:
        raise RuntimeError("; ".join(failures))


def run_sample(args) -> None:
    output_root = Path(args.output_root).resolve()
    formal_root = Path(args.formal_root).resolve()
    checkpoint = _final_checkpoint(formal_root)
    gpus = _validate_gpus(args.gpus)
    question_file = Path(TEST_FILES[args.task])
    sample_dir = output_root / "sample_oracle" / "task{}".format(args.task)
    chunks = [sample_dir / "nll_chunks" / "chunk_{}_{}.jsonl".format(len(gpus), index) for index in range(len(gpus))]
    logs = [output_root / "logs" / "sample_nll_task{}_chunk{}.log".format(args.task, index) for index in range(len(gpus))]

    def nll_command(index):
        return [
            PYTHON, "-m", "compose.eval.no_router_oracle", "sample-nll-worker",
            "--checkpoint", str(checkpoint), "--question-file", str(question_file),
            "--output-file", str(chunks[index]), "--task", str(args.task),
            "--num-chunks", str(len(gpus)), "--chunk-idx", str(index),
            "--batch-size", str(args.batch_size), "--device", "cuda:0",
        ]

    _run_parallel(nll_command, gpus, logs)
    records = json.loads(question_file.read_text(encoding="utf-8"))
    merged = sample_dir / "nll.jsonl"
    all_rows = []
    for index, chunk in enumerate(chunks):
        rows = _read_jsonl(chunk)
        begin, end = _bounds(len(records), len(gpus), index)
        if len(rows) != end - begin:
            raise RuntimeError("incomplete NLL chunk {}".format(chunk))
        all_rows.extend(rows)
    if [str(row["sample_id"]) for row in all_rows] != [
        _record_id(record, index) for index, record in enumerate(records)
    ]:
        raise RuntimeError("merged NLL sample order mismatch")
    with merged.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    answer_chunks = [sample_dir / "generation_chunks" / "chunk_{}_{}.jsonl".format(len(gpus), index) for index in range(len(gpus))]
    generation_logs = [output_root / "logs" / "sample_generate_task{}_chunk{}.log".format(args.task, index) for index in range(len(gpus))]

    def generation_command(index):
        return [
            PYTHON, "-m", "compose.eval.no_router_oracle", "sample-generate-worker",
            "--checkpoint", str(checkpoint), "--question-file", str(question_file),
            "--selection-file", str(merged), "--output-file", str(answer_chunks[index]),
            "--task", str(args.task), "--num-chunks", str(len(gpus)),
            "--chunk-idx", str(index), "--device", "cuda:0",
            "--max-new-tokens", str(args.max_new_tokens),
        ]

    _run_parallel(generation_command, gpus, generation_logs)
    answers = sample_dir / "answers.jsonl"
    with answers.open("w", encoding="utf-8") as handle:
        for index, chunk in enumerate(answer_chunks):
            rows = _read_jsonl(chunk)
            begin, end = _bounds(len(records), len(gpus), index)
            if len(rows) != end - begin:
                raise RuntimeError("incomplete sample generation chunk")
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metric = _score(args.task, _annotation_for(output_root, args.task, "test"), answers, sample_dir / "score")
    metric["warning"] = "ORACLE ONLY / TARGET-ANSWER SELECTION / NOT A VALID TEST-TIME METHOD"
    _write_json(sample_dir / "score" / "metric.json", metric)
    marker = output_root / "markers" / "sample_task{}.done".format(args.task)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(_git_commit() + "\n", encoding="utf-8")
    print("sample oracle task {} complete".format(args.task))


def _common_worker_arguments(parser) -> None:
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--task", type=int, required=True, choices=range(6))
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--chunk-idx", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("prepare")
    command.add_argument("--formal-root", required=True)
    command.add_argument("--output-root", required=True)
    command.set_defaults(function=prepare)

    command = subparsers.add_parser("fixed-worker")
    _common_worker_arguments(command)
    command.add_argument("--output-root", required=True)
    command.add_argument("--split", choices=("val", "test"), required=True)
    command.add_argument("--max-new-tokens", type=int, default=128)
    command.add_argument("--expected-candidates", type=int, default=56)
    command.add_argument("--candidate-indices", default=None)
    command.add_argument("--max-samples", type=int, default=None)
    command.set_defaults(function=fixed_worker)

    command = subparsers.add_parser("run-fixed")
    command.add_argument("--formal-root", required=True)
    command.add_argument("--output-root", required=True)
    command.add_argument("--task", type=int, required=True, choices=range(6))
    command.add_argument("--split", choices=("val", "test"), required=True)
    command.add_argument("--gpus", default="4,5,6,7")
    command.add_argument("--max-new-tokens", type=int, default=128)
    command.set_defaults(function=run_fixed)

    command = subparsers.add_parser("sample-nll-worker")
    _common_worker_arguments(command)
    command.add_argument("--output-file", required=True)
    command.add_argument("--batch-size", type=int, default=8)
    command.set_defaults(function=sample_nll_worker)

    command = subparsers.add_parser("sample-generate-worker")
    _common_worker_arguments(command)
    command.add_argument("--selection-file", required=True)
    command.add_argument("--output-file", required=True)
    command.add_argument("--max-new-tokens", type=int, default=128)
    command.set_defaults(function=sample_generate_worker)

    command = subparsers.add_parser("run-sample")
    command.add_argument("--formal-root", required=True)
    command.add_argument("--output-root", required=True)
    command.add_argument("--task", type=int, required=True, choices=range(6))
    command.add_argument("--gpus", default="4,5,6,7")
    command.add_argument("--batch-size", type=int, default=8)
    command.add_argument("--max-new-tokens", type=int, default=128)
    command.set_defaults(function=run_sample)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
