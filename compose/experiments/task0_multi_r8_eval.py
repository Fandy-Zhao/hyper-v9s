"""Task0 multi rank-8 expert experiment -- evaluation.

Pool assembly (per config), RMS calibration, fixed-composition generation
(per-expert singles, pairs, equal compositions; raw and RMS-calibrated),
per-sample teacher-forced NLL for oracle selection (Mode C), centroid
routing (Mode A), and offline mode/analysis summaries.

Modes (spec §9-§12):
  Mode A  centroid top-1 routing  -- per-sample nearest train centroid
  Mode B  equal composition       -- 1/sqrt(2)x(E0+E1) / (1/4)sum(E0..E3)
  Mode C  oracle (diagnostic only) -- best single / best pair by NLL

All generation and NLL workers run on physical GPUs 4-7 only.  Scoring is
exact-match Accuracy via llava.eval.eval_deepseek_r1 (needs JAVA_BIN).
"""

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images, tokenizer_image_token

from compose.data.records import answer_text, question_text
from compose.eval.eval_task import _prompt
from compose.eval.load_compose import load_compose_model
from compose.oracle.candidate_sets import CandidateSet

REPO = Path("/home/zhaozhuofan/Hyper-LlaVA")
PYTHON = "/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"
JAVA_BIN = "/home/zhaozhuofan/miniconda3/envs/hyper/lib/jvm/bin"
BASE_MODEL = "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
VISION_TOWER = "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
PROJECTOR_PATH = os.path.join(BASE_MODEL, "mm_projector.bin")
IMAGE_FOLDER = "/data/dataset/zhaozhuofan/UCIT/datasets"
TEST_FILE = "/data/dataset/zhaozhuofan/UCIT/instructions/ImageNet-R/test_3000.json"
SEED = 42
ALLOWED_GPUS = {"4", "5", "6", "7"}
PAIR_GATE = 1.0 / math.sqrt(2.0)
EQUAL4_GATE = 0.5

CONFIGS = {
    "single_r8": {"k": 1, "rank": 8, "alpha": 16.0},
    "two_r8": {"k": 2, "rank": 8, "alpha": 16.0},
    "four_r8": {"k": 4, "rank": 8, "alpha": 16.0},
    "rank48": {"k": 1, "rank": 48, "alpha": 96.0},
}


def read_json(path: Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_json_object_from_stdout(stdout: str) -> dict:
    """Decode the final JSON object even when dependencies log to stdout."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"(?m)^\s*\{", stdout):
        try:
            value, end = decoder.raw_decode(stdout[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not stdout[match.start() + end:].strip():
            return value
    raise json.JSONDecodeError(
        "no standalone JSON object found in subprocess stdout", stdout, 0
    )


def write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def _bounds(length: int, count: int, index: int) -> Tuple[int, int]:
    if count <= 0 or not 0 <= index < count:
        raise ValueError("invalid chunk selection {}/{}".format(index, count))
    size = int(math.ceil(length / count))
    return min(index * size, length), min((index + 1) * size, length)


def record_id(record: Dict[str, object], fallback: int) -> str:
    return str(record.get("question_id", record.get("id", fallback)))


def validate_gpus(value: str) -> List[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus or any(item not in ALLOWED_GPUS for item in gpus):
        raise ValueError("only physical GPUs 4,5,6,7 are permitted")
    if len(set(gpus)) != len(gpus):
        raise ValueError("duplicate GPU ids")
    return gpus


def build_candidates(config: str) -> List[Dict[str, object]]:
    """Explicit candidate manifest per config.

    Singles are always raw (a single-expert kappa calibrates to 1.0).
    Pairs carry the spec Mode B / oracle pair gates (1/sqrt(2), l2); the
    oracle-pair convention (formal no-router oracle) applies RMS
    calibration, so pair candidates are emitted in raw and RMS variants
    for two_r8 (Mode B raw + Mode B RMS) and RMS-only for the four_r8
    oracle pairs; the 4-expert equal composition is raw + RMS.
    """
    k = int(CONFIGS[config]["k"])
    candidates: List[Dict[str, object]] = []
    if config == "single_r8" or config == "rank48":
        candidates.append({
            "index": 0, "label": "single:0", "kind": "single",
            "expert_ids": [0], "gates": [1.0], "normalization": "none",
            "apply_rms": False,
        })
    elif config == "two_r8":
        candidates.append({
            "index": 0, "label": "single:0", "kind": "single",
            "expert_ids": [0], "gates": [1.0], "normalization": "none",
            "apply_rms": False,
        })
        candidates.append({
            "index": 1, "label": "single:1", "kind": "single",
            "expert_ids": [1], "gates": [1.0], "normalization": "none",
            "apply_rms": False,
        })
        candidates.append({
            "index": 2, "label": "pair:0+1:raw", "kind": "pair",
            "expert_ids": [0, 1], "gates": [PAIR_GATE, PAIR_GATE],
            "normalization": "l2", "apply_rms": False,
        })
        candidates.append({
            "index": 3, "label": "pair:0+1:rms", "kind": "pair",
            "expert_ids": [0, 1], "gates": [PAIR_GATE, PAIR_GATE],
            "normalization": "l2", "apply_rms": True,
        })
    elif config == "four_r8":
        for e in range(4):
            candidates.append({
                "index": e, "label": "single:{}".format(e), "kind": "single",
                "expert_ids": [e], "gates": [1.0], "normalization": "none",
                "apply_rms": False,
            })
        index = 4
        for pair in itertools.combinations(range(4), 2):
            candidates.append({
                "index": index, "label": "pair:{}+{}:rms".format(pair[0], pair[1]),
                "kind": "pair", "expert_ids": list(pair),
                "gates": [PAIR_GATE, PAIR_GATE], "normalization": "l2",
                "apply_rms": True,
            })
            index += 1
        candidates.append({
            "index": index, "label": "equal4:raw", "kind": "equal",
            "expert_ids": [0, 1, 2, 3], "gates": [EQUAL4_GATE] * 4,
            "normalization": "l2", "apply_rms": False,
        })
        candidates.append({
            "index": index + 1, "label": "equal4:rms", "kind": "equal",
            "expert_ids": [0, 1, 2, 3], "gates": [EQUAL4_GATE] * 4,
            "normalization": "l2", "apply_rms": True,
        })
    else:
        raise ValueError("unknown config {}".format(config))
    return candidates


def _candidate_set(candidate: Dict[str, object]) -> CandidateSet:
    return CandidateSet(
        index=int(candidate["index"]),
        expert_ids=tuple(int(value) for value in candidate["expert_ids"]),
        gates=tuple(float(value) for value in candidate["gates"]),
        normalization=str(candidate["normalization"]),
    )


# --------------------------------------------------------------------------
# test centroid assignments (Mode A routing map)
# --------------------------------------------------------------------------

def assign(root_arg: str) -> None:
    root = Path(root_arg)
    test_features = read_json(root / "query_cache" / "test_features.json")
    for k in (2, 4):
        formation = read_json(root / "clustering" / "k{}".format(k) / "formation.json")
        centroids = torch.tensor(
            [expert["centroid"] for expert in formation["formed_experts"]],
            dtype=torch.float32,
        )
        assignments = {}
        for sample_id, record in test_features["records"].items():
            query = torch.tensor(record["query"], dtype=torch.float32)
            similarity = torch.nn.functional.cosine_similarity(
                query.unsqueeze(0), centroids, dim=-1
            )
            assignments[sample_id] = int(similarity.argmax().item())
        write_json(
            root / "clustering" / "k{}".format(k) / "test_assignments.json",
            {"schema_version": 1, "k": k, "assignments": assignments,
             "samples": len(assignments), "git_commit": git_commit()},
        )
        counts = {}
        for cluster in assignments.values():
            counts[cluster] = counts.get(cluster, 0) + 1
        print("test assignments k={}: {}".format(k, counts), flush=True)
    print("assign complete", flush=True)


# --------------------------------------------------------------------------
# pool assembly + RMS calibration
# --------------------------------------------------------------------------

def assemble(root_arg: str) -> None:
    root = Path(root_arg)
    manifest = read_json(root / "experiment_manifest.json")
    for config, spec in CONFIGS.items():
        k = int(spec["k"])
        pool_dir = root / "pools" / config
        assembly_meta = []
        for expert_id in range(k):
            state_dict = (
                root / "checkpoints" / config / "expert_{}".format(expert_id)
                / "expert_{:04d}.pt".format(expert_id)
            )
            if not state_dict.is_file():
                raise FileNotFoundError("missing trained expert: {}".format(state_dict))
            command = [
                PYTHON, "-m", "compose.eval.assemble_expert",
                "--model-path", BASE_MODEL,
                "--vision-tower", VISION_TOWER,
                "--projector-path", PROJECTOR_PATH,
                "--expert-state-dict", str(state_dict),
                "--expert-id", str(expert_id),
                "--rank", str(int(spec["rank"])),
                "--alpha", str(float(spec["alpha"])),
                "--output-dir", str(pool_dir),
            ]
            if expert_id > 0:
                command += ["--old-expert-checkpoint", str(pool_dir)]
            print("assemble {} expert {}: {}".format(config, expert_id, " ".join(command[4:])), flush=True)
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("assemble failed for {} expert {}: {}".format(
                    config, expert_id, result.stderr[-4000:]))
            assembly_meta.append(parse_json_object_from_stdout(result.stdout))
        bin_path = pool_dir / "compose_experts.bin"
        write_json(root / "pools" / "{}_assembly.json".format(config), {
            "config": config,
            "experts": assembly_meta,
            "bin_sha256": sha256_file(bin_path),
            "expert_ids": list(range(k)),
        })
        write_json(root / "evaluations" / "candidates_{}.json".format(config),
                   {"config": config, "candidates": build_candidates(config)})
        print("assemble {} complete ({} experts, bin sha256 {:.12s})".format(
            config, k, sha256_file(bin_path)), flush=True)
    print("assemble complete", flush=True)


def rms(root_arg: str, gpus: str, batch_size: int, max_samples: int) -> None:
    root = Path(root_arg)
    gpu_list = validate_gpus(gpus)
    question_file = root / "boundary" / "data" / "teacher_val.json"
    for config in CONFIGS:
        pool_dir = root / "pools" / config
        k = int(CONFIGS[config]["k"])
        if k == 1:
            continue  # single-expert kappa is vacuous (calibrates to 1.0)
        bin_sha = sha256_file(pool_dir / "compose_experts.bin")
        output_dir = root / "evaluations" / "rms" / config
        marker = output_dir / "complete.txt"
        if marker.is_file() and read_json(marker).get("bin_sha256") == bin_sha:
            print("rms {} already complete".format(config), flush=True)
            continue
        # RMS computes per-layer output deltas with fp64 moments; it must
        # never share a GPU with training/generation workers.
        gpu = gpu_list[len([name for name in CONFIGS if name < config]) % len(gpu_list)]
        command = [
            PYTHON, "-m", "compose.eval.rms_stats",
            "--model-path", BASE_MODEL,
            "--vision-tower", VISION_TOWER,
            "--projector-path", PROJECTOR_PATH,
            "--checkpoint-dir", str(pool_dir),
            "--question-file", str(question_file),
            "--image-folder", IMAGE_FOLDER,
            "--checkpoint-hash", bin_sha,
            "--dataset-manifest-hash", "",
            "--composition-config-hash", "",
            "--output-dir", str(output_dir),
            "--device", "cuda:0",
            "--batch-size", str(batch_size),
            "--new-expert-ids", ",".join(str(e) for e in range(k)),
        ]
        if max_samples:
            command += ["--max-samples", str(max_samples)]
        print("rms {} on GPU {}: {}".format(config, gpu, " ".join(command[4:])), flush=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        started = time.time()
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("rms failed for {}: {}".format(config, result.stderr[-4000:]))
        write_json(marker, {
            "config": config, "bin_sha256": bin_sha,
            "duration_seconds": time.time() - started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        print("rms {} complete ({:.0f}s)".format(config, time.time() - started), flush=True)
    print("rms complete", flush=True)


# --------------------------------------------------------------------------
# fixed-composition generation (per candidate)
# --------------------------------------------------------------------------

def load_bundle(checkpoint: str, device: str, apply_rms: bool):
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
    if apply_rms:
        calibration = bundle.load_summary.get("rms_calibration")
        if not calibration:
            raise RuntimeError("checkpoint has no RMS calibration: {}".format(checkpoint))
        from compose.lora.rms import apply_kappa_calibration
        apply_kappa_calibration(bundle.model, calibration)
        bundle.load_summary["rms_calibration_applied"] = True
    else:
        bundle.load_summary["rms_calibration_applied"] = False
    return bundle


def set_candidate(bundle, candidate: Dict[str, object]) -> None:
    manager = bundle.expert_pool.manager
    ids = [int(value) for value in candidate["expert_ids"]]
    if ids:
        manager.set_default_selection(
            ids,
            [float(value) for value in candidate["gates"]],
            normalization=str(candidate["normalization"]),
        )
    else:
        manager.clear_default_selection()


def prepare_generation_inputs(bundle, record: dict, device: str):
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


def generate_with_inputs(bundle, input_ids, image_tensor, max_new_tokens: int) -> str:
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
        output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip()


def gen_worker(args) -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    candidate = read_json(Path(args.candidate_file))
    bundle = load_bundle(args.checkpoint, args.device, bool(candidate["apply_rms"]))
    records_all = json.loads(Path(args.question_file).read_text(encoding="utf-8"))
    if args.max_samples:
        records_all = records_all[: args.max_samples]
    start, end = _bounds(len(records_all), args.num_chunks, args.chunk_idx)
    records = records_all[start:end]
    chunk_path = (
        Path(args.output_root) / "evaluations" / "answers" / args.config
        / candidate["label"] / "chunks" / "chunk_{}_{}.jsonl".format(args.num_chunks, args.chunk_idx)
    )
    commit = git_commit()
    rows = read_jsonl(chunk_path)
    expected_ids = [record_id(record, start + i) for i, record in enumerate(records[: len(rows)])]
    if [str(row["question_id"]) for row in rows] != expected_ids:
        raise RuntimeError("resume prefix mismatch: {}".format(chunk_path))
    if any(row.get("metadata", {}).get("git_commit") != commit for row in rows):
        raise RuntimeError("resume commit mismatch: {}".format(chunk_path))
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with chunk_path.open("a", encoding="utf-8") as handle:
        for local_index, record in enumerate(tqdm(records, desc="{} {}".format(args.config, candidate["label"]))):
            if len(rows) > local_index:
                continue
            input_ids, image_tensor = prepare_generation_inputs(bundle, record, args.device)
            set_candidate(bundle, candidate)
            text = generate_with_inputs(bundle, input_ids, image_tensor, args.max_new_tokens)
            row = {
                "question_id": record_id(record, start + local_index),
                "prompt": question_text(record),
                "text": text,
                "model_id": "task0-multi-r8",
                "metadata": {
                    "checkpoint": args.checkpoint,
                    "git_commit": commit,
                    "selection": {
                        "config": args.config,
                        "label": candidate["label"],
                        "expert_ids": list(candidate["expert_ids"]),
                        "gates": list(candidate["gates"]),
                        "normalization": candidate["normalization"],
                        "apply_rms": bool(candidate["apply_rms"]),
                        "selection_source": "explicit_fixed",
                        "router_called": False,
                    },
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
    rows = read_jsonl(chunk_path)
    if len(rows) != len(records):
        raise RuntimeError("worker did not complete every record: {}/{}".format(len(rows), len(records)))
    write_json(chunk_path.with_suffix(".summary.json"), {
        "config": args.config, "label": candidate["label"],
        "chunk_idx": args.chunk_idx, "num_chunks": args.num_chunks,
        "records": len(records), "duration_seconds": time.time() - started,
        "git_commit": commit,
    })


def _score(annotation: Path, answers: Path, score_dir: Path) -> dict:
    score_dir.mkdir(parents=True, exist_ok=True)
    scorer_env = dict(os.environ, PYTHONPATH=str(REPO))
    scorer_env["PATH"] = JAVA_BIN + os.pathsep + scorer_env.get("PATH", "")
    result = subprocess.run(
        [
            PYTHON, "-m", "llava.eval.eval_deepseek_r1",
            "--annotation-file", str(annotation),
            "--result-file", str(answers),
            "--output-dir", str(score_dir),
        ],
        text=True, capture_output=True, env=scorer_env,
    )
    if result.returncode != 0:
        raise RuntimeError("scorer failed: {}".format(result.stderr[-4000:]))
    result_text = score_dir / "Result.text"
    value = None
    for line in result_text.read_text(encoding="utf-8").splitlines():
        match = _score_re().match(line)
        if match:
            value = float(match.group(1))
            break
    if value is None or not math.isfinite(value):
        raise RuntimeError("no finite score in {}".format(result_text))
    metric = {
        "metric": "Accuracy", "value": value, "score_unit": "percentage_points",
        "scorer": "llava.eval.eval_deepseek_r1",
        "annotation_file": str(annotation), "answers_file": str(answers),
        "answers_sha256": sha256_file(answers),
        "result_text_sha256": sha256_file(result_text),
    }
    write_json(score_dir / "metric.json", metric)
    return metric


def _score_re():
    import re
    return re.compile(r"^\s*Accuracy\s*:\s*([0-9.]+)")


def merge_and_score(root: Path, config: str, label: str, num_chunks: int,
                    question_file: Path, max_samples: int) -> dict:
    records = json.loads(question_file.read_text(encoding="utf-8"))
    if max_samples:
        records = records[: max_samples]
    candidate_dir = root / "evaluations" / "answers" / config / label
    answers = candidate_dir / "answers.jsonl"
    chunks = [candidate_dir / "chunks" / "chunk_{}_{}.jsonl".format(num_chunks, idx)
              for idx in range(num_chunks)]
    rows = []
    for index, chunk in enumerate(chunks):
        chunk_rows = read_jsonl(chunk)
        begin, end = _bounds(len(records), num_chunks, index)
        if len(chunk_rows) != end - begin:
            raise RuntimeError("incomplete chunk {} for {}/{}".format(chunk, config, label))
        rows.extend(chunk_rows)
    expected_ids = [record_id(record, index) for index, record in enumerate(records)]
    if [str(row["question_id"]) for row in rows] != expected_ids:
        raise RuntimeError("merged record order mismatch for {}/{}".format(config, label))
    write_jsonl(answers, rows)
    metric_file = candidate_dir / "score" / "metric.json"
    if metric_file.is_file():
        previous = read_json(metric_file)
        if previous.get("answers_sha256") == sha256_file(answers):
            return previous
    metric = _score(question_file, answers, candidate_dir / "score")
    metric["config"] = config
    metric["label"] = label
    write_json(metric_file, metric)
    return metric


def gen(args) -> None:
    root = Path(args.root)
    gpus = validate_gpus(args.gpus)
    question_file = Path(TEST_FILE if not args.question_file else args.question_file)
    passes = []
    if args.config and args.label:
        passes.append((args.config, args.label))
    else:
        for config in CONFIGS:
            for candidate in build_candidates(config):
                passes.append((config, str(candidate["label"])))
    if args.base:
        passes.append(("base", "base"))
        write_json(root / "evaluations" / "candidates_base.json", {"candidates": [{
            "index": 0, "label": "base", "kind": "base",
            "expert_ids": [], "gates": [], "normalization": "none",
            "apply_rms": False,
        }]})
    for config, label in passes:
        candidate_dir = root / "evaluations" / "answers" / config / label
        metric_file = candidate_dir / "score" / "metric.json"
        if metric_file.is_file():
            print("skip {}/{} (already scored)".format(config, label), flush=True)
            continue
        checkpoint = str(root / "pools" / ("single_r8" if config == "base" else config))
        candidate_file = str(root / "evaluations" / ("candidates_base.json" if config == "base"
                                                     else "candidates_{}.json".format(config)))
        num_chunks = len(gpus)
        processes = []
        for index, gpu in enumerate(gpus):
            command = [
                PYTHON, "-m", "compose.experiments.task0_multi_r8_eval", "gen-worker",
                "--checkpoint", checkpoint,
                "--candidate-file", candidate_file,
                "--question-file", str(question_file),
                "--output-root", str(root),
                "--config", config,
                "--num-chunks", str(num_chunks),
                "--chunk-idx", str(index),
                "--device", "cuda:0",
                "--max-new-tokens", str(args.max_new_tokens),
            ]
            if args.max_samples:
                command += ["--max-samples", str(args.max_samples)]
            log_path = root / "logs" / "gen_{}_{}_chunk{}.log".format(config, label.replace(":", "_"), index)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            processes.append((gpu, handle, subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)))
        failures = []
        for gpu, handle, process in processes:
            code = process.wait()
            handle.close()
            if code:
                failures.append("chunk on gpu {} exit {}".format(gpu, code))
        if failures:
            raise RuntimeError("gen failed for {}/{}: {}".format(config, label, "; ".join(failures)))
        metric = merge_and_score(root, config, label, num_chunks, question_file, args.max_samples)
        print("{} {} accuracy: {:.2f}".format(config, label, metric["value"]), flush=True)
    print("gen complete", flush=True)


# --------------------------------------------------------------------------
# per-sample teacher-forced NLL (Mode C oracle selection + cluster analysis)
# --------------------------------------------------------------------------

def nll_worker(args) -> None:
    from compose.oracle.evaluator import _candidate_nll, _collate, _prepare_multimodal_batch

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    candidate_defs = [
        candidate for candidate in read_json(Path(args.candidates_file))["candidates"]
        if candidate["kind"] in ("single", "pair")
        and not (candidate["kind"] == "pair" and not candidate["apply_rms"])
    ]
    # NLL always runs the RMS-calibrated bundle (formal oracle convention);
    # a "raw" pair candidate would duplicate the RMS pair exactly, so it is
    # excluded here.  Singles are unaffected by kappa (calibrates to 1.0).
    candidates = [_candidate_set(candidate) for candidate in candidate_defs]
    labels = [str(candidate["label"]) for candidate in candidate_defs]
    bundle = load_bundle(args.checkpoint, args.device, apply_rms=True)
    records_all = json.loads(Path(args.question_file).read_text(encoding="utf-8"))
    if args.max_samples:
        records_all = records_all[: args.max_samples]
    start, end = _bounds(len(records_all), args.num_chunks, args.chunk_idx)
    records = records_all[start:end]
    output = Path(args.output_file)
    rows = read_jsonl(output)
    if len(rows) > len(records):
        raise RuntimeError("NLL resume file too long")
    expected = [record_id(record, start + index) for index, record in enumerate(records[: len(rows)])]
    if [str(row["sample_id"]) for row in rows] != expected:
        raise RuntimeError("NLL resume prefix mismatch")
    commit = git_commit()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with output.open("a", encoding="utf-8") as handle:
        for offset in tqdm(range(len(rows), len(records), args.batch_size), desc="NLL"):
            batch_records = records[offset: offset + args.batch_size]
            raw = _collate(batch_records, bundle, IMAGE_FOLDER, args.device)
            batch = _prepare_multimodal_batch(bundle, raw)
            losses_by_candidate = []
            token_counts = None
            for candidate in candidates:
                losses, counts = _candidate_nll(bundle, candidate, batch)
                losses_by_candidate.append(losses)
                if token_counts is None:
                    token_counts = counts
                elif not torch.equal(token_counts, counts):
                    raise AssertionError("target token count changed across candidates")
            matrix = torch.stack(losses_by_candidate, dim=1)
            for row_index, record in enumerate(batch_records):
                losses = [float(value) for value in matrix[row_index].tolist()]
                singles = [i for i, c in enumerate(candidates) if len(c.expert_ids) == 1]
                pairs = [i for i, c in enumerate(candidates) if len(c.expert_ids) == 2]
                best_single = min(singles, key=lambda i: (losses[i], i))
                best_pair = min(pairs, key=lambda i: (losses[i], i)) if pairs else -1
                best_overall = min(range(len(candidates)), key=lambda i: (losses[i], i))
                row = {
                    "sample_id": record_id(record, start + offset + row_index),
                    "candidate_labels": labels,
                    "candidate_expert_ids": [list(c.expert_ids) for c in candidates],
                    "candidate_gates": [list(c.gates) for c in candidates],
                    "candidate_normalization": [c.normalization for c in candidates],
                    "set_nll": losses,
                    "target_token_count": int(token_counts[row_index]),
                    "best_single_index": best_single,
                    "best_pair_index": best_pair,
                    "best_overall_index": best_overall,
                    "synergy": losses[best_single] - losses[best_pair],
                    "pair_oracle": len(candidates[best_overall].expert_ids) == 2,
                    "checkpoint_ids": bundle.expert_pool.expert_ids(),
                    "model_commit": commit,
                    "router_called": False,
                    "rms_calibration_applied": True,
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
    rows = read_jsonl(output)
    write_json(output.with_suffix(".summary.json"), {
        "config": args.config, "chunk_idx": args.chunk_idx,
        "records": len(rows), "duration_seconds": time.time() - started,
        "git_commit": commit,
    })
    print("nll worker {} complete ({} rows)".format(args.config, len(rows)))


def nll(args) -> None:
    root = Path(args.root)
    gpus = validate_gpus(args.gpus)
    question_file = Path(TEST_FILE if not args.question_file else args.question_file)
    records_all = json.loads(question_file.read_text(encoding="utf-8"))
    if args.max_samples:
        records_all = records_all[: args.max_samples]
    configs = [args.config] if args.config else list(CONFIGS)
    for config in configs:
        sample_dir = root / "evaluations" / "nll" / config
        chunks = [sample_dir / "chunk_{}_{}.jsonl".format(len(gpus), index) for index in range(len(gpus))]
        processes = []
        for index, gpu in enumerate(gpus):
            command = [
                PYTHON, "-m", "compose.experiments.task0_multi_r8_eval", "nll-worker",
                "--checkpoint", str(root / "pools" / config),
                "--candidates-file", str(root / "evaluations" / "candidates_{}.json".format(config)),
                "--question-file", str(question_file),
                "--output-file", str(chunks[index]),
                "--config", config,
                "--num-chunks", str(len(gpus)),
                "--chunk-idx", str(index),
                "--device", "cuda:0",
                "--batch-size", str(args.batch_size),
            ]
            if args.max_samples:
                command += ["--max-samples", str(args.max_samples)]
            log_path = root / "logs" / "nll_{}_chunk{}.log".format(config, index)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            processes.append((gpu, handle, subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)))
        failures = []
        for gpu, handle, process in processes:
            code = process.wait()
            handle.close()
            if code:
                failures.append("chunk on gpu {} exit {}".format(gpu, code))
        if failures:
            raise RuntimeError("nll failed for {}: {}".format(config, "; ".join(failures)))
        merged = sample_dir / "nll.jsonl"
        all_rows = []
        for index, chunk in enumerate(chunks):
            rows = read_jsonl(chunk)
            begin, end = _bounds(len(records_all), len(gpus), index)
            if len(rows) != end - begin:
                raise RuntimeError("incomplete NLL chunk {}".format(chunk))
            all_rows.extend(rows)
        if [str(row["sample_id"]) for row in all_rows] != [
            record_id(record, index) for index, record in enumerate(records_all)
        ]:
            raise RuntimeError("merged NLL sample order mismatch")
        write_jsonl(merged, all_rows)
        print("nll {} merged ({} rows)".format(config, len(all_rows)), flush=True)
    print("nll complete", flush=True)


# --------------------------------------------------------------------------
# LoRA delta statistics (from expert state dicts, CPU)
# --------------------------------------------------------------------------

def _layer_deltas(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Compose each projection's LoRA delta B@A from the state dict."""
    deltas = {}
    for key, tensor in state.items():
        if key.endswith(".lora_B.weight"):
            layer = key[: -len(".lora_B.weight")]
            deltas[layer] = tensor.float() @ state["{}.lora_A.weight".format(layer)].float()
    return deltas


def delta_stats(root_arg: str) -> None:
    root = Path(root_arg)
    deltas: Dict[str, Dict[str, torch.Tensor]] = {}
    for config, spec in CONFIGS.items():
        for expert_id in range(int(spec["k"])):
            state = torch.load(
                root / "checkpoints" / config / "expert_{}".format(expert_id)
                / "expert_{:04d}.pt".format(expert_id),
                map_location="cpu", weights_only=False,
            )
            deltas["{}/e{}".format(config, expert_id)] = _layer_deltas(state["state_dict"])
    per_expert = {}
    for key, layers in deltas.items():
        norms = {layer: float(tensor.norm().item()) for layer, tensor in layers.items()}
        rms = {layer: float(tensor.square().mean().sqrt().item()) for layer, tensor in layers.items()}
        per_expert[key] = {
            "layers": sorted(layers),
            "norms": norms,
            "rms": rms,
            "total_frobenius_norm": math.sqrt(sum(norms[layer] ** 2 for layer in layers)),
            "mean_layer_rms": float(sum(rms.values()) / len(rms)),
        }
    pairwise = {}
    keys = sorted(deltas)
    for a, b in itertools.combinations(keys, 2):
        shared = [layer for layer in deltas[a] if layer in deltas[b]]
        cosines = [float(torch.nn.functional.cosine_similarity(
            deltas[a][layer].flatten().unsqueeze(0),
            deltas[b][layer].flatten().unsqueeze(0)).item())
            for layer in shared]
        pairwise["{}<->{}".format(a, b)] = {
            "mean_cosine": float(sum(cosines) / len(cosines)) if cosines else None,
            "n_layers": len(cosines),
        }
    write_json(root / "diagnostics" / "delta_stats.json", {
        "definition": "delta = lora_B @ lora_A per ComposeLinear projection; "
                      "norms/RMS per layer; pairwise cosine over flattened deltas",
        "experts": per_expert, "pairwise_cosine": pairwise, "git_commit": git_commit(),
    })
    print("delta stats complete: {} experts, {} pairs".format(len(per_expert), len(pairwise)), flush=True)


# --------------------------------------------------------------------------
# offline summary: modes, matrices, oracle decomposition
# --------------------------------------------------------------------------

def _answers_by_id(root: Path, config: str, label: str) -> Dict[str, dict]:
    answers = read_jsonl(root / "evaluations" / "answers" / config / label / "answers.jsonl")
    return {str(row["question_id"]): row for row in answers}


def _candidate_metrics(root: Path, config: str) -> Dict[str, dict]:
    metrics = {}
    for candidate in build_candidates(config):
        metric_file = root / "evaluations" / "answers" / config / candidate["label"] / "score" / "metric.json"
        if metric_file.is_file():
            metrics[candidate["label"]] = read_json(metric_file)
    return metrics


def _score_answers(root: Path, name: str, rows: Sequence[dict], question_file: Path) -> dict:
    """rows: [{question_id, text}] in test order; score a composed selection."""
    answers_path = root / "evaluations" / "composed" / "{}.jsonl".format(name)
    write_jsonl(answers_path, rows)
    metric = _score(question_file, answers_path, answers_path.parent / "score")
    metric["composed_name"] = name
    write_json(answers_path.parent / "score" / "metric.json", metric)
    return metric


def summary(args) -> None:
    root = Path(root_arg(args.root))
    question_file = Path(TEST_FILE if not args.question_file else args.question_file)
    records_all = json.loads(question_file.read_text(encoding="utf-8"))
    if args.max_samples:
        records_all = records_all[: args.max_samples]
    test_ids = [record_id(record, index) for index, record in enumerate(records_all)]
    row = {}

    # ---- candidate accuracies (Mode B rows, singles) ----
    row["candidate_metrics"] = {}
    for config in CONFIGS:
        row["candidate_metrics"][config] = _candidate_metrics(root, config)
    if (root / "evaluations" / "answers" / "base" / "base" / "score" / "metric.json").is_file():
        row["base_accuracy"] = read_json(
            root / "evaluations" / "answers" / "base" / "base" / "score" / "metric.json")

    # ---- Mode A: centroid top-1 ----
    answers_cache = {}
    for k in (2, 4):
        config = "two_r8" if k == 2 else "four_r8"
        assignments = read_json(root / "clustering" / "k{}".format(k) / "test_assignments.json")["assignments"]
        labels = ["single:{}".format(c) for c in range(k)]
        if any(not (root / "evaluations" / "answers" / config / label / "answers.jsonl").is_file()
               for label in labels):
            print("WARNING: Mode A k={} skipped (missing singles answers)".format(k), flush=True)
            continue
        rows_out = []
        for test_id in test_ids:
            cluster = int(assignments[test_id])
            label = "single:{}".format(cluster)
            answer_rows = answers_cache.setdefault(
                (config, label),
                _answers_by_id(root, config, label))
            rows_out.append({"question_id": test_id, "text": answer_rows[test_id]["text"]})
        metric = _score_answers(root, "mode_a_k{}".format(k), rows_out, question_file)
        row["mode_a"] = row.get("mode_a", {})
        row["mode_a"]["k{}".format(k)] = metric
        print("Mode A k={}: {:.2f}".format(k, metric["value"]), flush=True)

    # ---- Mode C: oracle selection from NLL ----
    for config, nll_config in (("two_r8", "two_r8"), ("four_r8", "four_r8"),
                               ("single_r8", "single_r8"), ("rank48", "rank48")):
        nll_path = root / "evaluations" / "nll" / nll_config / "nll.jsonl"
        if not nll_path.is_file():
            continue
        nll_rows = read_jsonl(nll_path)
        nll_by_id = {str(row["sample_id"]): row for row in nll_rows}
        # NLL candidate order mirrors build_candidates filtered to singles +
        # RMS pairs (raw pairs are excluded in nll_worker).  The answer map
        # is keyed by the NLL index.
        nll_candidates = [
            candidate for candidate in build_candidates(nll_config)
            if candidate["kind"] in ("single", "pair")
            and not (candidate["kind"] == "pair" and not candidate["apply_rms"])
        ]
        answer_map = {}
        for nll_index, candidate in enumerate(nll_candidates):
            label = str(candidate["label"])
            if candidate["kind"] == "pair":
                label = "pair:0+1:rms"  # two_r8 oracle pair is the RMS variant
            answer_map[nll_index] = (label, _answers_by_id(root, config, label))
        has_pairs = any(len(c["expert_ids"]) == 2 for c in nll_candidates)
        for selection_kind in ("single", "pair" if has_pairs else "single", "overall"):
            rows_out = []
            for test_id in test_ids:
                nll_row = nll_by_id[test_id]
                best = int(nll_row["best_{}_index".format(selection_kind)])
                if best < 0:
                    raise RuntimeError("no pair candidates for {}:{}".format(config, selection_kind))
                label, answers = answer_map[best]
                rows_out.append({"question_id": test_id, "text": answers[test_id]["text"]})
            metric = _score_answers(
                root, "mode_c_{}_{}".format(config, selection_kind), rows_out, question_file)
            row.setdefault("mode_c", {})[config + ":" + selection_kind] = metric
            print("Mode C {} {}: {:.2f}".format(config, selection_kind, metric["value"]), flush=True)

    # ---- cluster x expert accuracy matrix ----
    matrices = {}
    for k in (2, 4):
        config = "two_r8" if k == 2 else "four_r8"
        assignments = read_json(root / "clustering" / "k{}".format(k) / "test_assignments.json")["assignments"]
        per_cluster_ids = {c: [tid for tid in test_ids if int(assignments[tid]) == c] for c in range(k)}
        if any(not (root / "evaluations" / "answers" / config / "single:{}".format(e) / "answers.jsonl").is_file()
               for e in range(k)):
            print("WARNING: cluster matrix k={} skipped (missing singles)".format(k), flush=True)
            continue
        matrix = {}
        per_expert_overall = {}
        for expert_id in range(k):
            label = "single:{}".format(expert_id)
            answers = _answers_by_id(root, config, label)
            correct = {tid: (answers[tid]["text"].upper() == record["answer"].upper())
                       for tid, record in zip(test_ids, records_all)}
            per_expert_overall[expert_id] = 100.0 * sum(correct.values()) / len(correct)
            matrix[expert_id] = {
                c: (100.0 * sum(correct[tid] for tid in cluster) / len(cluster)
                    if cluster else None)
                for c, cluster in per_cluster_ids.items()
            }
        matrices["k{}".format(k)] = {
            "per_expert_overall": per_expert_overall,
            "matrix": matrix,
            "cluster_sizes": {c: len(cluster) for c, cluster in per_cluster_ids.items()},
        }
    row["cluster_expert_matrices"] = matrices

    # ---- NLL synergy summary ----
    synergies = {}
    for config in CONFIGS:
        nll_path = root / "evaluations" / "nll" / config / "nll.jsonl"
        if not nll_path.is_file():
            continue
        rows_nll = read_jsonl(nll_path)
        if len(rows_nll) != len(test_ids):
            raise RuntimeError("NLL row count mismatch for {}".format(config))
        synergies[config] = {
            "samples": len(rows_nll),
            "pair_oracle_rate": sum(bool(r["pair_oracle"]) for r in rows_nll) / len(rows_nll),
            "mean_synergy": sum(float(r["synergy"]) for r in rows_nll) / len(rows_nll),
            "positive_synergy_rate": sum(float(r["synergy"]) > 0.0 for r in rows_nll) / len(rows_nll),
            "mean_nll_by_label": {
                label: sum(float(r["set_nll"][i]) for r in rows_nll) / len(rows_nll)
                for i, label in enumerate(rows_nll[0]["candidate_labels"])
            },
        }
    row["nll_synergy"] = synergies

    write_json(root / "evaluations" / "summary.json", row)
    print("summary complete", flush=True)


def root_arg(value: str) -> Path:
    return Path(value)


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("assign")
    p.add_argument("--root", required=True)

    p = sub.add_parser("assemble")
    p.add_argument("--root", required=True)

    p = sub.add_parser("rms")
    p.add_argument("--root", required=True)
    p.add_argument("--gpus", default="7")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=0)

    p = sub.add_parser("gen-worker")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--candidate-file", required=True)
    p.add_argument("--question-file", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--num-chunks", type=int, required=True)
    p.add_argument("--chunk-idx", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-samples", type=int, default=0)

    p = sub.add_parser("gen")
    p.add_argument("--root", required=True)
    p.add_argument("--gpus", default="4,5,6,7")
    p.add_argument("--config", default="")
    p.add_argument("--label", default="")
    p.add_argument("--base", action="store_true", help="also generate the base model pass")
    p.add_argument("--question-file", default="")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-samples", type=int, default=0)

    p = sub.add_parser("nll-worker")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--candidates-file", required=True)
    p.add_argument("--question-file", required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--num-chunks", type=int, required=True)
    p.add_argument("--chunk-idx", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=0)

    p = sub.add_parser("nll")
    p.add_argument("--root", required=True)
    p.add_argument("--gpus", default="4,5,6,7")
    p.add_argument("--config", default="", help="one config only (smoke)")
    p.add_argument("--question-file", default="")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=0)

    p = sub.add_parser("delta-stats")
    p.add_argument("--root", required=True)

    p = sub.add_parser("summary")
    p.add_argument("--root", required=True)
    p.add_argument("--question-file", default="")
    p.add_argument("--max-samples", type=int, default=0)

    args = parser.parse_args()
    if args.command == "assign":
        assign(args.root)
    elif args.command == "assemble":
        assemble(args.root)
    elif args.command == "rms":
        rms(args.root, args.gpus, args.batch_size, args.max_samples)
    elif args.command == "gen-worker":
        gen_worker(args)
    elif args.command == "gen":
        gen(args)
    elif args.command == "nll-worker":
        nll_worker(args)
    elif args.command == "nll":
        nll(args)
    elif args.command == "delta-stats":
        delta_stats(args.root)
    elif args.command == "summary":
        summary(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
