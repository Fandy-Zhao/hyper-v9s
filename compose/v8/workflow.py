"""The V8 teacher-screening stage: few-shot Teacher -> task-level ``R_t``.

This module is the *only* place the formal pipeline touches answer supervision
outside the training loss.  It runs between the fixed-query stage and the
candidate-initialisation stage of one task, and it produces exactly three
artifacts:

``<root>/teacher/teacher_samples.json``
    the seeded, stratified teacher subset drawn from ``train_full``, with the
    sample-id hash and the test-overlap check that prove it is reproducible and
    disjoint from test (written by the teacher runner itself).

``<root>/tasks/v8_reusable_screening.json``
    the task-level reusable historical expert set ``R_t`` plus the per-expert
    evidence it was derived from.  **This is the only teacher output the rest of
    the task ever reads**, and it is read for two things: which historical
    experts may be routed to, and how to initialise their current-task reuse
    keys.  It is never read by the training loop.

``<root>/state/reuse_key_init.pt``
    the initial reuse-key tensors, so the candidate-initialisation stage does not
    have to re-read the teacher result.

Task 0 has no history and therefore no teacher: :func:`run_teacher_screening`
writes the *same* artifact schema with ``historical_expert_ids = []`` and
``teacher_sample_count = 0``.  A task that skips the teacher still produces the
artifact, so a reader never has to distinguish "no screening ran" from "screening
ran and found nothing".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from compose.v7.query import full_train_task_center  # noqa: E402
from compose.v8.config import V8_METHOD_NAME  # noqa: E402
from compose.v8.screening import (  # noqa: E402
    aggregate_reusable_experts,
    initialize_reuse_keys,
    teacher_router_recall,
)

#: Where one task's screening artifacts live under its own root.
SCREENING_DIR = "tasks"
SCREENING_NAME = "v8_reusable_screening.json"
REUSE_KEY_INIT_NAME = "reuse_key_init.pt"


class TeacherStageError(RuntimeError):
    """Raised when the teacher stage cannot produce a usable artifact."""


@dataclass(frozen=True)
class TeacherSpec:
    """Everything that determines ``R_t``, declared *before* the run starts.

    These fields -- not the screening artifact they produce -- are what the run
    contract hashes, so a stage marker written before the teacher ran stays valid
    after it.  ``num_samples`` and ``sample_ratio`` are mutually exclusive and
    exactly one is required for a task with history.
    """

    num_samples: Optional[int] = None
    sample_ratio: Optional[float] = None
    seed: int = 42
    min_teacher_support: int = 4
    min_teacher_usage_rate: float = 0.02
    #: Below this many selected-solver samples the reuse key falls back to the
    #: task center instead of the teacher-query centroid.
    min_reuse_key_support: int = 4
    teacher_sampling_strategy: str = "answer_stratified_feature_kcenter"
    teacher_duplicate_cosine_threshold: float = 0.99
    reuse_key_init_strategy: str = "teacher_selected_query_centroid"
    reuse_key_quality_enabled: bool = True
    reuse_key_quality_mode: str = "routed_answer_nll"
    reuse_key_quality_temperature: float = 1.0
    reuse_key_quality_floor: float = 0.10
    recall_top_m: int = 8
    pair_search_mode: str = "bounded"
    shortlist_ks: int = 4
    max_new_tokens: int = 128
    verify_nll_trials: int = 8
    device: str = "cuda:0"
    wait_timeout_seconds: float = 7200.0
    min_free_mib: int = 18000

    def __post_init__(self) -> None:
        if (self.num_samples is None) == (self.sample_ratio is None):
            raise ValueError(
                "set exactly one of teacher num_samples / sample_ratio"
            )
        if self.num_samples is not None and int(self.num_samples) < 1:
            raise ValueError("teacher num_samples must be positive")
        if self.sample_ratio is not None and not 0.0 < float(self.sample_ratio) <= 1.0:
            raise ValueError("teacher sample_ratio must be in (0, 1]")
        if int(self.min_teacher_support) < 1:
            raise ValueError("min_teacher_support must be positive")
        if not 0.0 <= float(self.min_teacher_usage_rate) <= 1.0:
            raise ValueError("min_teacher_usage_rate must be in [0, 1]")
        if int(self.min_reuse_key_support) < 1:
            raise ValueError("min_reuse_key_support must be positive")
        if int(self.min_reuse_key_support) > int(self.min_teacher_support):
            raise ValueError("min_reuse_key_support may not exceed min_teacher_support in formal centroid mode")
        if self.teacher_sampling_strategy not in ("answer_stratified_seeded_random", "answer_stratified_feature_kcenter"):
            raise ValueError("unsupported teacher sampling strategy")
        if not 0.0 < self.teacher_duplicate_cosine_threshold <= 1.0:
            raise ValueError("teacher duplicate cosine threshold must be in (0,1]")
        if self.reuse_key_init_strategy != "teacher_selected_query_centroid":
            raise ValueError("formal V8.1 requires teacher_selected_query_centroid")
        if self.reuse_key_quality_mode != "routed_answer_nll" or self.reuse_key_quality_temperature <= 0 or not 0 <= self.reuse_key_quality_floor <= 1:
            raise ValueError("invalid reuse-key quality specification")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "sample_ratio": self.sample_ratio,
            "seed": int(self.seed),
            "min_teacher_support": int(self.min_teacher_support),
            "min_teacher_usage_rate": float(self.min_teacher_usage_rate),
            "min_reuse_key_support": int(self.min_reuse_key_support),
            "recall_top_m": int(self.recall_top_m),
            "pair_search_mode": str(self.pair_search_mode),
            "shortlist_ks": int(self.shortlist_ks),
            "max_new_tokens": int(self.max_new_tokens),
            "teacher_sampling_strategy": self.teacher_sampling_strategy,
            "teacher_duplicate_cosine_threshold": self.teacher_duplicate_cosine_threshold,
            "reuse_key_init_strategy": self.reuse_key_init_strategy,
            "reuse_key_perturbation": 0.0,
            "reuse_key_quality_enabled": self.reuse_key_quality_enabled,
            "reuse_key_quality_mode": self.reuse_key_quality_mode,
            "reuse_key_quality_temperature": self.reuse_key_quality_temperature,
            "reuse_key_quality_floor": self.reuse_key_quality_floor,
        }


def empty_screening(task_index: int) -> Dict[str, Any]:
    """The legal Task 0 artifact: no history, no teacher, no oracle evaluations."""
    return {
        "schema_version": 1,
        "task_id": int(task_index),
        "method": V8_METHOD_NAME,
        "teacher_sample_count": 0,
        "teacher_search_mode": None,
        "historical_expert_ids": [],
        "reusable_historical_expert_ids": [],
        "criterion": {
            "operator": "and",
            "min_teacher_support": None,
            "min_teacher_usage_rate": None,
            "evidence": "task0 has no historical pool; no screening was run",
            "diagnostic_only": [],
        },
        "expert_statistics": {},
        "teacher_router_agreement": {
            "denominator": 0,
            "TeacherRouterRecall@1": 0.0,
            "TeacherRouterRecall@2": 0.0,
        },
        "reuse_key_initialization": {
            "task_id": int(task_index),
            "support_definition": "teacher_selected_solver",
            "experts": {},
        },
        "full_training_oracle_eval_sample_count": 0,
        "historical_keys_trainable": False,
        "historical_lora_trainable": False,
        "task_specific_historical_keys": False,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _run(command: Sequence[str], env: Mapping[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ {}\n".format(" ".join(str(value) for value in command)))
        handle.flush()
        process = subprocess.run(
            [str(value) for value in command],
            env=dict(env),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if process.returncode != 0:
        tail = "".join(log_path.read_text(encoding="utf-8").splitlines(True)[-40:])
        raise TeacherStageError(
            "teacher stage failed with exit code {} (log {}, {:.1f}s)\n{}".format(
                process.returncode, log_path, time.time() - started, tail
            )
        )


def run_teacher_screening(
    *,
    task_index: int,
    root: Path,
    train_json: Path,
    test_file: Optional[str],
    previous_checkpoint: Optional[str],
    spec: Optional[TeacherSpec],
    python: str,
    env: Mapping[str, str],
    formal_root: Path,
    query_cache_root: Optional[str],
    model_path: str,
    vision_tower: str,
    projector_path: str,
    image_folder: str,
    log_path: Path,
) -> Dict[str, Any]:
    """Run S2/S3 and return the screening artifact (also written to disk).

    ``query_cache_root`` must be the V8 Exact query cache the run already uses;
    the teacher reads the *train* split queries from it so that the reuse-key
    centroid is computed in exactly the coordinate system the router will use.
    """
    screening_path = Path(root) / SCREENING_DIR / SCREENING_NAME
    if int(task_index) == 0 or not previous_checkpoint:
        if spec is not None:
            raise TeacherStageError(
                "Task0 (or a task with no previous checkpoint) has no historical "
                "pool to screen; pass no teacher spec instead of an empty one"
            )
        payload = empty_screening(task_index)
        _write_json(screening_path, payload)
        _write_json(
            Path(root) / SCREENING_DIR / "teacher_stage.json",
            {
                "task_index": int(task_index),
                "teacher_ran": False,
                "reason": "no_previous_checkpoint",
                "historical_expert_ids": [],
                "reusable_historical_expert_ids": [],
            },
        )
        return payload

    if spec is None:
        raise TeacherStageError(
            "Task{} has a previous checkpoint, so it needs a teacher spec: "
            "--v8-teacher-num-samples or --v8-teacher-sample-ratio".format(task_index)
        )
    if not test_file:
        raise TeacherStageError(
            "the teacher subset must be proven disjoint from test; pass --test-file"
        )
    if not query_cache_root:
        raise TeacherStageError(
            "the teacher reads the train fixed-query cache; pass "
            "--compose-v8-query-cache-root"
        )

    checkpoint_dir = Path(previous_checkpoint)
    for required in ("v7_keys.pt", "compose_experts.json", "compose_experts.bin"):
        if not (checkpoint_dir / required).is_file():
            raise TeacherStageError(
                "previous checkpoint {} is missing {}".format(checkpoint_dir, required)
            )

    teacher_command: List[str] = [
        python, "-m", "compose.experiments.v8_teacher_run",
        "--task", str(int(task_index)),
        # ``--root`` is where the teacher writes its artifacts (the task's own
        # pipeline root); ``--formal-root`` is where it *reads* this task's
        # data/*_full.json from.  Both are the formal run root here -- the
        # pipeline writes train_full.json into ``<run_root>/taskN/data``.
        "--root", str(Path(formal_root)),
        "--formal-root", str(Path(formal_root)),
        "--split", "train",
        "--checkpoint-dir", str(checkpoint_dir),
        "--model-path", model_path,
        "--vision-tower", vision_tower,
        "--projector-path", projector_path,
        "--image-folder", image_folder,
        "--query-cache", str(Path(query_cache_root)),
        "--test-id-file", str(Path(test_file)),
        "--history-only",
        "--recall-top-m", str(int(spec.recall_top_m)),
        "--shortlist-ks", str(int(spec.shortlist_ks)),
        "--pair-search-mode", str(spec.pair_search_mode),
        "--max-new-tokens", str(int(spec.max_new_tokens)),
        "--verify-nll-trials", str(int(spec.verify_nll_trials)),
        "--teacher-seed", str(int(spec.seed)),
        "--teacher-sampling-strategy", spec.teacher_sampling_strategy,
        "--teacher-duplicate-cosine-threshold", str(spec.teacher_duplicate_cosine_threshold),
        "--device", str(spec.device),
        "--min-free-mib", str(int(spec.min_free_mib)),
        "--wait-timeout-seconds", str(float(spec.wait_timeout_seconds)),
    ]
    if spec.num_samples is not None:
        teacher_command += ["--teacher-num-samples", str(int(spec.num_samples))]
    else:
        teacher_command += ["--teacher-sample-ratio", str(float(spec.sample_ratio))]
    # Independent frozen-model inference workers, not DDP.  The subset sampler
    # is deterministic before sharding, and the merger verifies its exact
    # round-robin sample-ID partition.  V8_TEACHER_GPUS is intentionally
    # separate from S3's DDP plan because this stage runs before S3.
    requested_gpus = [value.strip() for value in str(env.get("V8_TEACHER_GPUS", "")).split(",") if value.strip()]
    teacher_samples = int(spec.num_samples) if spec.num_samples is not None else 1
    shard_count = min(len(requested_gpus), teacher_samples) if requested_gpus else 1
    if shard_count == 1:
        _run(teacher_command, env, log_path)
    else:
        shard_root = Path(root) / "teacher_shards"
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        processes = []
        shard_dirs = []
        for shard_index in range(shard_count):
            directory = shard_root / "shard{}".format(shard_index)
            shard_dirs.append(directory / "task{}".format(task_index))
            command = list(teacher_command)
            command[command.index("--root") + 1] = str(directory)
            command += ["--shard-count", str(shard_count), "--shard-index", str(shard_index)]
            shard_env = dict(env)
            shard_env["CUDA_VISIBLE_DEVICES"] = requested_gpus[shard_index]
            handle = (Path(log_path).parent / "v8_teacher_shard{}.log".format(shard_index)).open("w", encoding="utf-8")
            processes.append((subprocess.Popen(command, env=shard_env, stdout=handle, stderr=subprocess.STDOUT), handle))
        failures = []
        for process, handle in processes:
            code = process.wait(); handle.close()
            if code:
                failures.append(code)
        if failures:
            raise TeacherStageError("teacher shard workers failed: {}".format(failures))
        merge_command = [python, "-m", "compose.experiments.v8_merge_teacher_shards",
                         "--train-json", str(train_json),
                         "--teacher-sample-manifest", str(shard_dirs[0] / "teacher_samples.json"),
                         "--output", str(Path(formal_root) / "task{}".format(task_index) / "teacher_result.json")]
        for directory in shard_dirs:
            merge_command += ["--shard-dir", str(directory)]
        _run(merge_command, env, Path(log_path).with_name("v8_teacher_merge.log"))

    teacher_path = Path(formal_root) / "task{}".format(task_index) / "teacher_result.json"
    if not teacher_path.is_file():
        raise TeacherStageError(
            "teacher run reported success but wrote no {}".format(teacher_path)
        )
    teacher_payload = _read_json(teacher_path)
    if int(teacher_payload.get("task_id", -1)) != int(task_index):
        raise TeacherStageError("teacher result belongs to a different task")

    # The teacher searched a capability universe; ``R_t`` may only be built from
    # the pool that training will actually freeze.  A mismatch here means the
    # teacher ran against a different checkpoint than the one this task trains
    # from, which is the one failure mode that would silently fabricate history.
    key_state = torch.load(checkpoint_dir / "v7_keys.pt", map_location="cpu", weights_only=False)
    frozen_ids = sorted(
        int(key_id) for key_id in (key_state.get("keys") or {})
        if str(key_id).isdigit()
    )
    visible = sorted(int(value) for value in teacher_payload["historical_experts_visible"])
    if visible != frozen_ids:
        raise TeacherStageError(
            "teacher searched {} but the frozen pool holds {}; refusing to build "
            "R_t from a mismatched history".format(visible, frozen_ids)
        )

    screening = aggregate_reusable_experts(
        teacher_payload,
        min_teacher_support=int(spec.min_teacher_support),
        min_teacher_usage_rate=float(spec.min_teacher_usage_rate),
    )
    screening["method"] = V8_METHOD_NAME
    screening["teacher_spec"] = spec.to_dict()
    screening["teacher_result_sha256"] = _sha256(teacher_path)
    screening["previous_checkpoint"] = str(checkpoint_dir.resolve())

    # The reuse-key centroid must live in the router's own coordinate system, so
    # it is read from the same fixed-query tensor cache the teacher used rather
    # than re-derived from the JSON features.
    query_tensor_path = (
        Path(query_cache_root) / "query_cache"
        / "task{}".format(task_index) / "train" / "queries.pt"
    )
    if not query_tensor_path.is_file():
        raise TeacherStageError(
            "fixed-query tensor cache missing {}; the reuse-key centroid must be "
            "computed in the router's own coordinate system".format(query_tensor_path)
        )
    query_payload = torch.load(query_tensor_path, map_location="cpu", weights_only=False)
    screening["teacher_router_agreement"] = teacher_router_recall(
        teacher_payload, screening, query_payload, key_state
    )

    features = _read_json(Path(root) / "features" / "train.json")
    train_queries = torch.tensor(
        [features["records"][sample_id]["query"] for sample_id in sorted(features["records"])],
        dtype=torch.float32,
    )
    center, _coverage = full_train_task_center(train_queries, int(train_queries.shape[0]))

    reuse_keys, reuse_audit = initialize_reuse_keys(
        teacher_payload, screening["reusable_historical_expert_ids"], query_payload,
        center=center,
        perturbation=0.0,
        min_support=int(spec.min_reuse_key_support),
        seed=int(spec.seed) + int(task_index),
        task_index=int(task_index),
        expected_support_by_expert=screening["experts"],
    )
    screening["reuse_key_initialization"] = reuse_audit
    screening["task_center"] = center.tolist()
    screening["train_query_count"] = int(train_queries.shape[0])
    _write_json(screening_path, screening)
    torch.save(
        {"keys": reuse_keys, "audit": reuse_audit},
        Path(root) / SCREENING_DIR / REUSE_KEY_INIT_NAME,
    )
    return screening


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
