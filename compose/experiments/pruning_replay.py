"""Standalone S5 pruning replay harness for the RMS/pruning acceleration study.

Why this exists
---------------
``compose/experiments/v7_task_run.py`` binds every stage marker to a
``run_contract_hash`` built by ``build_run_contract``, and that hash embeds
``git_sha``.  ``stage_done()`` raises on a stale hash, and
``rebind_run_contract()`` refuses when git HEAD did not move.  Any working-tree
change therefore makes the orchestrator refuse to re-enter S5 -- so the
orchestrator cannot serve as the *measurement rig* this study needs.

This module is that rig, and nothing else.  It is only a driver: everything
that decides the pruning outcome is imported, never reimplemented --
``CandidatePruner`` (remove-and-reroute, gain, redundancy, tie-breaking),
``GlobalTop2Router``, ``V7ExpertKeyPool``, ``commit_retained_candidates``,
``route_manifest``, ``mean_nll`` and ``deep_resolve``.  The one piece this
module owns is the *scorer* -- the subprocess pipeline producing per-candidate
evidence -- and it is a line-for-line copy of ``execute_scoring_job`` from
``v7_task_run.py`` (S5).

Faithfulness is not asserted here, it is *measured*.  ``--verify-against``
byte-compares every artifact this harness writes with the production tree, so
"the replay reproduces production" is a checked claim rather than a comment.

The reference bundle it emits follows the schema documented in
``compose/experiments/compare_rms_pruning.py``::

    {"label", "routing", "usage", "removal", "redundancy", "decision",
     "route_types", "retained_expert_ids", "pruned_expert_ids",
     "commit_manifest", "wall_clock"}

Usage::

    # production-faithful baseline, one GPU
    python -m compose.experiments.pruning_replay \\
        --task-root  <run>/task5 \\
        --output-root <scratch>/prune_baseline \\
        --gpus 6 --mode baseline --label prune_baseline \\
        --verify-against <run>/task5

    # accelerated: digest-keyed job reuse + persistent evidence cache
    python -m compose.experiments.pruning_replay \\
        --task-root  <run>/task5 \\
        --output-root <scratch>/prune_accel \\
        --gpus 6,7 --mode accelerated --label prune_accel \\
        --evidence-cache <scratch>/evidence
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import torch
import yaml

from compose.v7.commit import commit_retained_candidates
from compose.v7.config import V7Config
from compose.v7.gpu_plan import GpuPlanError, idle_gpu_subset, probe_gpu_state
from compose.v7.pool import V7ExpertKeyPool
from compose.v7.pruning import CandidatePruner
from compose.v7.workers import (
    PooledJobRunner,
    deep_resolve,
    deferred_from,
    make_worker_env,
    run_job_logged,
)
from compose.v7.workflow import mean_nll, queries_from_cache, route_manifest

#: Whole-file artifacts that are pure functions of the computation, so they
#: must match production byte for byte.
VERIFIED_BYTE_EXACT = (
    "selections_{}.json",
    "nll_{}.json",
)
#: Artifacts carrying environment-dependent fields (wall-clock, peak memory,
#: the git SHA stamped at generation time, absolute output paths).  These are
#: compared field-wise against :data:`STABLE_FIELDS`, which names every field
#: that *is* a function of the computation; anything excluded is listed here so
#: the exclusion is reviewable rather than silent.
VERIFIED_FIELD_WISE = {
    # ``result_text_sha256`` hashes the *path* of the scorer's Result.text file
    # (see formal_ucit_eval.finalize: ``_sha256(str(result_text))``), so it is a
    # property of the scratch directory, not of the score.  ``prediction_file``
    # is likewise an absolute output path.  Both are excluded; the generated
    # text they would have stood in for is compared directly in answers.jsonl.
    "official_metric_{}.json": (
        "value",
        "task_id",
        "metric",
        "scorer",
        "dataset",
        "validation_only",
        "score_unit",
        "annotation_file",
    ),
    "generation_{}.json": (
        "adapter_kind",
        "samples",
        "seed",
        "selection_mode",
        "router_selection_histogram",
        "cross_task_expert_pair_frequency",
    ),
}
#: ``answers.jsonl`` is compared record-wise on the generated content.  Its
#: ``metadata.git_commit`` is the HEAD at generation time and therefore cannot
#: match a replay run from a later commit -- it is the only field dropped, so
#: the generated text itself is what carries the comparison.
ANSWER_RECORD_FIELDS = ("question_id", "prompt", "text", "model_id")
ANSWER_SELECTION_FIELDS = (
    "expert_ids",
    "selection_source",
    "answer_features_used",
    "oracle_used",
    "task_id_lookup_used",
    "clustering_used_at_test",
)

MANIFEST_NAME = "compose_experts.json"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def route_digest(rows: torch.Tensor) -> str:
    """Content digest of a ``[N, 2]`` int route tensor.

    Two jobs with the same digest request byte-identical evidence, so the
    second one can be served from the first one's artifacts.  The shape is fed
    into the hash so a truncated manifest can never collide with a full one.
    """
    tensor = rows.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def configuration_name(excluded: FrozenSet[int]) -> str:
    """Stable, human-legible name of a router configuration.

    The excluded set *is* the configuration: ``GlobalTop2Router`` is a pure
    function of ``(queries, excluded)``, so naming a configuration by its
    excluded set is exactly naming it by the routing function applied.
    """
    if not excluded:
        return "full"
    return "exclude_" + "+".join(str(value) for value in sorted(excluded))


@dataclass
class StageInputs:
    """Everything S5 reads, loaded and cross-checked against the contract."""

    root: Path
    output: Path
    config: V7Config
    pool: V7ExpertKeyPool
    train_queries: torch.Tensor
    train_ids: Tuple[str, ...]
    val_queries: torch.Tensor
    val_ids: Tuple[str, ...]
    center: torch.Tensor
    model_path: str
    vision_tower: str
    projector_path: str
    image_folder: str
    python: str
    val_json: Path
    annotation_file: str
    validation_metric: str
    task_index: int
    runtime_contract: Path
    candidate_ids: Tuple[int, ...]
    contract_hash: str
    git_sha: str


def load_stage_inputs(
    task_root: str,
    *,
    python: Optional[str] = None,
    model_path: Optional[str] = None,
    vision_tower: Optional[str] = None,
    projector_path: Optional[str] = None,
    image_folder: Optional[str] = None,
    annotation_file: Optional[str] = None,
    validation_metric: Optional[str] = None,
    runtime_contract: Optional[str] = None,
) -> StageInputs:
    """Load S5's inputs from a completed task root.

    The method config is re-hashed against ``run_contract.json`` before use:
    replaying under a *different* pruning config would produce a
    self-consistent bundle that proves nothing, so a mismatch fails loudly.
    """
    root = Path(task_root).resolve()
    contract = json.loads((root / "data" / "run_contract.json").read_text(encoding="utf-8"))
    coverage = json.loads((root / "data" / "coverage.json").read_text(encoding="utf-8"))

    config_entry = contract["files"]["method_config"]
    config_path = Path(config_entry["path"])
    observed = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if observed != config_entry["sha256"]:
        raise ValueError(
            "method config {} hashes to {}, but the run contract pinned {}; "
            "the replay would not reproduce this run".format(
                config_path, observed, config_entry["sha256"]
            )
        )
    config = V7Config.from_dict(yaml.safe_load(config_path.read_text(encoding="utf-8")))

    output = root / "training"
    pool = V7ExpertKeyPool.from_state(
        torch.load(output / "v7_key_pool.pt", map_location="cpu", weights_only=False)
    )
    train_queries, train_ids = queries_from_cache(
        str(root / "features" / "train.json"), coverage["num_train_samples"]
    )
    val_queries, val_ids = queries_from_cache(
        str(root / "features" / "val.json"), coverage["num_validation_samples"]
    )
    center = torch.tensor(
        json.loads(
            (root / "metrics" / "candidate_initialization.json").read_text(encoding="utf-8")
        )["task_center"]
    )

    resolved_annotation = annotation_file or contract["files"]["validation_annotation"]["path"]
    val_json = root / "data" / "val_full.json"
    if Path(resolved_annotation).resolve() == Path(val_json).resolve():
        # Mirrors resolve_annotation_file() in v7_task_run.py: rewritten
        # validation IDs must be scored against the rewritten artifact.
        resolved_annotation = str(val_json)

    return StageInputs(
        root=root,
        output=output,
        config=config,
        pool=pool,
        train_queries=train_queries,
        train_ids=train_ids,
        val_queries=val_queries,
        val_ids=val_ids,
        center=center,
        model_path=model_path or contract["model_path"],
        vision_tower=vision_tower or contract["vision_tower"],
        projector_path=projector_path or contract["projector_path"],
        image_folder=image_folder or contract["image_folder"],
        python=python or sys.executable,
        val_json=val_json,
        annotation_file=resolved_annotation,
        validation_metric=validation_metric or contract["validation_metric"],
        task_index=int(contract["task_index"]),
        runtime_contract=(
            Path(runtime_contract) if runtime_contract else root / "data" / "runtime_contract.json"
        ),
        candidate_ids=tuple(int(value) for value in pool.current_ids),
        contract_hash=str(contract["contract_hash"]),
        git_sha=str(contract["git_sha"]),
    )


@dataclass
class RouteRecord:
    """One router invocation, i.e. one configuration of the pruning trajectory."""

    configuration: str
    excluded: FrozenSet[int]
    expert_ids: List[List[int]]
    route_types: List[str]
    digest: str
    scored: bool = False


class RouteRecorder:
    """Transparent proxy around ``GlobalTop2Router`` that records its calls.

    The router is a pure function of ``(queries, excluded)``, so recording the
    excluded set alongside the emitted routes is enough to label every
    configuration the trajectory visits -- without reimplementing any part of
    ``CandidatePruner.evaluate``.
    """

    def __init__(self, router: Any, records: List[RouteRecord]) -> None:
        self._router = router
        self._records = records

    def __call__(self, queries: torch.Tensor, excluded: Sequence[int] = ()) -> Any:
        excluded_set = frozenset(int(value) for value in excluded)
        result = self._router(queries, excluded=excluded_set)
        self._records.append(
            RouteRecord(
                configuration=configuration_name(excluded_set),
                excluded=excluded_set,
                expert_ids=result.expert_ids.detach().cpu().tolist(),
                route_types=list(result.route_types),
                digest=route_digest(result.expert_ids),
            )
        )
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._router, name)


@dataclass
class ScoringJobs:
    """The scorer handed to ``CandidatePruner`` -- production's, plus reuse.

    ``mode="baseline"`` reproduces the legacy sequential scorer exactly: one
    fresh three-subprocess job per score request, executed inline.

    ``mode="accelerated"`` differs in three ways, each independently
    attributable and each additive to the baseline's evidence:

    * a job whose route digest was already scored is served from that job's
      artifacts instead of being recomputed (structural identity: the
      full-pool of iteration ``k+1`` is the removed candidate's reroute of
      iteration ``k``);
    * jobs are dispatched through ``PooledJobRunner``, so the independent
      removal hypotheses of one iteration overlap on separate GPUs while the
      serial trajectory logic -- and therefore the decisions -- is untouched;
    * ``--evidence-cache`` lets the per-sample scorer subprocesses persist and
      reuse their own per-sample results across jobs.
    """

    inputs: StageInputs
    output_root: Path
    gpus: Sequence[int]
    mode: str
    python: str
    evidence_cache: Optional[Path] = None
    timings: Dict[str, float] = field(default_factory=dict)
    reuse_log: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.pruning_dir = self.output_root / "pruning"
        self.log_dir = self.output_root / "logs"
        self.pruning_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.base_env = dict(os.environ)
        self.by_digest: Dict[str, List[int]] = {}
        self.digest_of: Dict[int, str] = {}
        self.score_index = 0
        self._runner: Optional[PooledJobRunner] = None
        self._deferred = self.mode == "accelerated"

    # -- infrastructure -------------------------------------------------

    def _runner_or_none(self) -> Optional[PooledJobRunner]:
        if self.mode != "accelerated":
            return None
        if self._runner is None:
            idle = idle_gpu_subset(list(self.gpus))
            if len(idle) != len(self.gpus):
                raise GpuPlanError(
                    "replay GPU pool narrowed by the idle probe: {} -> {} "
                    "(live state {}); another tenant is holding a requested "
                    "GPU, which would change the wall-clock being measured".format(
                        list(self.gpus), idle, probe_gpu_state()
                    )
                )
            self._runner = PooledJobRunner(
                idle,
                usage_log_path=str(self.output_root / "data" / "stage_gpu_usage.jsonl"),
            )
        return self._runner

    def shutdown(self) -> None:
        if self._runner is not None:
            self._runner.shutdown()
            self._runner = None

    # -- the scoring pipeline (a copy of execute_scoring_job) -------------

    def _execute_job(self, index: int, rows: torch.Tensor, worker_env: Dict[str, str]) -> Dict[str, Any]:
        selections = self.pruning_dir / "selections_{}.json".format(index)
        nll_output = self.pruning_dir / "nll_{}.json".format(index)
        write_json(selections, route_manifest(self.inputs.val_ids, rows))
        cache_args = (
            ["--evidence-cache", str(self.evidence_cache)] if self.evidence_cache else []
        )
        run_job_logged(
            [
                self.python, "-m", "compose.eval.nll_eval",
                "--model-path", self.inputs.model_path,
                "--vision-tower", self.inputs.vision_tower,
                "--projector-path", self.inputs.projector_path,
                "--checkpoint-dir", str(self.inputs.output),
                "--question-file", str(self.inputs.val_json),
                "--image-folder", self.inputs.image_folder,
                "--selections", str(selections),
                "--output", str(nll_output),
                "--device", "cuda:0", "--batch-size", "1",
                "--image-aspect-ratio", self.inputs.config.runtime.image_aspect_ratio,
                "--runtime-contract", str(self.inputs.runtime_contract),
            ] + cache_args,
            worker_env,
            str(self.log_dir / "pruning_{}.log".format(index)),
        )
        loss = mean_nll(str(nll_output))
        if self.inputs.validation_metric == "official_ucit":
            answers = self.pruning_dir / "answers_{}.jsonl".format(index)
            summary = self.pruning_dir / "generation_{}.json".format(index)
            run_job_logged(
                [
                    self.python, "-m", "compose.eval.eval_task",
                    "--adapter-kind", "compose",
                    "--model-path", self.inputs.model_path,
                    "--checkpoint-dir", str(self.inputs.output),
                    "--projector-path", self.inputs.projector_path,
                    "--vision-tower", self.inputs.vision_tower,
                    "--question-file", str(self.inputs.val_json),
                    "--image-folder", self.inputs.image_folder,
                    "--answers-file", str(answers),
                    "--run-summary-file", str(summary),
                    "--selection-manifest", str(selections),
                    "--device", "cuda:0",
                    "--runtime-contract", str(self.inputs.runtime_contract),
                ] + cache_args,
                worker_env,
                str(self.log_dir / "pruning_generation_{}.log".format(index)),
            )
            metric_output = self.pruning_dir / "official_metric_{}.json".format(index)
            run_job_logged(
                [
                    self.python, "-m", "compose.eval.v7_validation_metric",
                    "--task-index", str(self.inputs.task_index),
                    "--annotation-file", self.inputs.annotation_file,
                    "--predictions-file", str(answers),
                    "--work-root", str(self.pruning_dir / "official_work_{}".format(index)),
                    "--output", str(metric_output),
                ],
                worker_env,
                str(self.log_dir / "pruning_metric_{}.log".format(index)),
            )
            official = json.loads(metric_output.read_text(encoding="utf-8"))
            return {
                "metric": float(official["value"]),
                "loss": float(loss),
                "official_metric": official,
                "answer_nll": float(loss),
                "metric_fallback": False,
            }
        return {
            "metric": -float(loss),
            "loss": float(loss),
            "official_metric": None,
            "answer_nll": float(loss),
            "metric_fallback": True,
            "fallback_reason": "explicit_nll_fallback",
        }

    def _cached_result(self, source_index: int) -> Dict[str, Any]:
        """Re-read a previously scored job's evidence -- never re-forward."""
        loss = mean_nll(str(self.pruning_dir / "nll_{}.json".format(source_index)))
        if self.inputs.validation_metric == "official_ucit":
            official = json.loads(
                (self.pruning_dir / "official_metric_{}.json".format(source_index)).read_text(
                    encoding="utf-8"
                )
            )
            return {
                "metric": float(official["value"]),
                "loss": float(loss),
                "official_metric": official,
                "answer_nll": float(loss),
                "metric_fallback": False,
            }
        return {
            "metric": -float(loss),
            "loss": float(loss),
            "official_metric": None,
            "answer_nll": float(loss),
            "metric_fallback": True,
            "fallback_reason": "explicit_nll_fallback",
        }

    # -- the scorer ------------------------------------------------------

    def scorer(self, rows: torch.Tensor) -> Mapping[str, Any]:
        index = self.score_index
        self.score_index += 1
        # Keep the CPU tensor: route_manifest requires a tensor (.shape).
        cpu_rows = rows.detach().cpu()
        digest = route_digest(cpu_rows)
        self.digest_of[index] = digest

        if self.mode == "accelerated" and digest in self.by_digest:
            source = self.by_digest[digest][0]
            self.reuse_log.append(
                {"job": index, "reused_from": source, "digest": digest, "kind": "digest"}
            )
            return self._cached_result(source)

        self.by_digest.setdefault(digest, []).append(index)
        if not self._deferred:
            return self._execute_job(index, cpu_rows, make_worker_env(self.base_env, self.gpus[0]))

        def body(gpu_id: int, extra_env: Dict[str, str]) -> Dict[str, Any]:
            worker_env = make_worker_env(self.base_env, gpu_id)
            return self._execute_job(index, cpu_rows, worker_env)

        future = self._runner_or_none().submit(body)
        result = {
            "metric": deferred_from(future, "metric"),
            "loss": deferred_from(future, "loss"),
            "official_metric": deferred_from(future, "official_metric"),
            "answer_nll": deferred_from(future, "answer_nll"),
            "metric_fallback": (
                False if self.inputs.validation_metric == "official_ucit" else True
            ),
        }
        if self.inputs.validation_metric != "official_ucit":
            result["fallback_reason"] = "explicit_nll_fallback"
        return result


def build_bundle(
    inputs: StageInputs,
    retained: Sequence[int],
    metrics: Mapping[int, Mapping[str, Any]],
    audit: Mapping[str, Any],
    records: List[RouteRecord],
    jobs: ScoringJobs,
    label: str,
    wall_clock: Dict[str, float],
    committed_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Flatten the run into the comparator's bundle schema.

    Every value here is derived from ``audit``/``metrics``, which are the
    objects production writes to ``candidate_pruning.json`` -- so the bundle
    cannot disagree with the run it describes.
    """
    val_count = len(inputs.val_ids)

    routing: Dict[str, Dict[str, List[int]]] = {}
    route_types: Dict[str, Dict[str, str]] = {}
    for record in records:
        if len(record.expert_ids) != val_count:
            continue  # the train-split call, not part of the pruning evidence
        routing[record.configuration] = {
            str(sample_id): row
            for sample_id, row in zip(inputs.val_ids, record.expert_ids)
        }
        route_types[record.configuration] = {
            str(sample_id): value
            for sample_id, value in zip(inputs.val_ids, record.route_types)
        }

    current = tuple(int(value) for value in inputs.pool.current_ids)
    trajectory = audit["pruning_trajectory"]

    removal: Dict[str, Optional[float]] = {}
    decision: Dict[str, str] = {}
    for entry in trajectory:
        pool_before = {int(value) for value in entry["pool_before"]}
        excluded = frozenset(set(current) - pool_before)
        configuration = configuration_name(excluded)
        key = "{}/{}".format(configuration, int(entry["candidate"]))
        removal[key] = (
            None
            if entry["metric_minus_candidate"] is None
            else float(entry["metric_minus_candidate"])
        )
        decision[key] = str(entry["decision"])

    redundancy: Dict[str, float] = {}
    for expert_id, detail in metrics.items():
        for record in detail.get("redundancy") or ():
            other = int(record["other_expert_id"])
            pair = "{}|{}".format(min(int(expert_id), other), max(int(expert_id), other))
            redundancy[pair] = float(record["key_cosine"])

    usage = {str(int(key)): int(value["selection_count"]) for key, value in metrics.items()}
    retained_ids = [int(value) for value in retained]
    pruned_ids = [int(value) for value in current if int(value) not in set(retained_ids)]

    commit_manifest = None
    if committed_dir is not None and (committed_dir / MANIFEST_NAME).is_file():
        commit_manifest = json.loads(
            (committed_dir / MANIFEST_NAME).read_text(encoding="utf-8")
        )

    digest_by_job = {
        str(index): digest for index, digest in sorted(jobs.digest_of.items())
    }
    return {
        "label": label,
        "task_index": inputs.task_index,
        "task_root": str(inputs.root),
        "contract_hash": inputs.contract_hash,
        "git_sha": inputs.git_sha,
        "candidate_ids": [int(value) for value in inputs.candidate_ids],
        "sample_ids": [str(value) for value in inputs.val_ids],
        "routing": routing,
        "route_types": route_types,
        "usage": usage,
        "removal": removal,
        "decision": decision,
        "redundancy": redundancy,
        "retained_expert_ids": retained_ids,
        "pruned_expert_ids": pruned_ids,
        "final_selectable_pool": [int(value) for value in audit["final_selectable_pool"]],
        "thresholds": audit["thresholds"],
        "pruning_trajectory": trajectory,
        "commit_manifest": commit_manifest,
        "job_digests": digest_by_job,
        "reuse_log": jobs.reuse_log,
        "wall_clock": wall_clock,
    }


def _answer_rows(path: Path) -> List[Dict[str, Any]]:
    """Project an ``answers.jsonl`` onto the fields that are computation."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        selection = (record.get("metadata") or {}).get("selection") or {}
        projected = {key: record.get(key) for key in ANSWER_RECORD_FIELDS}
        projected["selection"] = {key: selection.get(key) for key in ANSWER_SELECTION_FIELDS}
        rows.append(projected)
    return rows


def _verify(replay_dir: Path, production_dir: Path, job_count: int) -> Dict[str, Any]:
    """Compare replay artifacts against the production tree.

    A replay that does not reproduce production proves nothing about the
    optimization built on top of it, so this comparison is the load-bearing
    check, not a formality.

    ``selections_*.json`` and ``nll_*.json`` are pure functions of the
    computation and are compared byte for byte.  The remaining artifacts carry
    environment-dependent fields, so they are compared on a *named* set of
    computation-only fields (:data:`VERIFIED_FIELD_WISE`,
    :data:`ANSWER_RECORD_FIELDS`) -- never on a hand-waved subset, and never by
    rounding a number.  The one dropped field is ``metadata.git_commit``, which
    is the repository HEAD at generation time; that it is dropped is visible
    here, and ``official_metric``'s ``result_text_sha256`` independently pins
    the generated texts.
    """
    rows: List[Dict[str, Any]] = []

    def record(name: str, status: str, **extra: Any) -> None:
        rows.append({"artifact": name, "status": status, **extra})

    for index in range(job_count):
        for pattern in VERIFIED_BYTE_EXACT:
            name = pattern.format(index)
            left, right = replay_dir / name, production_dir / name
            if not right.is_file():
                record(name, "absent_in_production")
            elif not left.is_file():
                record(name, "missing_in_replay")
            else:
                left_bytes, right_bytes = left.read_bytes(), right.read_bytes()
                record(
                    name,
                    "byte_identical" if left_bytes == right_bytes else "differs",
                    bytes=len(left_bytes),
                )

        for pattern, fields in VERIFIED_FIELD_WISE.items():
            name = pattern.format(index)
            left, right = replay_dir / name, production_dir / name
            if not right.is_file():
                record(name, "absent_in_production")
                continue
            if not left.is_file():
                record(name, "missing_in_replay")
                continue
            left_payload = json.loads(left.read_text(encoding="utf-8"))
            right_payload = json.loads(right.read_text(encoding="utf-8"))
            differing = [
                field
                for field in fields
                if left_payload.get(field) != right_payload.get(field)
            ]
            record(
                name,
                "field_identical" if not differing else "differs",
                compared_fields=list(fields),
                **({"differing_fields": differing} if differing else {}),
            )

        name = "answers_{}.jsonl".format(index)
        left, right = replay_dir / name, production_dir / name
        if not right.is_file():
            record(name, "absent_in_production")
        elif not left.is_file():
            record(name, "missing_in_replay")
        else:
            left_rows, right_rows = _answer_rows(left), _answer_rows(right)
            record(
                name,
                "record_identical" if left_rows == right_rows else "differs",
                records=len(left_rows),
                dropped_fields=["metadata.git_commit"],
            )

    mismatches = [row for row in rows if row["status"] == "differs"]
    return {
        "artifacts": rows,
        "mismatches": mismatches,
        "production_faithful": not mismatches,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task-root", required=True, help="completed task dir, e.g. <run>/task5")
    parser.add_argument("--output-root", required=True, help="scratch dir for this replay")
    parser.add_argument("--gpus", default="0", help="comma-separated physical GPU ids")
    parser.add_argument("--mode", choices=("baseline", "accelerated"), default="baseline")
    parser.add_argument(
        "--evidence-cache",
        default=None,
        help="per-sample evidence cache shared by the scorer subprocesses",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--annotation-file", default=None)
    parser.add_argument("--runtime-contract", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--verify-against",
        default=None,
        help="production task dir whose pruning/ artifacts must match byte for byte",
    )
    parser.add_argument("--output", default=None, help="bundle path (default <output-root>/reference_bundle.json)")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    label = args.label or "{}_task{}".format(args.mode, Path(args.task_root).name)
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]

    inputs = load_stage_inputs(
        args.task_root,
        python=args.python,
        annotation_file=args.annotation_file,
        runtime_contract=args.runtime_contract,
    )
    print(
        "[replay] task={} candidates={} val_samples={} mode={} gpus={}".format(
            inputs.task_index, list(inputs.candidate_ids), len(inputs.val_ids), args.mode, gpus
        ),
        flush=True,
    )

    jobs = ScoringJobs(
        inputs=inputs,
        output_root=output_root,
        gpus=gpus,
        mode=args.mode,
        python=args.python,
        evidence_cache=Path(args.evidence_cache) if args.evidence_cache else None,
    )
    records: List[RouteRecord] = []
    pruner = CandidatePruner(inputs.pool, inputs.config.pruning)
    pruner.router = RouteRecorder(pruner.router, records)

    started = time.time()
    try:
        retained, metrics, audit = pruner.evaluate(
            inputs.train_queries, inputs.val_queries, inputs.center, jobs.scorer
        )
    finally:
        jobs.shutdown()
    elapsed = time.time() - started

    if jobs._deferred:
        metrics = deep_resolve(metrics)
        audit = deep_resolve(audit)

    audit["performance_metric"] = (
        "task_specific_official_ucit"
        if inputs.validation_metric == "official_ucit"
        else "negative_answer_nll_explicit_fallback"
    )
    pruning_payload = {
        "retained_candidate_ids": list(retained),
        "retained_candidate_count": len(retained),
        "pool_size_before_task": len(inputs.pool.historical_ids),
        "pool_size_after_task": len(inputs.pool.historical_ids) + len(retained),
        "candidates": {str(key): value for key, value in metrics.items()},
        "audit": audit,
    }
    write_json(output_root / "metrics" / "candidate_pruning.json", pruning_payload)
    write_json(
        output_root / "metrics" / "candidate_pruning_trajectory.json",
        audit["pruning_trajectory"],
    )

    committed_dir = output_root / "committed"
    if committed_dir.exists():
        shutil.rmtree(committed_dir)
    commit_retained_candidates(
        str(inputs.output), str(committed_dir), inputs.pool, retained, metrics
    )

    wall_clock = {
        "pruning_seconds": elapsed,
        "jobs": jobs.score_index,
        "reused_jobs": len(jobs.reuse_log),
        "model_loads": 0 if args.mode == "accelerated" else jobs.score_index,
    }
    bundle = build_bundle(
        inputs, retained, metrics, audit, records, jobs, label, wall_clock, committed_dir
    )
    bundle_path = Path(args.output) if args.output else output_root / "reference_bundle.json"
    write_json(bundle_path, bundle)

    print(
        "[replay] retained={} pruned={} jobs={} reused={} elapsed={:.1f}s".format(
            list(retained), bundle["pruned_expert_ids"], jobs.score_index,
            len(jobs.reuse_log), elapsed,
        ),
        flush=True,
    )

    exit_code = 0
    if args.verify_against:
        verification = _verify(
            output_root / "pruning", Path(args.verify_against) / "pruning", jobs.score_index
        )
        write_json(output_root / "verification.json", verification)
        if verification["production_faithful"]:
            print("[replay] VERIFY: production-faithful (all artifacts identical)", flush=True)
        else:
            print(
                "[replay] VERIFY: MISMATCH against production: {}".format(
                    verification["mismatches"][:5]
                ),
                flush=True,
            )
            exit_code = 2

    print("[replay] bundle -> {}".format(bundle_path), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
