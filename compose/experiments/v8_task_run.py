"""V8-A: run the Answer-Supervised Expert Teacher on the committed V7 pool.

This is the cheapest experiment that can falsify V8's central claim.  The
committed pool at ``task5/committed`` already contains 23 frozen experts and
their frozen origin keys; V8-A asks, for a held-out validation split, **which of
these samples the existing pool can already solve, and with how few experts**.

What it does *not* do, deliberately:

* it does not train anything -- the pool is frozen, so the result is a property
  of V7's pool, not of V8's training loop;
* it does not create alias keys -- with one key per expert, V8's multi-key
  router reduces exactly to V7's single-key router, so Goal B is not testable
  here and the report says so instead of pretending otherwise;
* it never reads the test split.  Ground-truth answers are read for the
  *validation* split only, which is the same license the V7 reproduction script
  uses.

Nothing in the generation path may look at an answer: the teacher decides with
the task metric, and the metric is computed *after* generation from the
prediction and the validation ground truth.  ``compose/v8/inference.py``
enforces the same separation on the inference module by AST inspection.

The output layout, under ``--root/task{N}/``::

    run_config.json          exactly what was run, including the frozen hashes
    recall.json              Top-M recall and the full-pool ranking per sample
    generation_cache.jsonl   every generated answer, append-only
    nll_cache.jsonl          every teacher-forced answer NLL, append-only
    teacher_result.json      the per-sample V8 teacher decision
    v8_policy_answers.jsonl  the assembled minimal-cardinality policy
    evaluation/              official Result.text / metric.json
    analysis.json            recall curves, gap-closed, PART 36 diagnosis
    COMPLETE.json            written last; its presence means the run finished
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from compose.adapters.types import ComposeSelection, pad_selection  # noqa: E402
from compose.data.records import answer_text, question_text  # noqa: E402
from compose.eval.formal_ucit_eval import _score_answers  # noqa: E402
from compose.eval.load_compose import load_compose_model  # noqa: E402
from compose.v7.training import (  # noqa: E402
    supervised_token_mask,
    teacher_forcing_token_nll,
)
from compose.v8.audit import (  # noqa: E402
    diagnose,
    full_pool_oracle_recall_at_k,
    gap_closed,
    teacher_expert_recall_at_k,
)
from compose.v8.config import (  # noqa: E402
    PAIR_SEARCH_BOUNDED,
    PAIR_SEARCH_EXHAUSTIVE,
    STATE_BASE_ONLY,
    STATE_RESIDUAL,
    STATE_REUSE1,
    STATE_REUSE2,
    TARGET_CONTEXT_POSITIVE,
    TARGET_POSITIVE,
    TEACHER_SEARCH_FULL_HISTORY,
    V8Config,
    V8TeacherConfig,
)
from compose.v8.generate import (  # noqa: E402
    GenerationEngine,
    experts_from_route,
    route_key,
)
from compose.v8.metric_adapter import TaskMetricAdapter  # noqa: E402
from compose.v8.pool import MultiKeyExpertPool  # noqa: E402
from compose.v8.routing import MultiKeyRouter  # noqa: E402
from compose.v8.teacher import AnswerSupervisedTeacher  # noqa: E402


#: Paths the audited V7 reproduction used; identical here so nothing about the
#: model, the vision tower or the image root changes between V7 and V8 numbers.
PYTHON = "/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"
MODEL = "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
VISION = "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
PROJECTOR = os.path.join(MODEL, "mm_projector.bin")
IMAGES = "/data/dataset/zhaozhuofan/UCIT/datasets"

DEFAULT_FORMAL_ROOT = (
    "/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_gpu01_cached_query_formal_20260903"
)
DEFAULT_DIAGNOSTIC_ROOT = (
    "/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_final_pool_pair_upper_val256_20260907"
)
DEFAULT_QUERY_CACHE = (
    "/data/ckpt/zhaozhuofan/Hyper-LLaVA-runs/v7_fixed_query_cache_gpu01_20260903"
)

TASK_NAMES = {0: "ImageNet-R", 1: "ArxivQA", 2: "VizWiz",
              3: "IconQA", 4: "CLEVR-Math", 5: "Flickr30k"}

#: Artefact schema versions.  The audit trail is append-only in spirit: a reader
#: must be able to tell which teacher semantics produced a file, because the
#: pre-oracle campaign's ``teacher_result.json`` has the same shape but a
#: different meaning (it searched only the recalled Top-M).
#:
#: * ``recall.json`` v1 -> v2: adds ``teacher_visible_expert_ids`` and marks the
#:   existing ``visible_expert_ids`` as the active-pool field it always was.
#: * ``teacher_result.json`` v1 -> v2: adds ``teacher_search_mode``,
#:   ``historical_experts_visible`` and the per-record
#:   ``historical_experts_tested`` / ``context_positive_experts``.
#:
#: Old files are never rewritten in place; the reader tolerates a v1 file by
#: treating a missing ``teacher_search_mode`` as the pre-oracle semantics.
RECALL_SCHEMA_VERSION = 2
TEACHER_RESULT_SCHEMA_VERSION = 2
#: The search mode string written by the pre-oracle campaign, used only when
#: *reading* a v1 artefact.
LEGACY_TEACHER_SEARCH_MODE = "bounded_recall"


def _read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def _sha256(path: str | Path) -> Optional[str]:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _physical_gpu_index(device: str) -> Optional[int]:
    """Which physical GPU a ``cuda:N`` device string really refers to.

    The machine is shared, so the scheduler works on physical indices while the
    process works on the *visible* ones.  Getting this wrong would make the
    runner wait for the wrong card -- or wait forever on a card already free.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        first = visible.split(",")[0].strip()
        return int(first) if first.isdigit() else None
    if device.startswith("cuda"):
        _, _, index = device.partition(":")
        return int(index) if index.isdigit() else 0
    return None


def _free_memory_mib(index: int) -> Optional[int]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
             "-i", str(index)],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        return int(output[0]) if output else None
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def wait_for_free_memory(
    index: Optional[int],
    minimum_mib: int,
    timeout_seconds: float,
    poll_seconds: float = 20.0,
    log=print,
) -> Dict[str, Any]:
    """Block until the target GPU can hold the model, as the V7 harness does.

    A bf16 Llama-7B plus the 23-expert pool needs roughly 16 GiB.  Launching
    into a card that cannot hold it produces a CUDA OOM several minutes into
    model loading, which on a shared machine is an easy way to lose an hour, so
    the check happens before the load rather than around it.
    """
    if index is None or timeout_seconds <= 0:
        return {"waited": False, "reason": "disabled"}
    deadline = time.time() + float(timeout_seconds)
    waited = 0.0
    while True:
        free = _free_memory_mib(index)
        if free is None:
            return {"waited": waited > 0, "reason": "nvidia-smi unavailable",
                    "waited_seconds": waited}
        if free >= int(minimum_mib):
            return {"waited": waited > 0, "reason": "free memory sufficient",
                    "free_mib": free, "waited_seconds": waited}
        if time.time() >= deadline:
            return {"waited": True, "reason": "timeout", "free_mib": free,
                    "waited_seconds": waited}
        log("waiting for GPU {}: {} MiB free < {} MiB required".format(
            index, free, minimum_mib))
        sleep_for = min(poll_seconds, max(1.0, deadline - time.time()))
        time.sleep(sleep_for)
        waited += sleep_for


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
            cwd=str(REPO),
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class AppendOnlyJsonl:
    """Append-only keyed store, used for the NLL cache and its audit trail."""

    def __init__(self, path: str | Path, key_fields: Sequence[str]) -> None:
        self.path = Path(path)
        self.key_fields = tuple(key_fields)
        self._rows: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        if self.path.is_file():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        self._rows[self._key(row)] = row
        self._handle = None

    def _key(self, row: Mapping[str, Any]) -> Tuple[Any, ...]:
        return tuple(row[field] for field in self.key_fields)

    def __len__(self) -> int:
        return len(self._rows)

    def __bool__(self) -> bool:
        """See ``compose.v8.generate.RouteGenerationCache.__bool__``."""
        return True

    def rows(self) -> List[Dict[str, Any]]:
        return list(self._rows.values())

    def get(self, **fields: Any) -> Optional[Dict[str, Any]]:
        return self._rows.get(tuple(fields[field] for field in self.key_fields))

    def put(self, row: Mapping[str, Any]) -> None:
        key = self._key(row)
        if key in self._rows:
            return
        self._rows[key] = dict(row)
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        self._handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
        self._handle.flush()


# ----------------------------------------------------------------------
# NLL
# ----------------------------------------------------------------------


class AnswerNLLScorer:
    """Teacher-forced answer NLL for arbitrary per-sample expert sets.

    The supervision contract is the training one -- ``LazySupervisedDataset``
    plus ``DataCollatorForSupervisedDataset``, the same objects
    ``compose/eval/nll_eval.py`` uses -- so a V8 NLL is comparable with the V7
    pair diagnostic instead of being a lookalike.

    A V7 diagnostic may pre-seed this cache.  Seeding is only accepted when the
    seeded value is later reproduced by a live recomputation on a random spot
    check (see :meth:`verify_seed`), because a silently mismatched NLL would
    corrupt the tie-break among solved experts without changing any accuracy.
    """

    def __init__(
        self,
        bundle,
        question_file: str,
        image_folder: str,
        device: str,
        cache_path: str | Path,
    ) -> None:
        from compose.train.arguments import DataArguments
        from compose.train.data import (
            DataCollatorForSupervisedDataset,
            LazySupervisedDataset,
        )
        from llava import conversation as conversation_lib

        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
        data_args = DataArguments(
            data_path=question_file,
            image_folder=image_folder,
            image_aspect_ratio="pad",
        )
        data_args.image_processor = bundle.image_processor
        data_args.is_multimodal = True
        data_args.mm_use_im_start_end = False
        self.dataset = LazySupervisedDataset(question_file, bundle.tokenizer, data_args)
        self.collator = DataCollatorForSupervisedDataset(bundle.tokenizer)
        self.bundle = bundle
        self.device = str(device)
        self.index_by_id: Dict[str, int] = {}
        for index, record in enumerate(self.dataset.records):
            self.index_by_id[str(record.get("id", record.get("question_id")))] = index
        self.cache = AppendOnlyJsonl(cache_path, ("sample_id", "route"))
        self.computed = 0
        self.reused = 0

    def seed(self, sample_id: str, experts: Sequence[int], value: float) -> None:
        self.cache.put({
            "sample_id": str(sample_id),
            "route": route_key(experts),
            "mean_answer_nll": float(value),
            "source": "v7_diagnostic_seed",
        })

    def _forward(self, sample_id: str, experts: Sequence[int]) -> Dict[str, Any]:
        record_index = self.index_by_id[sample_id]
        batch = self.collator([self.dataset[record_index]])
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        images = batch["images"].to(self.device, dtype=torch.bfloat16)
        expanded = self.bundle.model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=labels,
            images=images,
        )
        prepared_ids = expanded[0]
        prepared_attention_mask = expanded[2]
        prepared_inputs_embeds = expanded[4]
        prepared_labels = expanded[5]
        supervised = int(supervised_token_mask(prepared_labels).sum().item())
        if supervised <= 0:
            raise ValueError("record {} has zero supervised answer tokens".format(sample_id))
        ids = [int(value) for value in experts]
        padded_ids, padded_gates = pad_selection(tuple(ids), tuple(1.0 for _ in ids))
        selection = ComposeSelection(
            torch.tensor([padded_ids], dtype=torch.long),
            torch.tensor([padded_gates], dtype=torch.float32),
        )
        with torch.inference_mode(), self.bundle.expert_pool.manager.selection_context(selection):
            outputs = self.bundle.model(
                input_ids=prepared_ids,
                inputs_embeds=prepared_inputs_embeds,
                attention_mask=prepared_attention_mask,
                labels=prepared_labels,
                return_dict=True,
            )
        nll = teacher_forcing_token_nll(outputs.logits, prepared_labels)
        return {
            "sample_id": str(sample_id),
            "route": route_key(experts),
            "mean_answer_nll": float(nll),
            "supervised_token_count": supervised,
            "source": "v8_live",
        }

    def __call__(self, selection: Mapping[str, Sequence[int]]) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for sample_id, experts in selection.items():
            row = self.cache.get(sample_id=str(sample_id), route=route_key(experts))
            if row is None:
                row = self._forward(str(sample_id), experts)
                self.cache.put(row)
                self.computed += 1
            else:
                self.reused += 1
            values[str(sample_id)] = float(row["mean_answer_nll"])
        return values

    def verify_seed(self, trials: int = 8, seed: int = 0) -> Dict[str, Any]:
        """Recompute a random sample of seeded rows and compare.

        Every trial is reported, not just the worst, because the two failure
        shapes need different fixes: a uniform offset means the seed came from a
        different composition rule, while a single outlier means one sample's
        record or image differs.
        """
        seeded = [row for row in self.cache.rows()
                  if row.get("source") == "v7_diagnostic_seed"]
        if not seeded:
            return {"seeded": 0, "verified": 0, "max_abs_diff": None,
                    "trials": [], "status": "NO_SEED"}
        rng = random.Random(seed)
        chosen = rng.sample(seeded, min(int(trials), len(seeded)))
        worst = 0.0
        details = []
        for row in chosen:
            experts = experts_from_route(row["route"])
            live = self._forward(row["sample_id"], experts)
            seeded_value = float(row["mean_answer_nll"])
            live_value = float(live["mean_answer_nll"])
            worst = max(worst, abs(live_value - seeded_value))
            details.append({
                "sample_id": row["sample_id"],
                "route": row["route"],
                "seeded_nll": seeded_value,
                "live_nll": live_value,
                "abs_diff": abs(live_value - seeded_value),
                "supervised_token_count": live["supervised_token_count"],
            })
        return {
            "seeded": len(seeded),
            "verified": len(chosen),
            "max_abs_diff": worst,
            "trials": details,
            "status": "MATCH" if worst < 1e-4 else "MISMATCH",
        }


# ----------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------


class V8TaskRun:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.task = int(args.task)
        self.root = Path(args.root)
        self.out = self.root / "task{}".format(self.task)
        self.out.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out / "run.log"
        self.started = time.time()
        self.timings: Dict[str, float] = {}

    # -- helpers --------------------------------------------------------
    def log(self, message: str) -> None:
        line = "[{:8.1f}s] {}".format(time.time() - self.started, message)
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _timed(self, label: str, fn):
        begin = time.time()
        result = fn()
        self.timings[label] = self.timings.get(label, 0.0) + (time.time() - begin)
        return result

    # -- setup ----------------------------------------------------------
    def setup(self) -> None:
        formal = Path(self.args.formal_root)
        checkpoint_dir = Path(self.args.checkpoint_dir or (formal / "task5" / "committed"))
        self.checkpoint_dir = checkpoint_dir
        question_file = formal / "task{}".format(self.task) / "data" / "val_full.json"
        self.question_file = question_file
        if not question_file.is_file():
            raise FileNotFoundError("validation file missing: {}".format(question_file))

        self.records = _read_json(question_file)
        self.records_by_id = {
            str(record.get("id", record.get("question_id"))): record
            for record in self.records
        }
        if self.args.limit:
            keep = sorted(self.records_by_id)[: int(self.args.limit)]
            self.records_by_id = {key: self.records_by_id[key] for key in keep}
        self.sample_ids = sorted(self.records_by_id)
        self.log("validation samples: {}".format(len(self.sample_ids)))

        # Contamination guard: the V8-A teacher may see validation answers, and
        # only validation answers.  The test_3000 split lives elsewhere; this
        # check makes an accidental switch loud instead of silent.
        if "test" in question_file.name:
            raise ValueError("refusing to run the teacher over a test split")

        query_path = (Path(self.args.query_cache) / "query_cache"
                      / "task{}".format(self.task) / "val" / "queries.pt")
        self.query_path = query_path
        payload = torch.load(query_path, map_location="cpu")
        cache_ids = [str(value) for value in payload["sample_ids"]]
        index = {sample_id: row for row, sample_id in enumerate(cache_ids)}
        missing = [sample_id for sample_id in self.sample_ids if sample_id not in index]
        if missing:
            raise ValueError(
                "query cache {} has no query for {}".format(query_path, missing[:4])
            )
        self.queries = {
            sample_id: payload["queries"][index[sample_id]].float()
            for sample_id in self.sample_ids
        }
        self.query_contract_hash = payload.get("contract_hash")
        self.log("query cache: {} ({} queries, contract {})".format(
            query_path.name, payload["queries"].shape[0], self.query_contract_hash))

        v7_state = torch.load(checkpoint_dir / "v7_keys.pt", map_location="cpu")
        manifest = _read_json(checkpoint_dir / "compose_experts.json")
        self.pool = MultiKeyExpertPool.load_v7_pool(v7_state, manifest, frozen=True)
        self.pool.validate()
        self.pool_audit = self.pool.audit()
        self.expert_ids = self.pool.active_expert_ids()
        self.log("pool: {} experts, {} keys ({})".format(
            len(self.expert_ids), self.pool_audit["num_keys"],
            self.pool_audit["key_lifecycle_counts"]))
        if self.pool_audit["num_alias_keys"] != 0:
            raise ValueError("the migrated V7 pool must not contain alias keys")

        self.bundle = self._timed("load_model", lambda: load_compose_model(
            model_path=MODEL,
            checkpoint_dir=str(checkpoint_dir),
            vision_tower=VISION,
            projector_path=PROJECTOR,
            expert_id=None,
            device=self.args.device,
            dtype=torch.bfloat16,
            model_max_length=2048,
        ))
        self.log("model loaded on {} (frozen base + {} experts)".format(
            self.args.device, len(self.expert_ids)))

        self.metric = TaskMetricAdapter()
        self.spec = self.metric.require_decomposable(self.task)
        self.log("metric: {} solved_value={}".format(
            self.spec.metric_name, self.spec.solved_value))

    # -- recall ---------------------------------------------------------
    def _excluded_experts(self) -> List[int]:
        """Experts the current task must not be credited with reusing.

        In the real continual setting the pool at task ``t`` holds the experts of
        tasks ``< t``: the task's own experts do not exist yet.  Asking which of
        this task's samples the *historical* pool can already solve against a
        pool that happens to contain the task's own experts would credit V8 with
        reuse it never had the chance to perform, so ``--history-only`` drops
        them from the routing candidates (their keys are excluded, not deleted).
        Task 0 has no history at all and becomes BaseOnly throughout.
        """
        if not self.args.history_only:
            return []
        return sorted(
            expert_id for expert_id in self.expert_ids
            if int(self.pool.expert_records[expert_id]["origin_task"]) >= self.task
        )

    def recall(self) -> None:
        top_m = int(self.args.recall_top_m)
        excluded = self._excluded_experts()
        self.excluded_expert_ids = excluded
        # The capability-search universe: the experts that exist, are alive
        # (``active_expert_ids`` drops pruned ones) and are in continual scope at
        # this task.  It is *not* the router's Top-M -- that is the whole point
        # of DECISION-1, and the teacher hard-fails if the two are conflated.
        self.visible_experts = sorted(set(self.expert_ids) - set(excluded))
        if len(excluded) >= len(self.expert_ids):
            # No historical pool exists yet (task 0).  V8's policy is BaseOnly
            # for every sample by construction, so the router is never asked to
            # rank an empty candidate set.
            self.recall_map = {sample_id: [] for sample_id in self.sample_ids}
            self.full_order = {sample_id: [] for sample_id in self.sample_ids}
            self.route_rows = []
            self.log("recall: task {} has no historical experts ({} excluded); "
                     "every sample is BaseOnly".format(self.task, len(excluded)))
        else:
            router = MultiKeyRouter(V8Config().routing)
            queries = torch.stack(
                [self.queries[sample_id] for sample_id in self.sample_ids])
            result = router(queries, self.pool, excluded_experts=excluded)
            # The router's own Top-K is fixed at V7's budget of 2.  V8's teacher
            # needs the whole ordering so the recall curve at K = 1, 2, 4, 8 can
            # be read off a single pass, so the ordering is reconstructed from
            # the max-aggregated per-expert scores rather than the Top-2 cut.
            per_expert = result.per_expert_scores
            order = torch.argsort(per_expert, dim=-1, descending=True, stable=True)
            pool_ids = result.pool_expert_ids.tolist()
            full_order = {
                sample_id: [int(pool_ids[int(col)]) for col in order[row]]
                for row, sample_id in enumerate(self.sample_ids)
            }
            self.recall_map = {
                sample_id: full_order[sample_id][:top_m]
                for sample_id in self.sample_ids
            }
            self.full_order = full_order
            self.route_rows = result.as_rows(self.sample_ids)
            self.log("recall: Top-{} over {} candidate experts{}".format(
                top_m, len(pool_ids),
                " ({} excluded: history-only)".format(len(excluded))
                if excluded else ""))
        _write_json(self.out / "recall.json", {
            "task_id": self.task,
            "schema_version": RECALL_SCHEMA_VERSION,
            "recall_top_m": top_m,
            "history_only": bool(self.args.history_only),
            "excluded_expert_ids": excluded,
            # NOTE (schema trap, kept for backward compatibility): this field has
            # always held the whole *active pool*, not the in-scope set -- the
            # scope is ``excluded_expert_ids``.  The unambiguous name is
            # ``teacher_visible_expert_ids`` below; both are written so readers
            # written against either revision agree.
            "visible_expert_ids": self.expert_ids,
            "teacher_visible_expert_ids": list(self.visible_experts),
            "router_top_m_diagnostic": True,
            "query_contract_hash": self.query_contract_hash,
            "recall": self.recall_map,
            "full_order": self.full_order,
            "routing_rows": self.route_rows,
        })

    # -- scoring --------------------------------------------------------
    def _diagnostic_dir(self) -> Path:
        return Path(self.args.diagnostic_root) / "task{}".format(self.task)

    def seed_pair_nll(self, nll: AnswerNLLScorer) -> Dict[str, Any]:
        """Seed the NLL cache from the frozen V7 pair diagnostic.

        The diagnostic scored ``pair_{a:02d}_{b:02d}`` over exactly the visible
        expert ids, with gates ``[1.0, 1.0]`` and ``normalization="none"``, on
        the same checkpoint.  That is the same forward pass V8's pair route
        performs, so reusing it saves hours of duplicated GPU time.  It is only
        trusted because :meth:`AnswerNLLScorer.verify_seed` recomputes a random
        spot check against the live model and the run aborts on a mismatch.
        """
        directory = self._diagnostic_dir()
        shards = sorted(directory.glob("nll.json.rank*"))
        if not shards:
            self.log("no V7 pair diagnostic at {}; pairs computed live".format(directory))
            return {"seeded": 0, "shards": 0}
        merged: Dict[str, Dict[str, Any]] = {}
        for shard in shards:
            for sample_id, routes in _read_json(shard).items():
                merged.setdefault(str(sample_id), {}).update(routes)
        seeded = 0
        for sample_id, routes in merged.items():
            if sample_id not in self.records_by_id:
                continue
            for route, payload in routes.items():
                if not route.startswith("pair_"):
                    continue
                # ``pair_XX_YY`` is the diagnostic's own naming, not a
                # ``route_key`` string -- here dropping the first field is
                # correct, because the prefix is a separate "pair" segment.
                experts = [int(value) for value in route.split("_")[1:]]
                nll.seed(sample_id, experts, float(payload["mean_answer_nll"]))
                seeded += 1
        return {"seeded": seeded, "shards": len(shards)}

    def make_metric_scorer(self, engine: GenerationEngine):
        """``selection -> official per-sample metric`` with a generation cache."""

        def scorer(selection: Mapping[str, Sequence[int]]) -> Dict[str, float]:
            answers = engine.generate_route(selection, self.records_by_id)
            values: Dict[str, float] = {}
            for sample_id, text in answers.items():
                ground_truth = answer_text(self.records_by_id[sample_id])
                values[sample_id] = float(
                    self.metric.sample_value(self.task, text, ground_truth)
                )
            return values

        return scorer

    # -- teacher --------------------------------------------------------
    def run_teacher(self) -> None:
        engine = GenerationEngine(
            self.bundle,
            image_folder=IMAGES,
            device=self.args.device,
            max_new_tokens=int(self.args.max_new_tokens),
            cache_path=self.out / "generation_cache.jsonl",
        )
        nll = AnswerNLLScorer(
            self.bundle,
            question_file=str(self.question_file),
            image_folder=IMAGES,
            device=self.args.device,
            cache_path=self.out / "nll_cache.jsonl",
        )
        self.nll_scorer = nll
        self.seed_report = self.seed_pair_nll(nll)
        self.log("pair NLL seed: {}".format(self.seed_report))

        teacher_config = V8TeacherConfig(
            historical_top_m=int(self.args.recall_top_m),
            pair_top_k_single=int(self.args.shortlist_ks),
            pair_search_mode=str(self.args.pair_search_mode),
        )
        teacher = AnswerSupervisedTeacher(self.metric, teacher_config)

        # The teacher is a Capability Discovery Oracle: every historical expert
        # visible at this task is scored as a single, independently of whether
        # the current keys happen to recall it.  ``visible_experts`` is passed
        # explicitly (it is a required argument) so no caller can silently fall
        # back to a recall-limited search.
        result = teacher.run(
            task_id=self.task,
            sample_ids=self.sample_ids,
            visible_experts=self.visible_experts,
            recall_map=self.recall_map,
            scorer=self.make_metric_scorer(engine),
            nll_scorer=nll,
            pair_min_metric_gain=self.args.pair_min_metric_gain,
            progress=self.log,
        )
        self.teacher_result = result
        self.generated = engine.generated_count
        self.cache_hits = engine.cache_hits
        self.nll_computed = nll.computed
        self.nll_reused = nll.reused
        _write_json(self.out / "teacher_result.json",
                    teacher_result_payload(result, teacher_config))
        self.log("teacher: mode={} visible={} {} states={} generated={} nll_live={}".format(
            result.search_mode, len(result.visible_experts), len(result.records),
            result.state_counts(), self.generated, self.nll_computed))

    def _diagnostic_covers_this_run(self, diagnostic_metric: Mapping[str, Any]) -> bool:
        """Is the diagnostic's V7 metric measured on *this* run's samples?

        Checked against the diagnostic's own prediction file rather than
        inferred from the sample count, so the answer is about the sample set and
        not about two integers happening to agree.  If the file cannot be read
        the comparison is refused -- an unverifiable baseline is not a baseline.
        """
        ids = _prediction_ids(diagnostic_metric.get("prediction_file") or "")
        if ids is None:
            return False
        return ids == set(self.sample_ids)

    # -- policy ---------------------------------------------------------
    def write_policy(self) -> None:
        """Assemble the minimal-cardinality policy from already-generated text.

        Every route the policy names was generated while the teacher was
        deciding, so this step is I/O: it re-reads the append-only cache and
        refuses if any route is missing.  Nothing is regenerated, which is what
        guarantees the reported metric and the teacher's decisions describe the
        same answers.
        """
        cache = {}
        with (self.out / "generation_cache.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    cache[(row["route"], row["sample_id"])] = row["text"]

        self.policy: Dict[str, List[int]] = {}
        self.policy_state: Dict[str, str] = {}
        missing: List[str] = []
        with (self.out / "v8_policy_answers.jsonl").open("w", encoding="utf-8") as handle:
            for record in self.teacher_result.records:
                sample_id = record.sample_id
                if record.state == STATE_RESIDUAL:
                    experts = list(record.residual_context)
                    state = STATE_RESIDUAL
                else:
                    experts = list(record.selected_experts)
                    state = record.state
                key = (route_key(experts), sample_id)
                if key not in cache:
                    missing.append("{}@{}".format(sample_id, key[0]))
                    continue
                self.policy[sample_id] = experts
                self.policy_state[sample_id] = state
                handle.write(json.dumps({
                    "question_id": sample_id,
                    "prompt": question_text(self.records_by_id[sample_id]),
                    "text": cache[key],
                    "model_id": "compose",
                    "metadata": {
                        "v8_state": state,
                        "v8_experts": experts,
                        "v8_route": key[0],
                    },
                }, ensure_ascii=False) + "\n")
        if missing:
            raise RuntimeError(
                "{} policy routes were never generated: {}".format(len(missing), missing[:8])
            )
        _write_json(self.out / "v8_policy.json", {
            "task_id": self.task,
            "states": {
                state: sum(1 for value in self.policy_state.values() if value == state)
                for state in (STATE_BASE_ONLY, STATE_REUSE1, STATE_REUSE2, STATE_RESIDUAL)
            },
            "policy": self.policy,
        })
        self.log("policy written: {} samples".format(len(self.policy)))

    def official_score(self) -> None:
        annotation = [
            {
                "question_id": sample_id,
                "answer": answer_text(self.records_by_id[sample_id]),
                "image": self.records_by_id[sample_id].get("image"),
            }
            for sample_id in self.sample_ids
        ]
        annotation_path = self.out / "questions.json"
        _write_json(annotation_path, annotation)
        self.official = _score_answers(
            self.out,
            self.task,
            self.task,
            self.out / "v8_policy_answers.jsonl",
            annotation_file=str(annotation_path),
        )
        self.log("official V8 policy metric: {}".format(self.official))

    # -- deployable Top-2 inference (PART 8) ----------------------------
    def top2_inference(self) -> Dict[str, Any]:
        """The *deployable* number, next to the teacher's oracle number.

        ``official_score`` replays the teacher's per-sample route, and that route
        is chosen with the validation answer in hand and may hold 0, 1 or 2
        experts.  It is an **oracle** metric: no deployed system can pick
        cardinality per sample from the ground truth.  This method measures what
        the committed pool actually delivers -- every held-out sample goes

            input -> frozen fixed query -> committed multi-key pool
                  -> max-per-expert -> fixed Top-2 distinct experts -> generation

        with no ground truth, no teacher, no NLL and no weighting anywhere in the
        path.  The two numbers answer different questions and must never be
        reported as one; the analysis records both under names that say which is
        which.
        """
        router = MultiKeyRouter(V8Config().routing)
        queries = torch.stack([self.queries[sample_id] for sample_id in self.sample_ids])
        result = router(queries, self.pool, excluded_experts=self.excluded_expert_ids)
        selection = {
            sample_id: [int(value) for value in result.expert_ids[row].tolist()]
            for row, sample_id in enumerate(self.sample_ids)
        }
        engine = GenerationEngine(
            self.bundle,
            image_folder=IMAGES,
            device=self.args.device,
            max_new_tokens=int(self.args.max_new_tokens),
            cache_path=self.out / "generation_cache.jsonl",
        )
        answers = engine.generate_route(selection, self.records_by_id)
        directory = self.out / "top2_inference"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "answers.jsonl").open("w", encoding="utf-8") as handle:
            for sample_id in self.sample_ids:
                handle.write(json.dumps({
                    "question_id": sample_id,
                    "prompt": question_text(self.records_by_id[sample_id]),
                    "text": answers[sample_id],
                    "model_id": "compose",
                    "metadata": {
                        "v8_state": _cardinality_state(selection[sample_id]),
                        "v8_experts": selection[sample_id],
                        "v8_route": route_key(selection[sample_id]),
                    },
                }, ensure_ascii=False) + "\n")
        annotation_path = directory / "questions.json"
        _write_json(annotation_path, [
            {
                "question_id": sample_id,
                "answer": answer_text(self.records_by_id[sample_id]),
                "image": self.records_by_id[sample_id].get("image"),
            }
            for sample_id in self.sample_ids
        ])
        official = _score_answers(
            directory,
            self.task,
            self.task,
            directory / "answers.jsonl",
            annotation_file=str(annotation_path),
        )
        payload = {
            "metric": official,
            "value": float(official["value"]),
            "num_experts_per_sample": sorted(
                {len(experts) for experts in selection.values()}
            ),
            "always_two_experts": all(
                len(experts) == 2 for experts in selection.values()
            ),
            "generated": engine.generated_count,
            "cache_hits": engine.cache_hits,
            "inference_inputs": [
                "frozen fixed query", "committed origin/alias keys",
                "max-per-expert aggregation", "fixed Top-2 distinct experts",
            ],
            "forbidden_inputs": ["ground_truth", "teacher", "answer_nll", "task_label"],
        }
        _write_json(directory / "metric.json", payload)
        self.top2 = payload
        self.log("actual fixed Top-2 inference metric: {}".format(payload["value"]))
        return payload

    # -- analysis -------------------------------------------------------
    def analyse(self) -> None:
        result = self.teacher_result
        states = result.state_counts()

        # Which experts actually *solved* a sample (target positive): this is the
        # reference set for the recall curves, and it is a superset of the
        # selected set because the teacher stops at the first solved single.
        solving: Dict[str, List[int]] = {}
        for record in result.records:
            solving[record.sample_id] = [
                int(expert_id)
                for expert_id, target in record.key_targets.items()
                if target == TARGET_POSITIVE
            ]
        self.solving = solving

        ks = (1, 2, 4, 8)
        recall_curve = teacher_expert_recall_at_k(solving, self.recall_map, ks=ks)
        pool_curve = full_pool_oracle_recall_at_k(solving, self.full_order, ks=ks)

        # V7's comparator comes from a *different* artefact -- the diagnostic
        # root's own run over its own split.  It is therefore only a comparator
        # when this run covers the same samples: a `--limit 32` run that quoted
        # the diagnostic's full-split 67.97 as its V7 baseline was comparing 32
        # V8 samples against 256 V7 samples, and `gap_closed` below would have
        # turned that into a published gap.  The sample sets are checked, not
        # assumed, and a mismatch routes the number to a reference field instead
        # of to the baseline.
        v7_metric: Optional[float] = None
        v7_reference: Optional[float] = None
        v7_comparable = False
        nll_oracle_metric: Optional[float] = None
        nll_oracle_comparable = False
        diagnostic_complete = (
            Path(self.args.diagnostic_root) / "generation_accuracy" / "COMPLETE.json"
        )
        if diagnostic_complete.is_file():
            for entry in _read_json(diagnostic_complete).get("tasks", []):
                if int(entry["task"]) != self.task:
                    continue
                metrics = entry.get("metrics", {})
                actual = metrics.get("actual", {})
                oracle = metrics.get("oracle", {})
                # both metrics are measured on the *same* split, so one
                # comparison settles both
                v7_comparable = bool(
                    actual) and self._diagnostic_covers_this_run(actual)
                if "actual" in metrics:
                    value = float(metrics["actual"]["value"])
                    if v7_comparable:
                        v7_metric = value
                    else:
                        v7_reference = value
                if "oracle" in metrics:
                    nll_oracle_comparable = v7_comparable
                    if v7_comparable:
                        nll_oracle_metric = float(metrics["oracle"]["value"])
                if not v7_comparable and (actual or oracle):
                    self.log(
                        "V7 comparator NOT comparable: diagnostic split has {} "
                        "samples, this run has {} -- V7 {} is reported as a "
                        "reference, not as this run's baseline".format(
                            actual.get("samples"), len(self.sample_ids),
                            actual.get("value"),
                        )
                    )

        v8_metric = float(self.official["value"])
        teacher_positives = sum(
            1 for record in result.records if record.state != STATE_RESIDUAL
        )
        ceiling = nll_oracle_metric if nll_oracle_metric is not None else v8_metric
        # No comparable V7 baseline -> no gap.  ``gap_closed`` against a number
        # measured on a different sample set would be arithmetic on a fiction,
        # so the honest answer is None, which is what a reader sees.
        gap = (gap_closed(v7_metric, v8_metric, ceiling)
               if v7_metric is not None else None)
        diagnosis = diagnose(
            full_pool_oracle_solves=teacher_positives,
            teacher_positives=teacher_positives,
            candidate_recall=float(recall_curve.get(int(self.args.recall_top_m), 0.0)),
            v7_metric=v7_metric,
            v8_metric=v8_metric,
            samples=len(result.records),
        )

        # How well does the teacher's own correctness agree with the official
        # scorer?  The teacher uses the documented per-sample exact-match rule,
        # the official scorer is a subprocess; if they disagreed, one of the two
        # numbers in this report would be describing a different run.
        teacher_correct = sum(1 for record in result.records if record.solved)
        official_correct = int(round(v8_metric / 100.0 * len(result.records)))
        reconciliation = {
            "teacher_solved_samples": teacher_correct,
            "official_correct_samples": official_correct,
            "difference": official_correct - teacher_correct,
            "note": (
                "teacher counts a sample solved when base, a single or a pair met "
                "the metric; the policy then replays exactly that route, so the "
                "two counts must agree"
            ),
        }

        cardinality: Dict[str, int] = {}
        for record in result.records:
            if record.state == STATE_RESIDUAL:
                key = "residual:{}".format(len(record.residual_context))
            else:
                key = "{}".format(len(record.selected_experts))
            cardinality[key] = cardinality.get(key, 0) + 1

        top2 = getattr(self, "top2", None)

        analysis = {
            "task_id": self.task,
            "task_name": TASK_NAMES.get(self.task),
            "samples": len(result.records),
            "v8_policy_metric": v8_metric,
            # PART 8: the two metrics answer different questions and are named so
            # that neither can be quoted as the other.
            **metric_report(v8_metric, top2),
            "teacher_search_mode": result.search_mode,
            "historical_experts_visible": list(result.visible_experts),
            "teacher_coverage": result.coverage_report(),
            # Measured on *this run's* sample set, or None when the diagnostic
            # covers a different one.  A non-comparable number is kept in
            # ``v7_actual_route_metric_reference`` with its scope named, so the
            # value is still available without being quotable as this run's
            # baseline.
            "v7_actual_route_metric": v7_metric,
            "v7_actual_route_metric_comparable": v7_metric is not None,
            "v7_actual_route_metric_reference": v7_reference,
            "v7_actual_route_metric_reference_note": (
                None if v7_reference is None else (
                    "V7's diagnostic metric for this task, measured on the "
                    "diagnostic root's own split -- NOT on this run's {} samples, "
                    "so it is not a baseline for any number in this file".format(
                        len(result.records))
                )
            ),
            "v7_nll_oracle_pair_metric": nll_oracle_metric,
            "v8_teacher_positive_samples": teacher_positives,
            "v8_teacher_positive_rate": teacher_positives / len(result.records),
            "full_pool_oracle_solves": teacher_positives,
            "full_pool_oracle_solves_note": (
                "the samples the Capability Discovery Oracle solved with a single "
                "expert from the full visible historical pool (base-first), so "
                "this is a capability statement about the pool and is not limited "
                "by what the keys recall.  It is still not an exhaustive claim "
                "about *pairs* when pair_search_mode is bounded"
            ),
            "router_top_m_diagnostic_only": {
                "recall_top_m": int(self.args.recall_top_m),
                "teacher_expert_recall_at_k": {
                    str(k): v for k, v in recall_curve.items()
                },
                "note": (
                    "diagnostic: how often the committed keys rank a capable "
                    "expert inside Top-K.  It is NOT an input to the teacher's "
                    "search, the state assignment or the alias support"
                ),
            },
            "states": states,
            "state_rates": result.state_rate(),
            "policy_cardinality_histogram": dict(sorted(cardinality.items())),
            "teacher_expert_recall_at_k": {str(k): v for k, v in recall_curve.items()},
            "full_pool_oracle_recall_at_k": {str(k): v for k, v in pool_curve.items()},
            "gap_closed": gap,
            "diagnosis": diagnosis,
            "reconciliation": reconciliation,
            "pool_audit": self.pool_audit,
            "pair_nll_seed": self.seed_report,
            "timings_seconds": dict(self.timings),
            "generation": {
                "generated": self.generated,
                "cache_hits": self.cache_hits,
                "nll_live": self.nll_computed,
                "nll_reused": self.nll_reused,
            },
            "solution_counts": result.solved_expert_histogram(),
            "marginal_summary": result.marginal_summary(),
            "residual_context": result.residual_context_by_sample(),
        }
        self.analysis = analysis
        _write_json(self.out / "analysis.json", analysis)
        self.log("analysis: states={} cardinality={} V8={} V7={} recall@M={:.4f}".format(
            states, cardinality, v8_metric,
            v7_metric if v7_metric is not None else "none (reference {})".format(
                v7_reference),
            float(recall_curve.get(int(self.args.recall_top_m), 0.0))))

    # -- driver ---------------------------------------------------------
    def write_config(self) -> None:
        _write_json(self.out / "run_config.json", {
            "task_id": self.task,
            "task_name": TASK_NAMES.get(self.task),
            "root": str(self.root),
            "checkpoint_dir": str(self.checkpoint_dir),
            "question_file": str(self.question_file),
            "query_cache": str(self.query_path),
            "query_contract_hash": self.query_contract_hash,
            "diagnostic_root": str(self.args.diagnostic_root),
            "recall_top_m": int(self.args.recall_top_m),
            "history_only": bool(self.args.history_only),
            # Resolved here rather than read off ``self``: ``write_config`` runs
            # before ``recall`` sets that attribute, and the previous
            # ``getattr(self, "excluded_expert_ids", [])`` therefore wrote an
            # empty exclusion list into the provenance file of *every*
            # history-only run -- the same "scope read from the wrong field"
            # trap that §16.1's audit fell into, in a second artefact.
            # ``_excluded_experts`` is pure, so calling it is safe this early.
            "excluded_expert_ids": list(self._excluded_experts()),
            "shortlist_ks": int(self.args.shortlist_ks),
            "max_new_tokens": int(self.args.max_new_tokens),
            "pair_min_metric_gain": self.args.pair_min_metric_gain,
            "device": self.args.device,
            "limit": self.args.limit,
            "gpu_wait": getattr(self, "gpu_wait", None),
            "git_commit": _git_commit(),
            "frozen_hashes": {
                "v7_keys": _sha256(self.checkpoint_dir / "v7_keys.pt"),
                "compose_experts_json": _sha256(self.checkpoint_dir / "compose_experts.json"),
                "compose_experts_bin": _sha256(self.checkpoint_dir / "compose_experts.bin"),
            },
            "trainable": "none (V8-A is inference only; the pool stays frozen)",
            "split": "validation",
        })

    def run(self) -> Dict[str, Any]:
        # ``write_config`` needs the resolved checkpoint path, so it runs after
        # setup rather than before; it is still written before any generation so
        # the provenance of a crashed run is on disk.
        index = _physical_gpu_index(self.args.device)
        self.gpu_wait = wait_for_free_memory(
            index,
            minimum_mib=int(self.args.min_free_mib),
            timeout_seconds=float(self.args.wait_timeout_seconds),
            log=lambda message: print(message, flush=True),
        )
        self.setup()
        self.write_config()
        self.recall()
        self.run_teacher()
        self.write_policy()
        self.official_score()
        self.top2 = None
        if bool(getattr(self.args, "top2_inference_eval", False)):
            self.top2_inference()
        self.analyse()
        verify = self.nll_scorer.verify_seed(trials=int(self.args.verify_nll_trials))
        _write_json(self.out / "seed_verification.json", verify)
        for trial in verify["trials"]:
            self.log("seed check {} {}: seeded={:.4f} live={:.4f} diff={:.4f}".format(
                trial["sample_id"], trial["route"], trial["seeded_nll"],
                trial["live_nll"], trial["abs_diff"]))
        self.log("seed verification: {}".format(verify["status"]))
        if verify["status"] == "MISMATCH":
            raise RuntimeError(
                "seeded V7 pair NLL does not reproduce live (max diff {})".format(
                    verify["max_abs_diff"])
            )
        peak = (
            torch.cuda.max_memory_allocated(torch.device(self.args.device))
            if torch.cuda.is_available() else 0
        )
        complete = {
            "status": "COMPLETE",
            "task_id": self.task,
            "v8_policy_metric": self.analysis["v8_policy_metric"],
            "teacher_oracle_metric": self.analysis["teacher_oracle_metric"],
            "actual_top2_inference_metric": self.analysis[
                "actual_top2_inference_metric"
            ],
            "v7_actual_route_metric": self.analysis["v7_actual_route_metric"],
            "states": self.analysis["states"],
            "seed_verification": verify,
            "git_commit": _git_commit(),
            "duration_seconds": time.time() - self.started,
            "peak_gpu_memory_bytes": int(peak),
        }
        _write_json(self.out / "COMPLETE.json", complete)
        self.log("COMPLETE: {}".format(json.dumps(complete, sort_keys=True)))
        return complete


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--formal-root", default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--diagnostic-root", default=DEFAULT_DIAGNOSTIC_ROOT)
    parser.add_argument("--query-cache", default=DEFAULT_QUERY_CACHE)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--recall-top-m", type=int, default=8)
    parser.add_argument("--history-only", action="store_true",
                        help="route only over experts of earlier tasks, as the "
                             "pool stands when the task is being learned")
    parser.add_argument("--shortlist-ks", type=int, default=4)
    parser.add_argument("--pair-search-mode", default=PAIR_SEARCH_BOUNDED,
                        choices=(PAIR_SEARCH_BOUNDED, PAIR_SEARCH_EXHAUSTIVE),
                        help="bounded sweeps the K_s best singles; exhaustive is "
                             "C(H,2) generations per pending sample")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--pair-min-metric-gain", type=float, default=None)
    parser.add_argument("--top2-inference-eval", action="store_true",
                        help="additionally generate the *deployable* fixed Top-2 "
                             "route (no answer, no teacher, no NLL) so the run "
                             "reports actual_top2_inference_metric next to the "
                             "teacher's variable-cardinality oracle metric")
    parser.add_argument("--verify-nll-trials", type=int, default=8)
    parser.add_argument("--min-free-mib", type=int, default=18000,
                        help="wait for this much free device memory before loading")
    parser.add_argument("--wait-timeout-seconds", type=float, default=7200.0,
                        help="give up waiting for the GPU after this long; 0 disables")
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke only: use the first N validation samples")
    return parser


#: The note attached to a teacher-oracle number.  It exists so that the oracle
#: metric can never be quoted as a deployable result by accident.
TEACHER_ORACLE_NOTE = (
    "teacher-selected route replayed: cardinality 0/1/2 is chosen per sample with "
    "the validation answer in hand, so this is an ORACLE number and not deployable"
)
TOP2_NOT_MEASURED_NOTE = "not measured in this run: pass --top2-inference-eval"
TOP2_MEASURED_NOTE = (
    "held-out input -> frozen query -> committed keys -> max-per-expert -> fixed "
    "Top-2 distinct experts -> generation; no ground truth, no teacher, no NLL"
)


def metric_report(
    teacher_oracle_metric: float,
    top2_payload: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """The two metric slots, formally separated (PART 8).

    ``teacher_oracle_metric`` describes a route the teacher chose with the
    answer in hand.  ``actual_top2_inference_metric`` describes the route the
    committed pool produces with no supervision at all.  They answer different
    questions, so they get different keys and each carries a note saying which
    is which -- the failure mode this prevents is a single number being read as
    "V8's accuracy".
    """
    return {
        "teacher_oracle_metric": float(teacher_oracle_metric),
        "teacher_oracle_metric_note": TEACHER_ORACLE_NOTE,
        "actual_top2_inference_metric": (
            None if top2_payload is None else float(top2_payload["value"])
        ),
        "actual_top2_inference_metric_note": (
            TOP2_NOT_MEASURED_NOTE if top2_payload is None else TOP2_MEASURED_NOTE
        ),
        "metrics_are_separate": True,
    }


def _prediction_ids(path: str | Path) -> Optional[set]:
    """Sample ids in a V7 prediction file, or ``None`` if it cannot be read.

    The V7 generation files key samples as ``question_id``; the teacher and the
    cache use ``sample_id``.  Both are accepted, and a file that has neither is
    reported as unreadable rather than as a set of ``None`` -- an id-less file
    must not be able to "match" anything.
    """
    path = Path(path)
    if not path.is_file():
        return None
    ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                return None
            for key in ("sample_id", "question_id"):
                if row.get(key) is not None:
                    ids.add(str(row[key]))
                    break
            else:
                return None
    return ids or None


def teacher_result_payload(
    result: TeacherResult,
    teacher_config: Any,
) -> Dict[str, Any]:
    """The ``teacher_result.json`` body, built as a pure function.

    Pure and module-level **on purpose**: the first version of this payload was
    assembled inline in ``run_teacher`` and called a coverage helper that did not
    exist, which is a ``NameError`` only a real run could reach.  Every unit test
    passed, because no unit test builds this dict.  Extracting it lets a test
    assert the artefact's shape without a model, a GPU or a teacher run.
    """
    return {
        "task_id": int(result.task_id),
        "schema_version": TEACHER_RESULT_SCHEMA_VERSION,
        "teacher_search_mode": result.search_mode,
        "historical_experts_visible": list(result.visible_experts),
        "config": dataclasses.asdict(teacher_config),
        "states": result.state_counts(),
        "state_rates": result.state_rate(),
        "coverage": result.coverage_report(),
        "records": [record.to_dict() for record in result.records],
    }


def _cardinality_state(experts: Sequence[int]) -> str:
    """The state *name* for a fixed route, used only as a label in the answers.

    At inference the state is not a decision -- the router always returns the
    fixed Top-2 -- so this is a description of the cardinality that was used,
    not a teacher verdict.  It is deliberately derived from the route alone.
    """
    count = len(list(experts))
    if count == 0:
        return STATE_BASE_ONLY
    if count == 1:
        return STATE_REUSE1
    return STATE_REUSE2


def main() -> None:
    V8TaskRun(_parser().parse_args()).run()


if __name__ == "__main__":
    main()
