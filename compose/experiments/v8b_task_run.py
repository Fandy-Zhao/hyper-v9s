"""Real V8-B single-task pilot: teacher labels -> multi-key gated training.

This entry point intentionally starts from the previous task's committed pool.
It never reads a test split and never overwrites the input checkpoint.  The
answer-supervised teacher result is produced separately by ``v8_task_run`` on
the current task's *train* split, then consumed here as an immutable contract.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F

from compose.data.records import answer_text, question_text
from compose.eval.formal_ucit_eval import _score_answers
from compose.eval.load_compose import load_compose_model
from compose.experts.checkpoint import save_expert_checkpoint
from compose.train.arguments import DataArguments
from compose.train.data import DataCollatorForSupervisedDataset, LazySupervisedDataset
from compose.v7.pool import initialize_candidate_keys
from compose.v7.training import per_sample_teacher_forcing_token_nll
from compose.v8.commit import write_v8_state
from compose.v8.config import STATE_RESIDUAL, V8Config
from compose.v8.generate import GenerationEngine
from compose.v8.key_learning import create_alias_keys
from compose.v8.metric_adapter import TaskMetricAdapter
from compose.v8.pool import MultiKeyExpertPool
from compose.v8.routing import MultiKeyRouter
from compose.v8.teacher import TeacherResult, TeacherSampleRecord
from compose.v8.trainer import TrainBatch, V8TaskTrainer


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True,
                              ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _teacher_result(path: str | Path, expected_task: int) -> TeacherResult:
    payload = _json(path)
    if int(payload["task_id"]) != int(expected_task):
        raise ValueError("teacher result belongs to a different task")
    if payload.get("teacher_search_mode") != "full_history_single_oracle":
        raise ValueError("V8-B requires the full-history capability oracle")
    coverage = payload.get("coverage", {})
    if not coverage.get("full_coverage", False):
        raise ValueError("teacher result does not cover every visible historical expert")
    records = []
    fields = {field.name for field in dataclasses.fields(TeacherSampleRecord)}
    for raw in payload["records"]:
        values = {name: raw[name] for name in fields if name in raw}
        values["key_targets"] = {
            str(key): value for key, value in values.get("key_targets", {}).items()
        }
        records.append(TeacherSampleRecord(**values))
    return TeacherResult(
        task_id=int(payload["task_id"]), records=records,
        config=dict(payload.get("config", {})),
        scored_routes=list(payload.get("scored_routes", [])),
    )


def _load_queries(root: str | Path, task: int, split: str,
                  wanted: Sequence[str]) -> Dict[str, torch.Tensor]:
    path = Path(root) / "query_cache" / f"task{task}" / split / "queries.pt"
    payload = torch.load(path, map_location="cpu")
    ids = [str(value) for value in payload["sample_ids"]]
    index = {sample_id: row for row, sample_id in enumerate(ids)}
    missing = sorted(set(wanted) - set(index))
    if missing:
        raise ValueError(f"query cache is missing samples: {missing[:8]}")
    return {
        sample_id: payload["queries"][index[sample_id]].detach().float()
        for sample_id in wanted
    }


class SupervisedForward:
    def __init__(self, bundle, data_path: str, image_folder: str, device: str) -> None:
        from llava import conversation as conversation_lib

        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
        args = DataArguments(data_path=data_path, image_folder=image_folder,
                             image_aspect_ratio="pad")
        args.image_processor = bundle.image_processor
        args.is_multimodal = True
        args.mm_use_im_start_end = False
        self.dataset = LazySupervisedDataset(data_path, bundle.tokenizer, args)
        self.collator = DataCollatorForSupervisedDataset(bundle.tokenizer)
        self.index = {
            str(record.get("id", record.get("question_id", row))): row
            for row, record in enumerate(self.dataset.records)
        }
        self.bundle = bundle
        self.device = str(device)

    def __call__(self, sample_ids: Sequence[str]) -> torch.Tensor:
        items = [self.dataset[self.index[str(sample_id)]] for sample_id in sample_ids]
        batch = self.collator(items)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        images = batch["images"]
        if isinstance(images, list):
            images = [image.to(self.device, dtype=torch.bfloat16) for image in images]
        else:
            images = images.to(self.device, dtype=torch.bfloat16)
        expanded = self.bundle.model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids, position_ids=None, attention_mask=attention_mask,
            past_key_values=None, labels=labels, images=images,
        )
        outputs = self.bundle.model(
            input_ids=expanded[0], position_ids=expanded[1],
            attention_mask=expanded[2], past_key_values=expanded[3],
            inputs_embeds=expanded[4], labels=expanded[5], return_dict=True,
        )
        return per_sample_teacher_forcing_token_nll(outputs.logits, expanded[5])


def _candidate_assignment(pool: MultiKeyExpertPool, candidate_ids: Sequence[int],
                          queries: Mapping[str, torch.Tensor],
                          residual_ids: Sequence[str]) -> Dict[str, int]:
    keys = torch.stack([
        F.normalize(pool.keys[pool.origin_key_id(expert_id)].detach().float(), dim=-1)
        for expert_id in candidate_ids
    ])
    result = {}
    for sample_id in residual_ids:
        query = F.normalize(queries[sample_id].float(), dim=-1)
        result[sample_id] = int(candidate_ids[int(torch.argmax(keys @ query).item())])
    return result


def _batches(result: TeacherResult, assignments: Mapping[str, int],
             batch_size: int, seed: int) -> List[TrainBatch]:
    records = list(result.records)
    random.Random(int(seed)).shuffle(records)
    batches = []
    for offset in range(0, len(records), int(batch_size)):
        chunk = records[offset: offset + int(batch_size)]
        ids = [record.sample_id for record in chunk]
        states = {record.sample_id: record.state for record in chunk}
        experts = {
            record.sample_id: (
                list(record.residual_context)
                if record.state == STATE_RESIDUAL
                else list(record.selected_experts)
            )
            for record in chunk
        }
        candidates = {
            sample_id: assignments[sample_id]
            for sample_id in ids if sample_id in assignments
        }
        batches.append(TrainBatch(ids, states, experts, candidates))
    return batches


def _evaluate(args, bundle, pool: MultiKeyExpertPool, output: Path) -> Dict[str, Any]:
    # Training enables gradient checkpointing and disables the KV cache.  The
    # smoke metric must exercise the deployable inference path instead.
    bundle.model.eval()
    bundle.model.config.use_cache = True
    records = _json(args.val_json)
    records_by_id = {
        str(record.get("id", record.get("question_id"))): record for record in records
    }
    ids = sorted(records_by_id)[: int(args.val_limit)]
    records_by_id = {sample_id: records_by_id[sample_id] for sample_id in ids}
    queries = _load_queries(args.query_cache_root, args.task, "val", ids)
    device = torch.device(args.device)
    result = MultiKeyRouter()(torch.stack([queries[sid] for sid in ids]).to(device), pool)
    routes = {
        sample_id: [int(value) for value in result.expert_ids[row].tolist()]
        for row, sample_id in enumerate(ids)
    }
    directory = output / "validation"
    engine = GenerationEngine(bundle, image_folder=args.image_folder,
                              device=args.device, max_new_tokens=args.max_new_tokens,
                              cache_path=directory / "generation_cache.jsonl")
    answers = engine.generate_route(routes, records_by_id)
    directory.mkdir(parents=True, exist_ok=True)
    answers_path = directory / "answers.jsonl"
    with answers_path.open("w", encoding="utf-8") as handle:
        for sample_id in ids:
            handle.write(json.dumps({
                "question_id": sample_id,
                "prompt": question_text(records_by_id[sample_id]),
                "text": answers[sample_id], "model_id": "compose-v8b",
                "metadata": {"expert_ids": routes[sample_id]},
            }, ensure_ascii=False) + "\n")
    questions = directory / "questions.json"
    _write(questions, {"records": []})
    # The official scorer accepts the original UCIT annotation shape: a list.
    tmp = questions.with_name(questions.name + ".tmp")
    tmp.write_text(json.dumps([
        {"question_id": sid, "answer": answer_text(records_by_id[sid]),
         "image": records_by_id[sid].get("image")}
        for sid in ids
    ], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, questions)
    official = _score_answers(directory, args.task, args.task, answers_path,
                              annotation_file=str(questions))
    payload = {
        "samples": len(ids), "official": official,
        "routes": routes, "pool_audit": pool.audit(),
        "generated": engine.generated_count, "cache_hits": engine.cache_hits,
    }
    _write(directory / "metric.json", payload)
    return payload


def run(args) -> Dict[str, Any]:
    started = time.time()
    output = Path(args.output_dir)
    if (output / "COMPLETE.json").exists():
        raise FileExistsError(f"completed output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if "test" in Path(args.train_json).name.lower():
        raise ValueError("V8-B teacher/training may not use a test split")

    teacher = _teacher_result(args.teacher_result, args.task)
    sample_ids = [record.sample_id for record in teacher.records]
    queries_cpu = _load_queries(args.query_cache_root, args.task, "train", sample_ids)
    v7_state = torch.load(Path(args.previous_checkpoint) / "v7_keys.pt",
                          map_location="cpu")
    manifest = _json(Path(args.previous_checkpoint) / "compose_experts.json")
    pool = MultiKeyExpertPool.load_v7_pool(v7_state, manifest, frozen=True)
    historical_ids = pool.expert_ids()

    bundle = load_compose_model(
        model_path=args.model_path, checkpoint_dir=args.previous_checkpoint,
        vision_tower=args.vision_tower, projector_path=args.projector_path,
        expert_id=None, device=args.device, dtype=torch.bfloat16,
        model_max_length=args.model_max_length,
    )
    # A 7B backbone plus the historical pool fits for inference but a real
    # Residual backward pass needs activation recomputation on a 24 GiB card.
    # This is the same memory contract as the formal trainer: no KV cache while
    # training, checkpoint decoder blocks, and keep the frozen input embedding
    # output attached so gradients can reach the selected LoRA parameters.
    bundle.model.config.use_cache = False
    bundle.model.gradient_checkpointing_enable()
    if hasattr(bundle.model, "enable_input_require_grads"):
        bundle.model.enable_input_require_grads()
    bundle.model.train()
    candidate_ids = list(range(max(historical_ids) + 1,
                               max(historical_ids) + 1 + args.candidate_count))
    for expert_id in candidate_ids:
        bundle.expert_pool.register(expert_id, origin_task_id=str(args.task),
                                    tags=["v8b", "candidate"])

    all_train = torch.stack([queries_cpu[sample_id] for sample_id in sample_ids])
    candidate_keys, _, init_report = initialize_candidate_keys(
        all_train, num_train_samples=len(sample_ids), count=args.candidate_count,
        seed=args.seed,
    )
    for expert_id, key in zip(candidate_ids, candidate_keys):
        pool.add_expert(expert_id, origin_task=args.task, lifecycle="candidate")
        pool.add_key(expert_id, args.task, "origin", key,
                     lifecycle="candidate", trainable=True)
    alias_report = create_alias_keys(teacher, pool, args.task, queries_cpu,
                                     trainable=True)

    pool.to(torch.device(args.device))
    queries = {sample_id: value.to(args.device) for sample_id, value in queries_cpu.items()}
    residual_ids = [record.sample_id for record in teacher.records
                    if record.state == STATE_RESIDUAL]
    assignments = _candidate_assignment(pool, candidate_ids, queries, residual_ids)
    training = dataclasses.replace(
        V8Config().training, learning_rate=float(args.learning_rate),
        num_train_epochs=float(args.epochs),
        per_device_train_batch_size=int(args.batch_size),
        gradient_accumulation_steps=1, lambda_key=float(args.lambda_key),
        model_max_length=int(args.model_max_length), seed=int(args.seed),
    )
    config = dataclasses.replace(V8Config(seed=args.seed), training=training)
    forward = SupervisedForward(bundle, args.train_json, args.image_folder, args.device)
    trainer = V8TaskTrainer(
        model=bundle.model, manager=bundle.expert_pool.manager, pool=pool,
        config=config, current_task=args.task, candidate_expert_ids=candidate_ids,
        forward_fn=forward, queries_by_sample=queries,
    )
    reports = []
    for epoch in range(int(args.epochs)):
        report = trainer.train_epoch(
            _batches(teacher, assignments, args.batch_size, args.seed + epoch),
            teacher_result=teacher,
            progress=lambda message: print(f"[v8b epoch {epoch}] {message}", flush=True),
        )
        reports.append(report.to_dict())

    checkpoint = output / "v8_checkpoint.pt"
    final = trainer.finalize(checkpoint_path=checkpoint, commit=True)
    committed = output / "committed"
    save_expert_checkpoint(
        bundle.expert_pool, str(committed), expert_ids=bundle.expert_pool.expert_ids(),
        rms_calibration=bundle.load_summary.get("rms_calibration") or None,
    )
    write_v8_state(pool, committed / "v8_pool.pt", extra={
        "task": args.task, "teacher_result_sha256": _sha256(args.teacher_result),
    })
    evaluation = _evaluate(args, bundle, pool, output)
    result = {
        "status": "COMPLETE", "method": config.method, "task": args.task,
        "previous_checkpoint": str(Path(args.previous_checkpoint).resolve()),
        "teacher_result": str(Path(args.teacher_result).resolve()),
        "teacher_result_sha256": _sha256(args.teacher_result),
        "train_json_sha256": _sha256(args.train_json),
        "historical_expert_ids": historical_ids,
        "candidate_expert_ids": candidate_ids,
        "candidate_assignment_counts": {
            str(expert_id): sum(value == expert_id for value in assignments.values())
            for expert_id in candidate_ids
        },
        "candidate_key_initialization": init_report,
        "alias_creation": alias_report,
        "train_reports": reports, "finalize": final,
        "pool_audit": pool.audit(), "validation": evaluation,
        "duration_seconds": time.time() - started,
    }
    _write(output / "COMPLETE.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--previous-checkpoint", required=True)
    parser.add_argument("--teacher-result", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--query-cache-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--projector-path", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-count", type=int, default=4, choices=(4,))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lambda-key", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-max-length", type=int, default=2048)
    parser.add_argument("--val-limit", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser


def main() -> None:
    payload = run(_parser().parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
