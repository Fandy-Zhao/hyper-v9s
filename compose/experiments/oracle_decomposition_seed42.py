"""Deterministic 512-sample capability/composition/routing decomposition."""

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from compose.data.records import question_text
from compose.eval.eval_task import _prompt
from compose.eval.formal_ucit_eval import HYPER_TASKS, TEST_FILES, VAL_COCO_FILES
from compose.eval.load_compose import load_compose_model
from compose.lora.rms import apply_kappa_calibration
from compose.oracle.candidate_sets import CandidateSet
from compose.oracle.evaluator import _candidate_nll, _collate, _prepare_multimodal_batch
from compose.router.router import ComposeRouter, load_compose_router_checkpoint
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images, tokenizer_image_token


PYTHON = "/home/zhaozhuofan/miniconda3/envs/hyper/bin/python"
BASE_MODEL = "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b"
VISION_TOWER = "/data/ckpt/zhaozhuofan/models/clip-vit-large-patch14-336"
PROJECTOR = BASE_MODEL + "/mm_projector.bin"
IMAGES = "/data/dataset/zhaozhuofan/UCIT/datasets"
SCORE_RE = re.compile(r"^\s*(?:Accuracy|Average)\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*%?\s*$", re.I)


def sid(record):
    return str(record.get("question_id", record.get("id")))


def prepare_subsets(output: Path):
    subsets = output / "subsets"
    subsets.mkdir(parents=True, exist_ok=True)
    manifest = {"policy": "first_512_in_official_test_order", "tasks": {}}
    for task_id, source in enumerate(TEST_FILES):
        records = json.loads(Path(source).read_text())[:512]
        task_dir = subsets / f"task{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        question_path = task_dir / "questions_512.json"
        question_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
        ids = [sid(record) for record in records]
        (task_dir / "sample_ids.json").write_text(json.dumps(ids, indent=2) + "\n")
        annotation = question_path
        if VAL_COCO_FILES[task_id]:
            coco = json.loads(Path(VAL_COCO_FILES[task_id]).read_text())
            images = coco["images"][:512]
            image_ids = {item["id"] for item in images}
            subset_coco = dict(coco)
            subset_coco["images"] = images
            subset_coco["annotations"] = [item for item in coco["annotations"] if item["image_id"] in image_ids]
            annotation = task_dir / "annotations_512.json"
            annotation.write_text(json.dumps(subset_coco, indent=2) + "\n")
        manifest["tasks"][str(task_id)] = {"source": source, "samples": len(records), "sample_ids": ids, "questions": str(question_path), "annotation": str(annotation)}
    (subsets / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PREPARED", "tasks": 6, "samples_per_task": 512}))


def compute_routes(records, router, device):
    clip = CLIPModel.from_pretrained(VISION_TOWER, torch_dtype=torch.float16).to(device).eval()
    processor = CLIPProcessor.from_pretrained(VISION_TOWER)
    routes, top_m = {}, {}
    for offset in range(0, len(records), 32):
        batch = records[offset:offset + 32]
        images = [Image.open(os.path.join(IMAGES, str(record["image"]))).convert("RGB") for record in batch]
        inputs = processor(text=[question_text(record) for record in batch], images=images, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.inference_mode():
            outputs = clip(**inputs)
            zv = torch.nn.functional.normalize(outputs.image_embeds.float(), dim=-1)
            zs = torch.nn.functional.normalize(outputs.text_embeds.float(), dim=-1)
            queries = router.query_encoder(zv, zs)
            selections = router.select(queries, router.expert_ids)
            retrieved = router.retrieve(queries, router.expert_ids).expert_ids
        for index, record in enumerate(batch):
            routes[sid(record)] = tuple(selections.sets[index])
            top_m[sid(record)] = tuple(int(value) for value in retrieved[index].tolist() if int(value) >= 0)
    del clip
    torch.cuda.empty_cache()
    return routes, top_m


def evaluate_nll(bundle, records, candidates_by_sample, batch_size, device):
    by_candidate = defaultdict(list)
    for index, record in enumerate(records):
        for ids in candidates_by_sample[sid(record)]:
            by_candidate[tuple(ids)].append(index)
    output = {sid(record): {} for record in records}
    for ids, indices in sorted(by_candidate.items()):
        candidate = CandidateSet(0, ids, tuple(1.0 for _ in ids), "none")
        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset:offset + batch_size]
            batch = [records[index] for index in batch_indices]
            raw = _collate(batch, bundle, IMAGES, device)
            prepared = _prepare_multimodal_batch(bundle, raw)
            losses, _ = _candidate_nll(bundle, candidate, prepared)
            key = "|".join(map(str, ids))
            for local, record in enumerate(batch):
                output[sid(record)][key] = float(losses[local])
    return output


def generate(bundle, records, mode_selections, device):
    outputs = {mode: {} for mode in mode_selections}
    generated = 0
    started = time.perf_counter()
    for record in records:
        sample = sid(record)
        selections = {mode: tuple(values[sample]) for mode, values in mode_selections.items()}
        unique = {}
        for mode, ids in selections.items():
            unique.setdefault(ids, []).append(mode)
        prompt = _prompt(record, bundle.model.config, "vicuna_v1")
        input_ids = tokenizer_image_token(prompt, bundle.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image = Image.open(os.path.join(IMAGES, str(record["image"]))).convert("RGB")
        image_tensor = process_images([image], bundle.image_processor, bundle.model.config)[0].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        for ids, modes in unique.items():
            if ids:
                bundle.expert_pool.manager.set_default_selection(ids, [1.0] * len(ids))
            else:
                bundle.expert_pool.manager.clear_default_selection()
            with torch.inference_mode():
                output_ids = bundle.model.generate(input_ids=input_ids, images=image_tensor, do_sample=False, num_beams=1, max_new_tokens=128, use_cache=True)
            text = bundle.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
            for mode in modes:
                outputs[mode][sample] = text
            generated += 1
    elapsed = time.perf_counter() - started
    return outputs, {"unique_generations": generated, "generation_seconds": elapsed, "unique_generations_per_second": generated / elapsed}


def score(task_id, questions, annotation, answers, score_dir):
    score_dir.mkdir(parents=True, exist_ok=True)
    module = "llava.eval.eval_caption" if VAL_COCO_FILES[task_id] else "llava.eval.eval_deepseek_r1"
    result = subprocess.run([PYTHON, "-m", module, "--annotation-file", str(annotation), "--result-file", str(answers), "--output-dir", str(score_dir)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    value = None
    for line in (score_dir / "Result.text").read_text().splitlines():
        match = SCORE_RE.match(line)
        if match:
            value = float(match.group(1)); break
    if value is None:
        raise RuntimeError("scorer output has no metric")
    return value, module


def run_stage(formal: Path, output: Path, stage: int, device: str, batch_size: int):
    task_dir = output / "subsets" / f"task{stage}"
    questions = task_dir / "questions_512.json"
    annotation = task_dir / ("annotations_512.json" if VAL_COCO_FILES[stage] else "questions_512.json")
    records = json.loads(questions.read_text())
    snapshot = formal / f"task{stage}" / "snapshots" / f"task{stage}"
    snapshot_manifest = json.loads((snapshot / "manifest.json").read_text())
    pool = snapshot_manifest["pool_checkpoint_dir"]
    router = ComposeRouter(); load_compose_router_checkpoint(str(snapshot / "router_checkpoint.pt"), router)
    router.to(device).eval()
    routes, top_m = compute_routes(records, router, device)
    candidates = {}
    for record in records:
        sample = sid(record)
        values = [tuple()] + [(expert,) for expert in router.expert_ids]
        values += [tuple(pair) for pair in combinations(top_m[sample], 2)]
        candidates[sample] = sorted(set(values), key=lambda value: (len(value), value))
    bundle = load_compose_model(model_path=BASE_MODEL, checkpoint_dir=pool, vision_tower=VISION_TOWER, projector_path=PROJECTOR, expert_id=None, device=device, dtype=torch.bfloat16, model_max_length=2048)
    calibration = bundle.load_summary.get("rms_calibration")
    if calibration:
        apply_kappa_calibration(bundle.model, calibration)
    nll = evaluate_nll(bundle, records, candidates, batch_size, device)
    selections = {"Base": {}, "BestSingle": {}, "BestPair": {}, "ActualRoute": routes}
    per_sample = []
    for record in records:
        sample = sid(record); values = nll[sample]
        singles = [(key, value) for key, value in values.items() if key and "|" not in key]
        pairs = [(key, value) for key, value in values.items() if "|" in key]
        best_single = min(singles, key=lambda item: item[1])[0]
        best_pair = min(pairs, key=lambda item: item[1])[0]
        selections["Base"][sample] = tuple()
        selections["BestSingle"][sample] = tuple(map(int, best_single.split("|")))
        selections["BestPair"][sample] = tuple(map(int, best_pair.split("|")))
        actual_key = "|".join(map(str, routes[sample]))
        per_sample.append({"sample_id": sample, "base_nll": values[""], "best_single_nll": values[best_single], "best_pair_nll": values[best_pair], "actual_route_nll": values[actual_key], "best_single": best_single, "best_pair": best_pair, "actual_route": actual_key, "top_m": list(top_m[sample])})
    generated, generation_stats = generate(bundle, records, selections, device)
    stage_dir = output / "stages" / f"task{stage}"; stage_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for mode in ("Base", "BestSingle", "BestPair", "ActualRoute"):
        answer_path = stage_dir / f"{mode}.jsonl"
        with answer_path.open("w", encoding="utf-8") as handle:
            for record in records:
                sample = sid(record)
                handle.write(json.dumps({"question_id": sample, "prompt": question_text(record), "text": generated[mode][sample], "model_id": "compose", "metadata": {"expert_ids": list(selections[mode][sample]), "oracle_diagnostic": mode in ("BestSingle", "BestPair")}}, ensure_ascii=False) + "\n")
        value, scorer = score(stage, questions, annotation, answer_path, stage_dir / "scores" / mode)
        metrics[mode] = {"metric": value, "scorer": scorer, "answers": str(answer_path)}
    nll_means = {"Base": statistics.fmean(row["base_nll"] for row in per_sample), "BestSingle": statistics.fmean(row["best_single_nll"] for row in per_sample), "BestPair": statistics.fmean(row["best_pair_nll"] for row in per_sample), "ActualRoute": statistics.fmean(row["actual_route_nll"] for row in per_sample)}
    summary = {"task_id": stage, "task": HYPER_TASKS[stage]["dataset"], "samples": len(records), "visible_experts": list(router.expert_ids), "top_m_limit": router.top_m, "nll": nll_means, "generation": metrics, "gaps": {"SingleGain": metrics["BestSingle"]["metric"] - metrics["Base"]["metric"], "PairGain": metrics["BestPair"]["metric"] - metrics["BestSingle"]["metric"], "RoutingGap": metrics["BestPair"]["metric"] - metrics["ActualRoute"]["metric"]}, "generation_stats": generation_stats, "peak_memory_bytes": int(torch.cuda.max_memory_allocated(torch.device(device)))}
    (stage_dir / "per_sample.json").write_text(json.dumps(per_sample, indent=2) + "\n")
    (stage_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


def summarize(output: Path):
    rows = []
    for stage in range(6):
        item = json.loads((output / "stages" / f"task{stage}" / "summary.json").read_text())
        rows.append({"stage": stage, "task": item["task"], "samples": item["samples"], "visible_experts": len(item["visible_experts"]), "top_m_limit": item["top_m_limit"], "base_nll": item["nll"]["Base"], "best_single_nll": item["nll"]["BestSingle"], "best_pair_nll": item["nll"]["BestPair"], "actual_route_nll": item["nll"]["ActualRoute"], "base_metric": item["generation"]["Base"]["metric"], "best_single_metric": item["generation"]["BestSingle"]["metric"], "best_pair_metric": item["generation"]["BestPair"]["metric"], "routed_metric": item["generation"]["ActualRoute"]["metric"], **item["gaps"]})
    with (output / "C_oracle_decomposition.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = ["# Experiment C — Oracle Capability Decomposition", "", "Deterministic first-512 subset per official test_3000. BestPair is restricted to per-sample formal Key Top-M=8 candidates.", "", "| task | Base | BestSingle | BestPair | Routed | SingleGain | PairGain | RoutingGap |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        report.append("| {task} | {base_metric:.2f} | {best_single_metric:.2f} | {best_pair_metric:.2f} | {routed_metric:.2f} | {SingleGain:+.2f} | {PairGain:+.2f} | {RoutingGap:+.2f} |".format(**row))
    (output / "C_oracle_decomposition_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"status": "COMPLETE", "rows": len(rows)}))


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare"); prepare.add_argument("--output-root", required=True)
    run = sub.add_parser("run-stage"); run.add_argument("--formal-root", required=True); run.add_argument("--output-root", required=True); run.add_argument("--stage", type=int, required=True); run.add_argument("--device", default="cuda:0"); run.add_argument("--batch-size", type=int, default=8)
    summary = sub.add_parser("summarize"); summary.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if args.command == "prepare": prepare_subsets(Path(args.output_root))
    elif args.command == "run-stage": run_stage(Path(args.formal_root), Path(args.output_root), args.stage, args.device, args.batch_size)
    else: summarize(Path(args.output_root))


if __name__ == "__main__":
    main()
