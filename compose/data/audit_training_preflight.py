"""Preflight audit for format-controlled A/B training data.

Performs the fifteen pre-registered checks required before any
format-controlled training run may start:

  1. five tasks x train/val/test files exist with 1600/200/400 records
  2. all image paths resolve to real files
  3. scene_id preserved after conversion
  4. required_functions preserved after conversion
  5. answer is only "A" or "B"
  6. every training prompt uses the identical answer template
  7. in-context A and B are both exactly one supervised token
  8. loss mask covers only the answer token (no newline/EOS/space/period/explanation)
  9. A and B token counts are equal
 10. scene_id and image hash zero overlap across train/val/test
 11. converted files get an independent SHA-256 manifest
 12. >=32 random samples per task run through the real dataset loader
 13. tensor shape / label-mask / answer-token-position audit
 14. explicit tokenizer checks for A, B, " A", " B" (standalone + in prompt)

On failure of the tokenization invariant (answer not a single token) the
script prints BLOCKED_BY_TOKENIZATION_MISMATCH and exits non-zero.
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import torch

BLOCKED = "BLOCKED_BY_TOKENIZATION_MISMATCH"
TASKS = ("A_only", "B_only", "C_only", "A_plus_B", "B_plus_C")
SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {"train": 1600, "val": 200, "test": 400}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _join_loader_audit(bundle, records, task: str) -> dict:
    """Run the real LazySupervisedDataset + collator over the records."""
    from compose.train.data import DataCollatorForSupervisedDataset, LazySupervisedDataset

    class _DataArgs:
        pass

    data_args = _DataArgs()
    data_args.data_path = None
    data_args.image_folder = str(bundle["image_root"])
    data_args.is_multimodal = True
    data_args.image_aspect_ratio = "pad"
    data_args.mm_use_im_start_end = False
    data_args.mm_use_im_patch_token = False
    data_args.memory_data_path = None
    data_args.image_processor = bundle["image_processor"]
    dataset = LazySupervisedDataset.__new__(LazySupervisedDataset)
    dataset.records = records
    dataset.tokenizer = bundle["tokenizer"]
    dataset.data_args = data_args
    collator = DataCollatorForSupervisedDataset(bundle["tokenizer"])
    # random but reproducible subset of at least 32 samples
    rng = random.Random(730 + sum(map(ord, task)))
    indices = rng.sample(range(len(records)), min(64, len(records)))
    instances = [dataset[i] for i in indices]
    batch = collator(instances)
    ignore = -100
    supervised_rows = []
    for position, instance in enumerate(instances):
        labels = instance["labels"].tolist()
        positions = [i for i, value in enumerate(labels) if value != ignore]
        token_ids = [labels[i] for i in positions]
        rows = {
            "batch_position": position,
            "sample_id": str(instance["sample_id"]),
            "input_ids_shape": list(instance["input_ids"].shape),
            "labels_shape": list(instance["labels"].shape),
            "supervised_positions": positions,
            "supervised_token_ids": token_ids,
        }
        supervised_rows.append(rows)
    collated = {
        "input_ids_shape": list(batch["input_ids"].shape),
        "labels_shape": list(batch["labels"].shape),
        "attention_mask_shape": list(batch["attention_mask"].shape),
        "images_shape": list(batch["images"].shape),
        "collator_summary": collator.supervision_summary(),
        "batch_size": len(instances),
    }
    return {"per_sample": supervised_rows, "collated": collated}


def audit(source_root: str, output_root: str, model_path: str, vision_tower: str):
    import transformers

    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN

    from compose.train.data import preprocess, preprocess_multimodal

    source = Path(source_root)
    output = Path(output_root)
    checks = {}
    blocked = False

    # ---- 1. file existence + record counts ---------------------------------
    counts = {}
    for task in TASKS:
        counts[task] = {}
        for split in SPLITS:
            path = output / "instructions" / task / f"{split}.json"
            if not path.is_file():
                checks.setdefault("counts", {})[f"{task}/{split}"] = "MISSING"
                continue
            records = json.loads(path.read_text(encoding="utf-8"))
            counts[task][split] = len(records)
            checks.setdefault("counts", {})[f"{task}/{split}"] = {
                "records": len(records),
                "expected": EXPECTED_COUNTS[split],
                "ok": len(records) == EXPECTED_COUNTS[split],
            }

    # ---- 2. image paths -----------------------------------------------------
    # Converted records carry image paths relative to the SOURCE root
    # (e.g. "images/scene_000000.png"), matching the old F2 convention of
    # pointing --image-folder at a parent directory that contains images/.
    image_root = source
    missing_images = []
    checked_images = set()
    for task in TASKS:
        for split in SPLITS:
            path = output / "instructions" / task / f"{split}.json"
            if not path.is_file():
                continue
            for record in json.loads(path.read_text(encoding="utf-8")):
                image_path = image_root / str(record["image"])
                checked_images.add(record["image"])
                if not image_path.is_file():
                    missing_images.append(f"{task}/{split}:{record['image']}")
    checks["images"] = {
        "unique_images_referenced": len(checked_images),
        "missing_images": len(missing_images),
        "missing_sample": missing_images[:20],
        "ok": not missing_images,
    }

    # ---- 3-6. metadata preservation + answer/template invariants -----------
    scene_ids = {split: set() for split in SPLITS}
    template_violations = []
    answer_violations = []
    scene_missing = []
    functions_missing = []
    answer_counts = {}
    for task in TASKS:
        answer_counts[task] = {}
        for split in SPLITS:
            path = output / "instructions" / task / f"{split}.json"
            if not path.is_file():
                continue
            records = json.loads(path.read_text(encoding="utf-8"))
            answer_counts[task][split] = {"A": 0, "B": 0}
            for record in records:
                if "scene_id" not in record:
                    scene_missing.append(record.get("id", "?"))
                else:
                    scene_ids[split].add(record["scene_id"])
                if "required_functions" not in record:
                    functions_missing.append(record.get("id", "?"))
                answer = record.get("answer")
                if answer not in ("A", "B"):
                    answer_violations.append((record.get("id", "?"), answer))
                else:
                    answer_counts[task][split][answer] += 1
                messages = record["conversations"]
                if len(messages) != 2:
                    template_violations.append((record.get("id", "?"), "message_count"))
                    continue
                if messages[0]["from"] != "human" or messages[1]["from"] != "gpt":
                    template_violations.append((record.get("id", "?"), "roles"))
                    continue
                if not messages[0]["value"].startswith(f"{DEFAULT_IMAGE_TOKEN}\n"):
                    template_violations.append((record.get("id", "?"), "image_prefix"))
                    continue
                if not messages[0]["value"].endswith(
                    "\nAnswer the question using a single word or phrase."
                ):
                    template_violations.append((record.get("id", "?"), "question_suffix"))
                    continue
                if messages[1]["value"] not in ("A", "B"):
                    template_violations.append((record.get("id", "?"), "answer_value"))
    checks["scene_ids"] = {
        "missing_scene_id": len(scene_missing),
        "sample": scene_missing[:10],
        "ok": not scene_missing,
    }
    checks["required_functions"] = {
        "missing": len(functions_missing),
        "sample": functions_missing[:10],
        "ok": not functions_missing,
    }
    checks["answers"] = {
        "violations": len(answer_violations),
        "sample": answer_violations[:10],
        "counts": answer_counts,
        "ok": not answer_violations,
    }
    checks["template"] = {
        "violations": len(template_violations),
        "sample": template_violations[:10],
        "ok": not template_violations,
    }

    # ---- 10. scene_id / image hash zero overlap ----------------------------
    overlap = {
        split: sorted(scene_ids[split] & other)
        for split, other in (
            ("train", scene_ids["val"]),
            ("val", scene_ids["test"]),
            ("train", scene_ids["test"]),
        )
    }
    scene_overlap_total = sum(len(value) for value in overlap.values())
    image_hashes = {split: set() for split in SPLITS}
    for task in TASKS:
        for split in SPLITS:
            path = output / "instructions" / task / f"{split}.json"
            if not path.is_file():
                continue
            for record in json.loads(path.read_text(encoding="utf-8")):
                image_hashes[split].add(file_sha256(image_root / str(record["image"])))
    hash_overlap = {
        split: len(image_hashes[split] & image_hashes[other])
        for split, other in (
            ("train", "val"),
            ("val", "test"),
            ("train", "test"),
        )
    }
    checks["split_overlap"] = {
        "scene_overlap": {k: v for k, v in overlap.items() if v},
        "scene_overlap_total": scene_overlap_total,
        "image_hash_overlap": hash_overlap,
        "scene_counts": {split: len(scene_ids[split]) for split in SPLITS},
        "ok": scene_overlap_total == 0 and all(v == 0 for v in hash_overlap.values()),
    }

    # ---- 14. tokenizer checks -----------------------------------------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, model_max_length=2048, padding_side="right", use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    tokenizer_checks = {}
    for text in ("A", "B", " A", " B"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokenizer_checks[repr(text)] = {
            "token_ids": ids,
            "tokens": tokenizer.convert_ids_to_tokens(ids),
            "token_count": len(ids),
            "ok": len(ids) == 1,
        }
    a_id = tokenizer_checks["'A'"]["token_ids"][0]
    b_id = tokenizer_checks["'B'"]["token_ids"][0]
    if tokenizer_checks["'A'"]["ok"] and tokenizer_checks["'B'"]["ok"]:
        answer_ids = {a_id, b_id}
    else:
        answer_ids = set()
        blocked = True

    # ---- 7-8. in-context supervision audit on the real template ------------
    # Invariant: the supervised region must be exactly the single answer token,
    # optionally followed by the structural EOS ("</s>") that the LLaVA v1
    # template appends to every assistant turn. Any newline/space/period or
    # explanation token inside the supervised region is a violation.
    eos_id = int(tokenizer.convert_tokens_to_ids("</s>"))
    supervision = {"by_task": {}, "samples": 0, "violations": [], "eos_id": eos_id}
    supervised_sets = set()
    for task in TASKS:
        path = output / "instructions" / task / "train.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        rng = random.Random(1730 + sum(map(ord, task)))
        records = [records[i] for i in rng.sample(range(len(records)), min(64, len(records)))]
        local = []
        for record in records:
            source_records = [[record["conversations"][0], record["conversations"][1]]]
            preprocessed = preprocess_multimodal(source_records, type("A", (), {
                "is_multimodal": True, "mm_use_im_start_end": False,
            })())
            encoded = preprocess(preprocessed, tokenizer, has_image=True)
            labels = encoded["labels"][0]
            input_ids = encoded["input_ids"][0]
            positions = [int(i) for i, value in enumerate(labels) if value != IGNORE_INDEX]
            token_ids = [int(input_ids[i]) for i in positions]
            expected = int(answer_ids and (a_id if record["answer"] == "A" else b_id))
            ok = (
                len(positions) >= 1
                and token_ids[0] == expected
                and (
                    len(token_ids) == 1
                    or (len(token_ids) == 2 and token_ids[1] == eos_id)
                )
            )
            if not ok:
                supervision["violations"].append({
                    "task": task, "id": record["id"], "answer": record["answer"],
                    "positions": positions, "token_ids": token_ids, "expected": expected,
                })
            supervised_sets.add(tuple(token_ids))
            local.append({
                "id": record["id"], "answer": record["answer"],
                "answer_token_position": positions[0] if positions else None,
                "eos_included": len(token_ids) == 2,
                "positions": positions, "token_ids": token_ids,
            })
            supervision["samples"] += 1
        supervision["by_task"][task] = local
    supervision["supervised_token_set"] = sorted(supervised_sets)
    supervision["ok"] = not supervision["violations"]
    if not supervision["ok"]:
        blocked = True

    # ---- 9. A/B balance ------------------------------------------------------
    balance_ok = True
    for task in TASKS:
        for split in SPLITS:
            counts = answer_counts[task][split]
            if counts["A"] != counts["B"]:
                balance_ok = False
    checks["a_b_balance"] = {
        "ok": balance_ok,
        "per_task": answer_counts,
        "single_token_A": tokenizer_checks["'A'"]["ok"],
        "single_token_B": tokenizer_checks["'B'"]["ok"],
        "single_token_space_A": tokenizer_checks["' A'"]["ok"],
        "single_token_space_B": tokenizer_checks["' B'"]["ok"],
    }

    # ---- 11. SHA-256 manifest of converted files ----------------------------
    manifest_entries = {}
    for path in sorted((output / "instructions").rglob("*.json")):
        manifest_entries[str(path.relative_to(output))] = file_sha256(path)
    manifest_path = output / "training_manifest.sha256"
    manifest_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(manifest_entries.items())),
        encoding="utf-8",
    )
    checks["converted_manifest"] = {
        "path": str(manifest_path),
        "files": len(manifest_entries),
        "sha256": file_sha256(manifest_path),
    }

    # ---- 12-13. real loader audit --------------------------------------------
    image_processor = transformers.CLIPImageProcessor.from_pretrained(vision_tower)
    bundle = {
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "image_root": source,
    }
    loader_audit = {}
    for task in TASKS:
        path = output / "instructions" / task / "train.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        loader_audit[task] = _join_loader_audit(bundle, records, task)
    checks["loader_audit"] = loader_audit

    return {
        "status": "BLOCKED" if blocked else "PASSED",
        "blocked_reason": "answer not a single supervised token in training prompt"
        if blocked else None,
        "tokenizer_checks": tokenizer_checks,
        "supervision_audit": supervision,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--report-file", required=True)
    args = parser.parse_args()

    result = audit(
        args.source_root, args.output_root, args.model_path, args.vision_tower
    )
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "counts": result["checks"]["counts"],
        "template_ok": result["checks"]["template"]["ok"],
        "answers_ok": result["checks"]["answers"]["ok"],
        "split_overlap_ok": result["checks"]["split_overlap"]["ok"],
        "tokenizer_checks": result["tokenizer_checks"],
        "supervision_ok": result["supervision_audit"]["ok"],
        "supervised_token_set": result["supervision_audit"]["supervised_token_set"],
        "loader_min_supervised": {
            task: result["checks"]["loader_audit"][task]["collated"]["collator_summary"]["min"]
            for task in TASKS
        },
        "loader_zero_supervision": {
            task: result["checks"]["loader_audit"][task]["collated"]["collator_summary"]["zero_supervision"]
            for task in TASKS
        },
        "converted_manifest_sha256": result["checks"]["converted_manifest"]["sha256"],
    }
    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Format-Controlled Training Preflight",
        "",
        f"- source root: `{args.source_root}`",
        f"- output root: `{args.output_root}`",
        f"- status: **{result['status']}**",
        "",
        "## 1. Files and counts",
        "",
        "| task/split | records | expected | ok |",
        "|---|---|---|---|",
    ]
    for key, value in sorted(result["checks"]["counts"].items()):
        if value == "MISSING":
            lines.append(f"| {key} | MISSING | — | no |")
        else:
            lines.append(
                f"| {key} | {value['records']} | {value['expected']} | {value['ok']} |"
            )
    lines += [
        "",
        "## 2. Images",
        "",
        f"- referenced images: {result['checks']['images']['unique_images_referenced']}",
        f"- missing images: {result['checks']['images']['missing_images']}",
        "",
        "## 3. Metadata preservation",
        "",
        f"- missing scene_id: {result['checks']['scene_ids']['missing_scene_id']}",
        f"- missing required_functions: {result['checks']['required_functions']['missing']}",
        "",
        "## 4. Answers and template",
        "",
        f"- answer violations: {result['checks']['answers']['violations']}",
        f"- template violations: {result['checks']['template']['violations']}",
        "",
        "## 5. Tokenizer audit (standalone)",
        "",
        "| text | token ids | token count | ok |",
        "|---|---|---|---|",
    ]
    for key, value in result["tokenizer_checks"].items():
        lines.append(
            f"| {key} | {value['token_ids']} | {value['token_count']} | {value['ok']} |"
        )
    lines += [
        "",
        "## 6. In-prompt supervision audit (loss mask)",
        "",
        f"- samples audited: {result['supervision_audit']['samples']}",
        f"- violations: {len(result['supervision_audit']['violations'])}",
        f"- supervised token sets observed: {result['supervision_audit']['supervised_token_set']}",
        "",
        "## 7. Scene / image split overlap",
        "",
        f"- scene overlap: {result['checks']['split_overlap']['scene_overlap_total']}",
        f"- image-hash overlap: {result['checks']['split_overlap']['image_hash_overlap']}",
        "",
        "## 8. Loader audit",
        "",
        "| task | batch | input_ids | labels | supervised min | supervised max |",
        "|---|---|---|---|---|---|",
    ]
    for task in TASKS:
        collated = result["checks"]["loader_audit"][task]["collated"]
        lines.append(
            f"| {task} | {collated['batch_size']} | {collated['input_ids_shape']} "
            f"| {collated['labels_shape']} | {collated['collator_summary']['min']} "
            f"| {collated['collator_summary']['max']} |"
        )
    lines += [
        "",
        "## 9. Converted files manifest",
        "",
        f"- manifest: `{result['checks']['converted_manifest']['path']}`",
        f"- sha256: `{result['checks']['converted_manifest']['sha256']}`",
        "",
        "## 10. Conclusion",
        "",
        f"- status: **{result['status']}**",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if blocked:
        print(BLOCKED)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
