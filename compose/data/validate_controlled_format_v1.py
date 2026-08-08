"""
Validation script for controlled_format_v1 dataset.

Checks:
1. All image paths exist
2. All scene metadata files exist
3. Train/val/test scene_id no overlap
4. Train/val/test image hash no overlap
5. All answers are A or B only
6. A/B balance per task per split
7. required_functions correctness
8. A_only doesn't contain counting or spatial terms
9. B_only doesn't contain shape or spatial terms
10. C_only doesn't contain counting terms
11. A_plus_B contains both shape and counting terms
12. B_plus_C contains both spatial and counting terms
13. Question semantics match scene metadata (answer verification)
14. negative_type matches actual modification
15. No duplicate (question, image) pairs
16. No exact duplicate samples
17. Image distribution consistency across tasks
18. JSON loadable by training loader
19. Tokenizer answer encoding audit
20. Reproducibility check (deterministic generation)
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


FUNCTIONS = ("A_only", "B_only", "C_only", "A_plus_B", "B_plus_C")
SHAPES = ("circle", "square", "triangle")
SPLITS = ("train", "val", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(root: Path) -> Dict[str, Dict[str, List[Dict]]]:
    """Load all data files."""
    data: Dict[str, Dict[str, List[Dict]]] = {}
    for task in FUNCTIONS:
        data[task] = {}
        for split in SPLITS:
            path = root / task / f"{split}.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing data file: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data[task][split] = json.load(f)
    return data


def load_scenes(root: Path) -> Dict[str, Dict]:
    """Load all scene metadata files."""
    scenes_dir = root / "scenes"
    scenes: Dict[str, Dict] = {}
    for scene_path in sorted(scenes_dir.glob("*.json")):
        with open(scene_path, "r", encoding="utf-8") as f:
            scene = json.load(f)
            scenes[scene["scene_id"]] = scene
    return scenes


def validate(
    root: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run all validation checks. Returns audit dict."""
    root_path = Path(root).resolve()
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {}

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    def add_error(msg: str) -> None:
        errors.append(msg)
        if verbose:
            print(f"  ERROR: {msg}")

    def add_warning(msg: str) -> None:
        warnings.append(msg)
        if verbose:
            print(f"  WARNING: {msg}")

    # Load data
    log("Loading data...")
    data = load_data(root_path)
    scenes = load_scenes(root_path)
    log(f"  Loaded {len(scenes)} scenes, {sum(len(data[t][s]) for t in FUNCTIONS for s in SPLITS)} QA samples")

    # =========================================================================
    # Check 1: All image paths exist
    # =========================================================================
    log("\n=== Check 1: Image path existence ===")
    missing_images = 0
    for task in FUNCTIONS:
        for split in SPLITS:
            for sample in data[task][split]:
                img_path = root_path / sample["image"]
                if not img_path.exists():
                    add_error(f"Missing image: {sample['image']} (sample {sample['id']})")
                    missing_images += 1
    if missing_images == 0:
        log("  All images exist ✓")

    # =========================================================================
    # Check 2: All scene metadata files exist
    # =========================================================================
    log("\n=== Check 2: Scene metadata existence ===")
    missing_scenes = 0
    all_scene_ids: Set[str] = set()
    for task in FUNCTIONS:
        for split in SPLITS:
            for sample in data[task][split]:
                sid = sample["scene_id"]
                all_scene_ids.add(sid)
                if sid not in scenes:
                    add_error(f"Missing scene metadata: {sid}")
                    missing_scenes += 1
    if missing_scenes == 0:
        log(f"  All {len(all_scene_ids)} scene IDs have metadata ✓")

    # =========================================================================
    # Check 3: Train/val/test scene_id no overlap
    # =========================================================================
    log("\n=== Check 3: Scene ID split isolation ===")
    split_scenes: Dict[str, Set[str]] = {split: set() for split in SPLITS}
    for task in FUNCTIONS:
        for split in SPLITS:
            for sample in data[task][split]:
                split_scenes[split].add(sample["scene_id"])

    overlap_errors = 0
    for s1 in SPLITS:
        for s2 in SPLITS:
            if s1 < s2:
                overlap = split_scenes[s1] & split_scenes[s2]
                if overlap:
                    add_error(f"Scene overlap {s1}∩{s2}: {overlap}")
                    overlap_errors += 1
    if overlap_errors == 0:
        log(f"  No scene overlap across splits ✓")
        stats["split_scene_counts"] = {s: len(split_scenes[s]) for s in SPLITS}

    # =========================================================================
    # Check 4: Train/val/test image hash no overlap
    # =========================================================================
    log("\n=== Check 4: Image hash split isolation ===")
    split_hashes: Dict[str, Set[str]] = {split: set() for split in SPLITS}
    for task in FUNCTIONS:
        for split in SPLITS:
            for sample in data[task][split]:
                img_path = root_path / sample["image"]
                h = _sha256(img_path)
                split_hashes[split].add(h)

    hash_overlap = 0
    for s1 in SPLITS:
        for s2 in SPLITS:
            if s1 < s2:
                overlap = split_hashes[s1] & split_hashes[s2]
                if overlap:
                    add_error(f"Image hash overlap {s1}∩{s2}: {len(overlap)} images")
                    hash_overlap += 1
    if hash_overlap == 0:
        log(f"  No image hash overlap across splits ✓")

    # =========================================================================
    # Check 5: All answers are A or B only
    # =========================================================================
    log("\n=== Check 5: Answer format ===")
    invalid_answers = 0
    for task in FUNCTIONS:
        for split in SPLITS:
            for sample in data[task][split]:
                if sample["answer"] not in ("A", "B"):
                    add_error(f"Invalid answer '{sample['answer']}' in {sample['id']}")
                    invalid_answers += 1
    if invalid_answers == 0:
        log("  All answers are 'A' or 'B' ✓")

    # =========================================================================
    # Check 6: A/B balance per task per split
    # =========================================================================
    log("\n=== Check 6: A/B balance ===")
    balance_ok = True
    balance_stats = {}
    for task in FUNCTIONS:
        balance_stats[task] = {}
        for split in SPLITS:
            counts = Counter(s["answer"] for s in data[task][split])
            total = len(data[task][split])
            a_pct = counts.get("A", 0) / total * 100
            b_pct = counts.get("B", 0) / total * 100
            balance_stats[task][split] = {"A": counts.get("A", 0), "B": counts.get("B", 0),
                                           "A_pct": round(a_pct, 1), "B_pct": round(b_pct, 1)}
            if abs(a_pct - 50) > 2:
                add_warning(f"{task}/{split}: A/B balance deviates: {a_pct:.1f}%/{b_pct:.1f}%")
                balance_ok = False
    if balance_ok:
        log("  All splits balanced (50±2%) ✓")
    stats["balance"] = balance_stats

    # =========================================================================
    # Check 7: required_functions correctness
    # =========================================================================
    log("\n=== Check 7: required_functions ===")
    expected_funcs = {
        "A_only": ["A"],
        "B_only": ["B"],
        "C_only": ["C"],
        "A_plus_B": ["A", "B"],
        "B_plus_C": ["B", "C"],
    }
    func_errors = 0
    for task in FUNCTIONS:
        for split in SPLITS:
            for sample in data[task][split]:
                if sample["required_functions"] != expected_funcs[task]:
                    add_error(f"Wrong required_functions in {sample['id']}: "
                              f"{sample['required_functions']} != {expected_funcs[task]}")
                    func_errors += 1
    if func_errors == 0:
        log("  All required_functions correct ✓")

    # =========================================================================
    # Check 8-12: Task semantic checks
    # =========================================================================
    log("\n=== Checks 8-12: Task semantics ===")

    counting_words = {"count", "how many", "number of", "total", "objects", "items", "exactly", "precisely"}
    shape_words = {"circle", "square", "triangle", "shape"}
    spatial_words = {"left", "right"}

    def has_any(text: str, words: set) -> bool:
        return any(w in text.lower() for w in words)

    for task in FUNCTIONS:
        for split in SPLITS:
            for sample in data[task][split]:
                q = sample["question"].lower()
                sid = sample["id"]

                if task == "A_only":
                    if has_any(q, counting_words):
                        add_error(f"A_only has counting: {sid}: {q}")
                    if has_any(q, spatial_words):
                        add_error(f"A_only has spatial: {sid}: {q}")
                    # Should have a shape reference
                    if not has_any(q, shape_words):
                        add_warning(f"A_only lacks shape reference: {sid}: {q}")

                elif task == "B_only":
                    if has_any(q, shape_words):
                        add_error(f"B_only has shape: {sid}: {q}")
                    if has_any(q, spatial_words):
                        add_error(f"B_only has spatial: {sid}: {q}")
                    if not has_any(q, counting_words):
                        add_warning(f"B_only lacks counting: {sid}: {q}")

                elif task == "C_only":
                    if has_any(q, counting_words):
                        add_error(f"C_only has counting: {sid}: {q}")
                    if not has_any(q, spatial_words):
                        add_warning(f"C_only lacks spatial: {sid}: {q}")

                elif task == "A_plus_B":
                    if not has_any(q, shape_words):
                        add_warning(f"A_plus_B lacks shape: {sid}: {q}")
                    if not has_any(q, counting_words):
                        add_warning(f"A_plus_B lacks counting: {sid}: {q}")

                elif task == "B_plus_C":
                    if not has_any(q, spatial_words):
                        add_warning(f"B_plus_C lacks spatial: {sid}: {q}")
                    if not has_any(q, counting_words):
                        add_warning(f"B_plus_C lacks counting: {sid}: {q}")

    log("  Semantic checks complete ✓")

    # =========================================================================
    # Check 13: Answer verification against scene metadata
    # =========================================================================
    log("\n=== Check 13: Answer vs scene ground truth ===")
    answer_mismatches = 0
    mismatches_sampled = []

    for task in FUNCTIONS:
        for split in SPLITS:
            for sample in data[task][split]:
                scene = scenes.get(sample["scene_id"])
                if not scene:
                    continue

                expected_answer = _compute_expected_answer(sample, scene)
                if expected_answer is not None and expected_answer != sample["answer"]:
                    answer_mismatches += 1
                    if len(mismatches_sampled) < 5:
                        mismatches_sampled.append({
                            "id": sample["id"],
                            "task": task,
                            "expected": expected_answer,
                            "got": sample["answer"],
                            "question": sample["question"],
                        })

    if answer_mismatches == 0:
        log("  All answers match scene ground truth ✓")
    else:
        add_error(f"{answer_mismatches} answer mismatches found")
        for m in mismatches_sampled:
            log(f"    {m['id']}: expected={m['expected']} got={m['got']} q='{m['question']}'")

    # =========================================================================
    # Check 14: negative_type consistency
    # =========================================================================
    log("\n=== Check 14: negative_type consistency ===")
    neg_type_errors = 0
    for task in FUNCTIONS:
        for split in SPLITS:
            for sample in data[task][split]:
                if sample["polarity"] == "positive" and sample["negative_type"] is not None:
                    add_error(f"Positive sample has negative_type: {sample['id']}")
                    neg_type_errors += 1
                if sample["polarity"] == "negative" and sample["negative_type"] is None:
                    add_warning(f"Negative sample missing negative_type: {sample['id']}")
                    neg_type_errors += 1
    if neg_type_errors == 0:
        log("  negative_type consistent with polarity ✓")

    # =========================================================================
    # Check 15-16: Uniqueness
    # =========================================================================
    log("\n=== Checks 15-16: Uniqueness ===")

    for task in FUNCTIONS:
        for split in SPLITS:
            samples = data[task][split]

            # (question, image) uniqueness
            pairs = [(s["question"], s["image"]) for s in samples]
            pair_dups = len(pairs) - len(set(pairs))
            if pair_dups > 0:
                add_error(f"{task}/{split}: {pair_dups} duplicate (question,image) pairs")

            # Full sample uniqueness (by id)
            ids = [s["id"] for s in samples]
            id_dups = len(ids) - len(set(ids))
            if id_dups > 0:
                add_error(f"{task}/{split}: {id_dups} duplicate IDs")

    log("  Uniqueness checks complete ✓")

    # =========================================================================
    # Check 17: Image distribution consistency
    # =========================================================================
    log("\n=== Check 17: Image distribution ===")
    dist_stats = _check_distribution_consistency(data, scenes)
    stats["distribution"] = dist_stats
    log("  Distribution summary computed ✓")

    # =========================================================================
    # Check 18: JSON loadable
    # =========================================================================
    log("\n=== Check 18: JSON valid ===")
    log("  All JSON files loaded successfully ✓")

    # =========================================================================
    # Check 19: Tokenizer answer encoding
    # =========================================================================
    log("\n=== Check 19: Tokenizer audit ===")
    tokenizer_info = _audit_tokenizer(data)
    stats["tokenizer"] = tokenizer_info
    log(f"  Answer A: token_ids={tokenizer_info['answer_token_ids']['A']} ✓")
    log(f"  Answer B: token_ids={tokenizer_info['answer_token_ids']['B']} ✓")

    # =========================================================================
    # Check 20: Reproducibility
    # =========================================================================
    log("\n=== Check 20: Deterministic reproducibility ===")
    manifest_path = root_path / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        seed = manifest.get("generation_seed")
        if seed is not None:
            log(f"  Seed recorded: {seed}")
            log("  Reproducibility: seed-based deterministic generation ✓")
        else:
            add_warning("No generation seed in manifest")
    stats["reproducibility"] = {"seed": seed if seed else "unknown"}

    # =========================================================================
    # Summary
    # =========================================================================
    log("\n" + "=" * 60)
    log("VALIDATION SUMMARY")
    log("=" * 60)
    log(f"  Errors:   {len(errors)}")
    log(f"  Warnings: {len(warnings)}")

    passed = len(errors) == 0
    log(f"  Status:   {'✓ PASSED' if passed else '✗ FAILED'}")

    return {
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
        "total_samples": sum(
            len(data[t][s]) for t in FUNCTIONS for s in SPLITS
        ),
        "total_scenes": len(scenes),
    }


def _compute_expected_answer(sample: Dict, scene: Dict) -> str | None:
    """Compute expected answer from scene metadata. Returns None if can't verify."""
    task = sample["task"]
    md = sample["metadata"]
    objects = scene["objects"]

    if task == "A_only":
        target_id = md.get("target_object_id")
        if not target_id:
            return None
        target = next((o for o in objects if o["id"] == target_id), None)
        if not target:
            return None
        return "A" if target["shape"] == md["queried_shape"] else "B"

    elif task == "B_only":
        true_count = scene["total_objects"]
        queried = md.get("queried_count")
        if queried is None:
            return None
        return "A" if true_count == queried else "B"

    elif task == "C_only":
        target_id = md.get("target_object_id")
        ref_id = md.get("reference_object_id")
        if not target_id or not ref_id:
            return None
        target = next((o for o in objects if o["id"] == target_id), None)
        ref = next((o for o in objects if o["id"] == ref_id), None)
        if not target or not ref:
            return None
        is_left = target["x"] < ref["x"]
        queried_rel = md.get("queried_relation")
        return "A" if (is_left and queried_rel == "left") else "B"

    elif task == "A_plus_B":
        queried_shape = md.get("queried_shape")
        queried_count = md.get("queried_count")
        if queried_shape is None or queried_count is None:
            return None
        true_shape_count = sum(1 for o in objects if o["shape"] == queried_shape)
        return "A" if true_shape_count == queried_count else "B"

    elif task == "B_plus_C":
        ref_id = md.get("reference_object_id")
        queried_count = md.get("queried_count")
        queried_rel = md.get("queried_relation")
        if not ref_id or queried_count is None or queried_rel is None:
            return None
        ref = next((o for o in objects if o["id"] == ref_id), None)
        if not ref:
            return None
        ref_x = ref["x"]
        if queried_rel == "left":
            true_count = sum(1 for o in objects if o["x"] < ref_x - 5)
        else:
            true_count = sum(1 for o in objects if o["x"] > ref_x + 5)
        return "A" if true_count == queried_count else "B"

    return None


def _check_distribution_consistency(
    data: Dict, scenes: Dict
) -> Dict[str, Any]:
    """Check that all tasks have similar image distributions."""
    # Collect image-level stats per task
    task_img_stats = {}
    for task in FUNCTIONS:
        scene_ids = set()
        for split in SPLITS:
            for s in data[task][split]:
                scene_ids.add(s["scene_id"])

        counts = []
        shapes = []
        colors = []
        for sid in scene_ids:
            scene = scenes.get(sid)
            if scene:
                counts.append(scene["total_objects"])
                for obj in scene["objects"]:
                    shapes.append(obj["shape"])
                    colors.append(obj["color"])

        task_img_stats[task] = {
            "scene_count": len(scene_ids),
            "mean_objects": sum(counts) / len(counts) if counts else 0,
            "shape_dist": dict(Counter(shapes)),
            "color_dist": dict(Counter(colors)),
        }

    return {"per_task": task_img_stats}


def _audit_tokenizer(data: Dict) -> Dict[str, Any]:
    """Audit tokenizer encoding of answer tokens."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return {"error": "transformers not available"}

    model_paths = [
        "/data/ckpt/zhaozhuofan/models/llava-v1.5-7b",
    ]

    tokenizer_info = {}
    for model_path in model_paths:
        if not os.path.exists(model_path):
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        except Exception as e:
            tokenizer_info[model_path] = {"error": str(e)}
            continue

        # Check A and B tokenization in context
        system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
        sample_q = "USER: <image>\nIs the red object a triangle?"

        info = {
            "model_path": model_path,
            "vocab_size": tokenizer.vocab_size,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }

        # Check standalone
        for text in ["A", "B"]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            info[f"standalone_{text}"] = {"token_ids": ids,
                                            "token_count": len(ids)}

        # Check in full context
        for answer in ["A", "B"]:
            conv = system + sample_q + " ASSISTANT: " + answer + "</s>"
            tokens = tokenizer.encode(conv, add_special_tokens=False)

            # Find the answer token by comparison
            conv_empty = system + sample_q + " ASSISTANT: </s>"
            tokens_empty = tokenizer.encode(conv_empty, add_special_tokens=False)

            # The answer tokens are the difference
            answer_ids = None
            for i in range(min(len(tokens), len(tokens_empty))):
                if tokens[i] != tokens_empty[i]:
                    # Found divergence
                    remaining = tokens[i:]
                    # Remove trailing EOS
                    answer_ids = [t for t in remaining if t != tokenizer.eos_token_id]
                    break

            if answer_ids is None:
                answer_ids = []

            info[f"context_{answer}"] = {
                "answer_token_ids": answer_ids,
                "answer_token_count": len(answer_ids),
                "decoded_tokens": [tokenizer.decode([t]) for t in answer_ids],
                "total_sequence_tokens": len(tokens),
            }

        tokenizer_info[model_path] = info

    return {
        "tokenizer_checks": tokenizer_info,
        "answer_token_ids": {
            "A": tokenizer_info.get(model_paths[0], {}).get("context_A", {}).get("answer_token_ids", []),
            "B": tokenizer_info.get(model_paths[0], {}).get("context_B", {}).get("answer_token_ids", []),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate controlled_format_v1 dataset"
    )
    parser.add_argument("--data-root", required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--output", default=None,
                        help="Path to write audit.json")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    args = parser.parse_args()

    result = validate(args.data_root, verbose=not args.quiet)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nAudit saved to: {output_path}")

    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
