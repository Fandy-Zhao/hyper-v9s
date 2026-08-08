#!/usr/bin/env python3
"""D6: expand the B+C diagnostic pool (statistical power).

Re-runs the function classifier over the full official TextVQA train+val
with relaxed B+C rules (broader min/max entities, "which number is bigger"
family, "how many more X than Y" with text entities, explicit time
arithmetic). Two tiers are reported:

  - candidate pool: all B+C-classified questions (image-conflict included)
  - clean pool     : questions whose images are disjoint from the B/C
                     training images AND from the original BC_test images

The clean pool is the only one used for evaluation (image isolation per
spec). If the clean pool has < 100 samples the experiment is marked
REAL_DATA_UNDERSPECIFIED and the clean pool is still evaluated as-is.

Outputs metrics/d6_expanded_pool.csv and metrics/d6_expanded_pool_manifest.json.
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/home/zhaozhuofan/Hyper-LlaVA/compose/data/real_p1")
from build_subsets import (  # noqa: E402
    OFFICIAL_TRAIN,
    OFFICIAL_VAL,
    MODALPROMPT_TRAIN,
    MODALPROMPT_VAL,
    _load_ocr_hints,
    _artifact_answer,
    classify,
    majority_answer,
    normalize,
)


def relaxed_bc(question, answer, ocr_tokens):
    """Relaxed B+C classifier (broad min/max, comparisons, time arithmetic)."""
    decision = classify(question, answer, ocr_tokens)
    if decision["function_label"] == "B_plus_C":
        return decision
    q = question.strip().lower().rstrip("?.")
    # broader min/max entities
    if re.search(
        r"what('s| is| are| was| were| s)? the (highest|lowest|biggest|smallest|largest|"
        r"greatest|smaller|bigger|larger|lower|higher|older|youngest|newest|oldest) "
        r"(number|amount|price|score|value|total|sum|year|date|month|day|temperature|"
        r"weight|speed|height|size|level|version|edition|volume|issue|rating|rank|"
        r"reading|capacity|yard line|face views|denomination|cost|time)\b", q):
        return {"function_label": "B_plus_C", "question_type": "compare_minmax",
                "operation": "compare", "reason": []}
    # which number is bigger / what number is larger
    if re.search(r"(which|what) number is (bigger|larger|smaller|greater|lower|higher)\b", q):
        return {"function_label": "B_plus_C", "question_type": "compare_minmax",
                "operation": "compare", "reason": []}
    # how many more/fewer X than Y with text entities
    if re.search(r"how many (more|fewer|less) (\w+ ){0,3}(pages|miles|minutes|hours|dollars|"
                 r"points|years|words|times|letters|days|books|seats|people|items|numbers)\b", q):
        return {"function_label": "B_plus_C", "question_type": "difference",
                "operation": "difference", "reason": []}
    # explicit time arithmetic
    if re.search(r"what time (will|would) (it )?be in \d|minutes (later|earlier|from now)|"
                 r"hours (later|earlier|from now)", q):
        return {"function_label": "B_plus_C", "question_type": "time_arithmetic",
                "operation": "time_arithmetic", "reason": []}
    # difference between with numeric answer
    if re.search(r"difference between|difference of", q) and re.search(r"\d", answer):
        return {"function_label": "B_plus_C", "question_type": "difference",
                "operation": "difference", "reason": []}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    p1_root = Path(args.p1_root)
    output_root = Path(args.output_root)
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)

    hints = {}
    for source in (MODALPROMPT_TRAIN, MODALPROMPT_VAL):
        for (image_id, question), tokens in _load_ocr_hints(source).items():
            hints.setdefault(image_id, []).extend(tokens)

    candidates = []
    for path, split in ((OFFICIAL_TRAIN, "train"), (OFFICIAL_VAL, "val")):
        for item in json.loads(Path(path).read_text(encoding="utf-8"))["data"]:
            image_id = str(item["image_id"])
            question = str(item["question"])
            answers = [str(a) for a in item["answers"]]
            majority = majority_answer(answers)
            if _artifact_answer(majority):
                continue
            ocr = hints.get(image_id, [])
            decision = relaxed_bc(question, majority, ocr)
            if decision is None or decision["function_label"] != "B_plus_C":
                continue
            if not re.search(r"\d", normalize(majority)):
                continue
            candidates.append({
                "sample_id": "{}-{}".format(split, item["question_id"]),
                "question_id": str(item["question_id"]),
                "dataset": "TextVQA",
                "official_split": split,
                "image_id": image_id,
                "image": "images/{}/{}.jpg".format(split, image_id),
                "question": question,
                "answer": majority,
                "answers": answers,
                "function_label": "B_plus_C",
                "question_type": decision["question_type"],
                "operation": decision["operation"],
                "requires_ocr": True,
                "requires_numeric_reasoning": True,
                "required_value_count": 2,
                "construction_source": "official-filtered",
                "diagnostic_pool": True,
            })
    print("candidate B+C pool: {} questions, {} images".format(
        len(candidates), len({c["image_id"] for c in candidates})))

    # training images used by B/C experts (exclude them for the clean pool)
    training_images = set()
    for name in ("B_train", "B_val", "C_train", "C_val"):
        records = json.loads((p1_root / "data" / "records" / "{}.json".format(name)).read_text())
        training_images.update(r["image_id"] for r in records)
    bc_test_images = set()
    for name in ("BC_test", "BC_calib"):
        records = json.loads((p1_root / "data" / "records" / "{}.json".format(name)).read_text())
        bc_test_images.update(r["image_id"] for r in records)

    clean = [c for c in candidates
             if c["image_id"] not in training_images and c["image_id"] not in bc_test_images]
    print("clean (image-isolated) B+C pool: {} questions, {} images".format(
        len(clean), len({c["image_id"] for c in clean})))
    leaked = [c for c in candidates if c["image_id"] in training_images]
    print("leaked (image overlaps B/C training): {}".format(len(leaked)))
    dup = collections.Counter(normalize(c["question"]) for c in clean)
    print("question near-duplicates in clean pool: {}".format(
        sum(1 for v in dup.values() if v > 1)))

    # question-type distribution of clean pool
    types = collections.Counter(c["question_type"] for c in clean)
    print("clean pool types:", dict(types))

    manifest = {
        "candidate_count": len(candidates),
        "clean_count": len(clean),
        "leaked_count": len(leaked),
        "clean_types": dict(types),
        "clean_question_duplicates": sum(1 for v in dup.values() if v > 1),
        "note": ("clean pool is image-isolated from B/C training and the "
                 "original BC test; used for the D6 diagnostic evaluation"),
        "marked_under_specified": len(clean) < 100,
    }
    (output_root / "metrics" / "d6_expanded_pool_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    import csv
    with (output_root / "metrics" / "d6_expanded_pool.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clean[0].keys()) if clean else ["sample_id"])
        writer.writeheader()
        writer.writerows(clean)
    # also write the clean pool as eval records (JSON) for evaluation
    (output_root / "data" / "records_d6_clean.json").parent.mkdir(parents=True, exist_ok=True)
    (output_root / "data" / "records_d6_clean.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=1), encoding="utf-8")
    # image symlinks for the clean pool
    for c in clean:
        link = output_root / "data" / "images_d6" / c["image_id"][:2] / "{}.jpg".format(c["image_id"])
        source = Path("/data/ckpt/zhangyanqin/project/ModalPrompt/datasets/TextVQA/train_images/{}.jpg".format(c["image_id"]))
        if not link.exists() and source.exists():
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(source)
    print("marked_under_specified:", manifest["marked_under_specified"])


if __name__ == "__main__":
    main()
