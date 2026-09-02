"""Deterministic V7 formal train/validation split builder (six UCIT tasks).

Carves a 256-unit validation set out of each task's ORIGINAL training split
(seed 42, per-task ``random.Random(42)`` streams) and writes the remainder
as the formal training split. Test files are only read for exclusion and
overlap auditing and are NEVER written.

Split contracts (binding user decisions; see docs/reports/V7_FORMAL_DATA_AUDIT.md):

* exactly 256 validation samples per task, carved disjointly with seed 42;
  test data never participates and test files are never substituted.
* no leakage group appears in both formal train and validation:
  - ImageNet-R: class-stratified two-stage largest remainder over the 200
    wnid classes (1 seat/class, then extra seats in ascending class size,
    smaller wnid on ties). Classes with n < 42 keep a single seat so every
    class keeps >= 40 members in formal train; total stays 256.
  - ArxivQA / IconQA / CLEVR: shuffled walk over the original records,
    skipping an already-selected image+question identity or native id; the
    whole exact-identity twin group leaves formal train with its member.
  - Flickr30k: whole images; exactly 256 images, reference-rich first
    (descending available-caption count, then image path); ALL caption
    records of those images leave formal train.
  - VizWiz: whole images only; a shuffled walk accumulates whole image
    groups to exactly 256 (each VizWiz image contributes one question row,
    so whole-group accumulation reaches 256 exactly - groups are never
    split to hit the target).
* Caption tasks (VizWiz, Flickr30k) emit one instruction record per selected
  image (mirroring the official test_3000 layout) and a COCO annotation
  whose references are every available sibling caption of the image
  (deduplicated, k <= 5; the original 5-reference Flickr30k annotation is
  not available on this machine - deviation recorded, never fabricated).
  COCO images[] carry positional ids 1..N in instruction-file order with
  file_name = basename, mirroring val_coco_type_3000.json exactly.

Output schema (uniform, no top-level ``text`` key):

* validation record: source record verbatim except the identifier is
  rewritten to ``v7_t{task}_{val}_{position}`` (top-level ``question_id``,
  val-unique) and a top-level ``answer`` equal to the gpt value verbatim is
  added. Original ids are preserved in per-task ``provenance_build.json``.
* formal-train record: source record verbatim with the identifier rewritten
  to ``v7_t{task}_{train}_{position}`` (top-level ``id``; duplicate-prone
  native ids never reach the pipeline).

All writes are atomic (tmp + fsync + os.replace). ``--command check``
re-verifies existing outputs without rebuilding; ``--command
scorer-preflight`` runs the real per-task scorers on two dummy prediction
rows against the built annotations (no model, no GPU).
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from compose.data.records import answer_text, question_text
from compose.v7.provenance import audit_split_isolation, sha256_file
from compose.v7.provenance import _record_identity as record_identity

VALIDATION_TARGET = 256
IMAGENET_R_MIN_POST_CARVE = 40

#: V7 task index by task name (configs/v7_ucit_formal.yaml task order).
TASK_INDEX = {
    "ImageNet-R": 0,
    "ArxivQA": 1,
    "VizWiz": 2,
    "IconQA": 3,
    "CLEVR": 4,
    "Flickr30k": 5,
}

#: Caption tasks are scored with COCO-style annotations and therefore emit
#: a validation_coco.json alongside validation.json.
COCO_TASKS = ("VizWiz", "Flickr30k")


def native_id(record):
    return record.get("id", record.get("question_id"))


def write_json_atomic(path, payload, compact=False):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            if compact:
                json.dump(payload, handle, ensure_ascii=False)
            else:
                json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def content_signature(record):
    """(image, question, answer) triple; multiplicity-preserving and
    id-independent - the content the group rules operate on."""
    return (
        str(record.get("image", "")),
        question_text(record).strip(),
        answer_text(record).strip(),
    )


def rewrite_validation_record(record, task_index, position):
    return {
        "image": record["image"],
        "conversations": record["conversations"],
        "question_id": "v7_t{}_{}_{}".format(int(task_index), "val", int(position)),
        "answer": answer_text(record),
    }


def rewrite_train_record(record, task_index, position):
    rebuilt = {"image": record["image"], "conversations": record["conversations"]}
    rebuilt["id"] = "v7_t{}_{}_{}".format(int(task_index), "train", int(position))
    return rebuilt


def _wnid_class(record):
    parts = Path(str(record["image"])).parts
    if len(parts) < 2:
        raise ValueError("ImageNet-R record without wnid directory: {}".format(record["image"]))
    return parts[-2]


def imagenet_r_seats(class_sizes, extra=VALIDATION_TARGET - 200,
                     min_post_carve=IMAGENET_R_MIN_POST_CARVE):
    """Two-stage largest remainder over wnid classes.

    1 seat per class for all 200 classes, then ``extra`` second seats to the
    classes with the largest remaining fraction ((n-1)/n, monotonic in n, so
    ascending class size) among classes that keep >= min_post_carve members
    in formal train (n - 2 >= min_post_carve); equal-size ties break toward
    the smaller wnid. Returns {wnid: seats} with 1 <= seats <= 2.
    """
    by_name = {str(wnid): int(size) for wnid, size in class_sizes.items()}
    ordered = sorted(by_name)
    if len(ordered) != 200:
        raise ValueError("ImageNet-R class audit expects 200 classes, got {}".format(len(ordered)))
    if any(size < 1 for size in by_name.values()):
        raise ValueError("ImageNet-R has an empty class")
    seats = {wnid: 1 for wnid in ordered}
    candidates = sorted(
        (wnid for wnid in ordered if by_name[wnid] - 2 >= min_post_carve),
        key=lambda wnid: (by_name[wnid], wnid),  # ascending size; smaller wnid first
    )
    if len(candidates) < extra:
        raise ValueError(
            "not enough classes can lose a second member while keeping >= {} in train".format(
                min_post_carve
            )
        )
    for wnid in candidates[:extra]:
        seats[wnid] = 2
    return seats


def _shuffled(sequence, rng):
    copy = list(sequence)
    rng.shuffle(copy)
    return copy


def select_imagenet_r(records, rng, *, target=VALIDATION_TARGET):
    """Returns (selected record indexes, {wnid: seats})."""
    classes = defaultdict(list)
    for index, record in enumerate(records):
        classes[_wnid_class(record)].append(index)
    seats = imagenet_r_seats({wnid: len(indexes) for wnid, indexes in classes.items()})
    selected = set()
    for wnid in sorted(classes):
        shuffled = _shuffled(classes[wnid], rng)
        selected.update(shuffled[: seats[wnid]])
    if len(selected) != target:
        raise ValueError("ImageNet-R stratification selected {} != {}".format(len(selected), target))
    return selected, seats


def select_walk(records, rng, excluded_identities=(), *, target=VALIDATION_TARGET):
    """Shuffled-index walk with identity / native-id dedupe for question-bank
    tasks (ArxivQA, IconQA, CLEVR). Records whose image+question identity is
    already selected, whose native id is already selected, or which match a
    test-file identity are skipped. Returns selected indexes (sorted)."""
    excluded = set(excluded_identities)
    selected_indexes = []
    seen_identities = set()
    seen_native = set()
    order = _shuffled(range(len(records)), rng)
    for index in order:
        if len(selected_indexes) >= target:
            break
        if record_identity(records[index]) in excluded:
            continue
        identity = record_identity(records[index])
        if identity in seen_identities:
            continue
        identifier = str(native_id(records[index]))
        if identifier in seen_native:
            continue
        selected_indexes.append(index)
        seen_identities.add(identity)
        seen_native.add(identifier)
    if len(selected_indexes) != target:
        raise ValueError(
            "walk selected {} of {} (duplicates exhausted the source)".format(
                len(selected_indexes), target
            )
        )
    return sorted(selected_indexes)


def group_records_by_image(records):
    """image path -> row indexes in original file order."""
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[str(record.get("image", ""))].append(index)
    return groups


def select_caption_images(records, rng, *, target=VALIDATION_TARGET,
                          ref_rich_first=False, excluded_identities=()):
    """Whole-image group selection for caption tasks (VizWiz, Flickr30k).

    Never splits an image group. With ``ref_rich_first`` the shuffled images
    are re-sorted by descending available-caption count then image path
    (Flickr30k: the original 5-reference annotation is unavailable, so the
    available sibling captions are maximized). Otherwise a plain shuffled
    walk accumulates whole groups to the target (VizWiz: every group
    contributes exactly one question row, so the accumulated count reaches
    the target exactly). Returns the selected image paths."""
    groups = group_records_by_image(records)
    excluded = set(excluded_identities)
    eligible = []
    for image, indexes in groups.items():
        if any(record_identity(records[index]) in excluded for index in indexes):
            continue
        eligible.append(image)
    order = _shuffled(eligible, rng)
    if ref_rich_first:
        order.sort(key=lambda image: (-len(groups[image]), image))
    selected = order[:target]
    if len(selected) != target:
        raise ValueError("caption group selection found {} of {} whole images".format(
            len(selected), target
        ))
    return list(selected)


def removed_by_identity(records, val_indexes):
    identities = {record_identity(records[index]) for index in val_indexes}
    return sorted(
        index for index, record in enumerate(records) if record_identity(record) in identities
    )


def build_task_split(*, task_name, task_index, source_train_path, test_path,
                     seed, target=VALIDATION_TARGET):
    """Constructs validation + formal-train payloads for one task in memory.

    Returns a dict with the output payloads and full audit/provenance data.
    Never touches disk itself; ``write_task_outputs`` persists it."""
    train_records = json.loads(Path(source_train_path).read_text(encoding="utf-8"))
    if not isinstance(train_records, list) or not train_records:
        raise ValueError("source train split invalid: {}".format(source_train_path))
    test_records = json.loads(Path(test_path).read_text(encoding="utf-8")) if test_path else []
    test_identities = [record_identity(record) for record in test_records]
    rng = random.Random(seed)
    caption_task = task_name in COCO_TASKS
    stats = {}

    if task_name == "ImageNet-R":
        val_indexes, seats = select_imagenet_r(train_records, rng, target=target)
        stats["class_seats"] = seats
        stats["stratification"] = "two-stage largest remainder over 200 wnid classes"
        removed_indexes = removed_by_identity(train_records, val_indexes)
    elif caption_task:
        groups = group_records_by_image(train_records)
        selected_images = select_caption_images(
            train_records, rng, target=target,
            ref_rich_first=(task_name == "Flickr30k"),
            excluded_identities=test_identities,
        )
        refs_by_image = {
            image: dedupe_keep_first(
                answer_text(train_records[index]).strip() for index in groups[image]
            )
            for image in selected_images
        }
        val_indexes = sorted(groups[image][0] for image in selected_images)
        stats["num_validation_images"] = len(selected_images)
        stats["num_validation_questions"] = len(val_indexes)
        stats["num_validation_reference_captions"] = sum(
            len(refs) for refs in refs_by_image.values()
        )
        stats["ref_rich_first"] = task_name == "Flickr30k"
        stats["stratification"] = "whole-image groups (never split)"
        removed_indexes = sorted(
            index for image in selected_images for index in groups[image]
        )
    else:
        val_indexes = select_walk(
            train_records, rng, excluded_identities=test_identities, target=target
        )
        stats["stratification"] = "shuffled identity/native-id dedup walk"
        removed_indexes = removed_by_identity(train_records, val_indexes)

    if len(val_indexes) != target:
        raise ValueError(
            "task {} selected {} validation samples (expected {})".format(
                task_name, len(val_indexes), target
            )
        )
    if len(set(val_indexes)) != len(val_indexes):
        raise ValueError("duplicated validation index for {}".format(task_name))
    if removed_indexes != sorted(set(removed_indexes)):
        raise ValueError("duplicated removal index for {}".format(task_name))

    val_records = [
        rewrite_validation_record(train_records[index], task_index, position)
        for position, index in enumerate(sorted(val_indexes))
    ]
    kept = remove_from_training(train_records, removed_indexes)
    train_records_out = [
        rewrite_train_record(record, task_index, position)
        for position, record in enumerate(kept)
    ]

    outputs = {}
    if caption_task:
        # COCO annotation mirrors the instruction-record order: images[] id
        # 1..N positional with file_name = basename; references are every
        # deduplicated sibling caption of the selected image.
        images = []
        annotations = []
        annotation_id = 1
        for position, index in enumerate(val_indexes, start=1):
            image_path = str(train_records[index]["image"])
            images.append({"id": position, "file_name": Path(image_path).name})
            for caption in refs_by_image[image_path]:
                annotations.append({
                    "id": annotation_id,
                    "image_id": position,
                    "caption": caption,
                })
                annotation_id += 1
        coco = {"images": images, "annotations": annotations}
        if task_name == "VizWiz":
            # mirrors the official VizWiz val_coco file; the official
            # Flickr30k files carry no categories key (eval_caption
            # normalizes either way).
            coco["categories"] = [{"id": 1, "name": "captioning"}]
        outputs["validation_coco"] = coco

    return {
        "task_name": task_name,
        "task_index": task_index,
        "seed": seed,
        "source_train": {
            "path": str(Path(source_train_path).resolve()),
            "sha256": sha256_file(source_train_path),
            "record_count": len(train_records),
        },
        "test_path": str(Path(test_path).resolve()) if test_path else None,
        "validation": {
            "records": val_records,
            "count": len(val_records),
            "val_original_indexes": sorted(val_indexes),
        },
        "train": {"records": train_records_out, "count": len(train_records_out)},
        "removed_indexes": removed_indexes,
        "removed_count": len(removed_indexes),
        "stats": stats,
        "outputs": outputs,
    }


def remove_from_training(records, removed_indexes):
    removed_set = set(removed_indexes)
    return [record for index, record in enumerate(records) if index not in removed_set]


def dedupe_keep_first(values):
    seen = set()
    kept = []
    for value in values:
        if value not in seen:
            seen.add(value)
            kept.append(value)
    return kept


def _assert_no_missing_images(records, image_folder, label):
    folder = Path(image_folder)
    missing = [
        str(record["image"])
        for record in records
        if not (folder / str(record["image"]).lstrip("/")).is_file()
    ]
    if missing:
        raise ValueError("{} records reference missing images ({}): {}".format(
            label, len(missing), missing[:3]
        ))


def validate_built_task(payload, *, image_folder, caption_task):
    task_name = payload["task_name"]
    val_records = payload["validation"]["records"]
    train_records = payload["train"]["records"]
    if len(val_records) != VALIDATION_TARGET:
        raise ValueError("validation count != 256 for {}".format(task_name))
    question_ids = [record["question_id"] for record in val_records]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("validation question_id not unique for {}".format(task_name))
    if any("text" in record or "id" in record for record in val_records):
        raise ValueError("validation records must carry question_id only (no id/text)")
    for record in val_records:
        if record["answer"] != record["conversations"][-1]["value"]:
            raise ValueError("validation answer != gpt value for {}".format(task_name))
        if not record["conversations"] or record["conversations"][0].get("from") != "human":
            raise ValueError("validation record schema broken for {}".format(task_name))
    _assert_no_missing_images(
        train_records + val_records, image_folder, "{} output splits".format(task_name)
    )
    if caption_task:
        coco = payload["outputs"]["validation_coco"]
        if len(coco["images"]) != len(val_records):
            raise ValueError("caption COCO image count mismatch for {}".format(task_name))
        linked = set()
        expected_images = {record["image"] for record in val_records}
        for position, record in enumerate(val_records, start=1):
            image_entry = coco["images"][position - 1]
            if image_entry["id"] != position:
                raise ValueError("caption COCO id order mismatch for {}".format(task_name))
            if image_entry["file_name"] != Path(str(record["image"])).name:
                raise ValueError("caption COCO file_name mismatch for {}".format(task_name))
            linked.add(str(record["image"]))
        if linked != expected_images:
            raise ValueError("caption COCO linkage mismatch for {}".format(task_name))
        # Caption gate: train == original minus the removed rows (moved,
        # never duplicated); every removed row is either the validation
        # representative of its image or its caption is retained in that
        # image's COCO references (the only place removed caption rows may
        # migrate to); and no COCO reference is ever fabricated (each ref is
        # a verbatim caption of a removed sibling row).
        original = json.loads(Path(payload["source_train"]["path"]).read_text(encoding="utf-8"))
        original_counts = Counter(content_signature(record) for record in original)
        kept_counts = Counter(content_signature(record) for record in train_records)
        removed_sigs = Counter(
            content_signature(original[index]) for index in payload["removed_indexes"]
        )
        if kept_counts != original_counts - removed_sigs:
            raise ValueError("caption train rows were not exactly the non-removed set")
        groups = group_records_by_image(original)
        val_by_image = {str(record["image"]): record for record in val_records}
        # annotation references per validation position (position i == coco
        # images[i-1] == val_records[i-1], already asserted above)
        refs_by_position = defaultdict(list)
        for item in coco["annotations"]:
            refs_by_position[item["image_id"]].append(item["caption"])
        expected_refs = {}
        for position, record in enumerate(val_records, start=1):
            image = str(record["image"])
            captions = [answer_text(original[index]).strip() for index in groups[image]]
            expected_refs[image] = set(dedupe_keep_first(captions))
            actual = set(refs_by_position.get(position, []))
            if actual != expected_refs[image]:
                raise ValueError(
                    "caption COCO refs mismatch for {} image {}".format(task_name, image)
                )
        for index in payload["removed_indexes"]:
            record = original[index]
            image = str(record["image"])
            signature = content_signature(record)
            representative = val_by_image.get(image)
            if representative is not None and content_signature(representative) == signature:
                continue  # this row IS the validation instruction row
            if signature[2] not in expected_refs.get(image, set()):
                raise ValueError(
                    "removed caption row neither in validation nor refs for {}: {}".format(
                        task_name, signature[2]
                    )
                )
        return
    # Non-caption gate: formal train is EXACTLY the original split minus the
    # removed rows (records are moved, never duplicated), and every removed
    # row is content-identical (image, question, answer) to its identity's
    # validation representative - whole exact-identity twin groups leave
    # formal train together and lose no content (their duplicate row is a
    # pure content copy of the representative that stayed in validation).
    original = json.loads(Path(payload["source_train"]["path"]).read_text(encoding="utf-8"))
    original_counts = Counter(content_signature(record) for record in original)
    kept_counts = Counter(content_signature(record) for record in train_records)
    removed_sigs = Counter(
        content_signature(original[index]) for index in payload["removed_indexes"]
    )
    if kept_counts != original_counts - removed_sigs:
        raise ValueError("{} formal train is not exactly original minus removed".format(task_name))
    representatives = {}
    for index in payload["validation"]["val_original_indexes"]:
        representatives[record_identity(original[index])] = content_signature(original[index])
    for index in payload["removed_indexes"]:
        row = original[index]
        signature = content_signature(row)
        if representatives.get(record_identity(row)) != signature:
            raise ValueError(
                "{} removed row is not the content copy of its validation "
                "representative (index {})".format(task_name, index)
            )
    if task_name == "ImageNet-R":
        remaining = Counter(_wnid_class(record) for record in train_records)
        if len(remaining) != 200:
            raise ValueError("ImageNet-R formal train lost a class")
        if min(remaining.values()) < IMAGENET_R_MIN_POST_CARVE:
            raise ValueError("ImageNet-R post-carve class gate failed")


def write_task_outputs(payload, validation_root, train_root):
    """Persists validation.json (+ validation_coco.json), train.json and the
    per-task provenance_build.json. Returns the provenance payload."""
    task_name = payload["task_name"]
    val_dir = Path(validation_root) / task_name
    train_dir = Path(train_root) / task_name
    validation_path = val_dir / "validation.json"
    write_json_atomic(validation_path, payload["validation"]["records"])
    train_path = train_dir / "train.json"
    write_json_atomic(train_path, payload["train"]["records"], compact=True)
    provenance = {
        "schema_version": 1,
        "task": task_name,
        "task_index": payload["task_index"],
        "seed": payload["seed"],
        "builder_file_sha256": sha256_file(Path(__file__)),
        "source_train": payload["source_train"],
        "test_path": payload["test_path"],
        "validation": {
            "path": str(validation_path.resolve()),
            "record_count": payload["validation"]["count"],
            "file_sha256": sha256_file(validation_path),
            "original_record_indexes": payload["validation"]["val_original_indexes"],
        },
        "train": {
            "path": str(train_path.resolve()),
            "record_count": payload["train"]["count"],
            "file_sha256": sha256_file(train_path),
            "removed_record_indexes": payload["removed_indexes"],
            "removed_count": payload["removed_count"],
        },
        "stats": payload["stats"],
    }
    if "validation_coco" in payload["outputs"]:
        coco_path = val_dir / "validation_coco.json"
        write_json_atomic(coco_path, payload["outputs"]["validation_coco"], compact=True)
        provenance["validation_coco"] = {
            "path": str(coco_path.resolve()),
            "file_sha256": sha256_file(coco_path),
            "images": len(payload["outputs"]["validation_coco"]["images"]),
            "annotations": len(payload["outputs"]["validation_coco"]["annotations"]),
        }
    write_json_atomic(val_dir / "provenance_build.json", provenance)
    return provenance


def _task_specs(config_path):
    import yaml

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    specs = {}
    for task in config["tasks"]:
        specs[task["name"]] = {
            "train_file": task["train_file"],
            "test_file": task["test_file"],
            "validation_file": task["validation_file"],
        }
    return config["data"]["image_folder"], specs


def _output_roots(args):
    validation_root = Path(args.ucit_root) / "v7_validation"
    train_root = Path(args.train_output_root or args.ucit_root) / "v7_train"
    return validation_root, train_root


def cmd_build(args):
    image_folder, specs = _task_specs(args.config)
    validation_root, train_root = _output_roots(args)
    for task in args.tasks:
        if task not in specs:
            raise ValueError("unknown task {} (config tasks: {})".format(task, sorted(specs)))
        payload = build_task_split(
            task_name=task,
            task_index=TASK_INDEX[task],
            source_train_path=specs[task]["train_file"],
            test_path=specs[task]["test_file"],
            seed=args.seed,
        )
        validate_built_task(payload, image_folder=image_folder,
                            caption_task=task in COCO_TASKS)
        provenance = write_task_outputs(payload, validation_root, train_root)
        print(
            "[built] {} val={} train={} removed={} refs={}".format(
                task,
                payload["validation"]["count"],
                payload["train"]["count"],
                payload["removed_count"],
                payload["stats"].get("num_validation_reference_captions", "-"),
            )
        )
        print(
            "        validation sha256 {}".format(provenance["validation"]["file_sha256"])
        )


def cmd_check(args):
    image_folder, specs = _task_specs(args.config)
    validation_root, train_root = _output_roots(args)
    for task in args.tasks:
        val_path = validation_root / task / "validation.json"
        train_path = train_root / task / "train.json"
        test_path = specs[task]["test_file"]
        for path, label in ((val_path, "validation"), (train_path, "formal train"),
                            (Path(test_path), "test")):
            if not path.is_file():
                raise ValueError("{} split missing: {}".format(label, path))
        payload = build_task_split(
            task_name=task,
            task_index=TASK_INDEX[task],
            source_train_path=specs[task]["train_file"],
            test_path=test_path,
            seed=args.seed,
        )
        validate_built_task(payload, image_folder=image_folder,
                            caption_task=task in COCO_TASKS)
        overlap = audit_split_isolation(str(train_path), str(val_path), str(test_path))
        for pair, checks in overlap["overlap_checks"].items():
            if checks["image_question_overlap"] or checks["normalized_record_overlap"]:
                raise ValueError("{} overlap leak on {}: {}".format(task, pair, checks))
        print(
            "[ok] {} train-val-test isolated (val-vs-test image overlap {})".format(
                task,
                overlap["overlap_checks"]["validation_vs_test"]["image_overlap"],
            )
        )


def cmd_scorer_preflight(args):
    image_folder, specs = _task_specs(args.config)
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).parent)] + env.get("PATH", "").split(os.pathsep)
    )
    work = Path(args.work_root) if args.work_root else Path(tempfile.mkdtemp(prefix="v7_preflight_"))
    work.mkdir(parents=True, exist_ok=True)
    summary = {}
    for task in args.tasks:
        val_path = specs[task]["validation_file"]
        if not Path(val_path).is_file():
            raise ValueError("validation file missing: {}".format(val_path))
        records = json.loads(Path(val_path).read_text(encoding="utf-8"))
        pred_file = work / "{}_pred.jsonl".format(task)
        with open(pred_file, "w", encoding="utf-8") as handle:
            for record in records[:2]:
                handle.write(json.dumps({
                    "question_id": record["question_id"],
                    "prompt": question_text(record),
                    "text": "dummy preflight prediction",
                    "model_id": "preflight",
                }, ensure_ascii=False) + "\n")
        annotation = Path(val_path).parent / "validation_coco.json" if task in COCO_TASKS else Path(val_path)
        out = work / "{}_metric.json".format(task)
        result = subprocess.run(
            [
                sys.executable, "-m", "compose.eval.v7_validation_metric",
                "--task-index", str(TASK_INDEX[task]),
                "--annotation-file", str(annotation),
                "--predictions-file", str(pred_file),
                "--work-root", str(work / task),
                "--output", str(out),
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "scorer preflight failed for {}:\n{}".format(task, result.stderr[-3000:])
            )
        metric = json.loads(out.read_text(encoding="utf-8"))
        summary[task] = {
            "scorer": metric.get("scorer"),
            "metric": metric.get("metric"),
            "value": metric.get("value"),
            "annotation": str(annotation),
        }
        print(
            "[ok] {} scorer preflight: {} = {}".format(task, metric.get("metric"),
                                                       metric.get("value"))
        )
    write_json_atomic(work / "scorer_preflight.json", summary)
    print("[done] preflight summary: {}".format(work / "scorer_preflight.json"))


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", choices=("build", "check", "scorer-preflight"),
                        default="build", help="action (default: build)")
    parser.add_argument("--config", default="configs/v7_ucit_formal.yaml")
    parser.add_argument("--ucit-root", default="/data/dataset/zhaozhuofan/UCIT")
    parser.add_argument("--train-output-root", default=None,
                        help="parent of the v7_train output directory (default: ucit-root)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="*", default=sorted(TASK_INDEX),
                        help="task names (default: all six)")
    parser.add_argument("--work-root", default=None,
                        help="scorer-preflight scratch directory")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.command == "build":
        cmd_build(args)
    elif args.command == "check":
        cmd_check(args)
    elif args.command == "scorer-preflight":
        cmd_scorer_preflight(args)
    else:  # pragma: no cover
        raise ValueError("unknown command {}".format(args.command))


if __name__ == "__main__":
    main()
