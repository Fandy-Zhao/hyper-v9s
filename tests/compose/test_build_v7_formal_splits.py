"""Unit tests for the deterministic V7 formal split builder.

Selection and validation logic only - never touches /data. Synthetic
records and a tmp image folder stand in for the UCIT files; the
real-filesystem build is exercised separately (Phase D build + --check).

Synthetic data mirrors the audited real-data properties: question-bank
rows (ArxivQA/IconQA/CLEVR/ImageNet-R) have one distinct question per
record; multi-row images only occur as exact content-identical twins with
fresh ids. Caption rows (VizWiz/Flickr30k) share one question string per
image with one answer/caption row per sibling (k <= 5).
"""

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from compose.data.build_v7_formal_splits import (
    VALIDATION_TARGET,
    build_task_split,
    content_signature,
    group_records_by_image,
    imagenet_r_seats,
    select_caption_images,
    select_imagenet_r,
    select_walk,
    validate_built_task,
    write_task_outputs,
)
from compose.v7.provenance import _record_identity


def conv_record(image, question, answer, native_value=None):
    record = {
        "image": image,
        "conversations": [
            {"from": "human", "value": "<image>\n" + question},
            {"from": "gpt", "value": answer},
        ],
    }
    if native_value is not None:
        record["id"] = str(native_value)
    return record


def dump(records, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle)


def _question_bank(n=5000, twins=5, dup_qids=3):
    """n records with one distinct question each (identity == row), plus
    ``twins`` exact content-identical copies (fresh id, same image/question/
    answer). ``dup_qids`` native-id collisions: rows (2i, 2i+1) share an id
    while remaining distinct identities."""
    records = []
    image_count = 0
    for index in range(n):
        if index % 3 == 0:
            image_count += 1
        records.append(conv_record(
            "ArxivQA/images/doc_{}.jpg".format(image_count),
            "question-{}?".format(index),
            "answer-{}".format(index),
            native_value=index,
        ))
    for index in range(twins):
        twin = records[index]
        records.append(conv_record(
            twin["image"], "question-{}?".format(index),
            twin["conversations"][-1]["value"], native_value=100000 + index,
        ))
    for index in range(dup_qids):
        records[2 * index]["id"] = "shared-id-{}".format(index)
        records[2 * index + 1]["id"] = "shared-id-{}".format(index)
    return records


class TestImagenetRSeats:
    def test_200_classes_1_or_2_seats_total_256(self):
        # two tiny classes (41) can never take a second seat (post-carve >= 40)
        sizes = {str(i).zfill(9): 41 for i in range(2)}
        sizes.update({str(i).zfill(9): 50 + i for i in range(2, 200)})
        assert len(sizes) == 200
        seats = imagenet_r_seats(sizes)
        assert sum(seats.values()) == VALIDATION_TARGET
        assert all(1 <= seats[wnid] <= 2 for wnid in sizes)
        for wnid, size in sizes.items():
            assert size - seats[wnid] >= 40

    def test_second_seats_prefer_smaller_classes_but_keep_40(self):
        sizes = {str(i).zfill(9): 41 for i in range(2)}
        for i in range(2, 200):
            sizes[str(i).zfill(9)] = 300
        seats = imagenet_r_seats(sizes)
        for wnid in [str(i).zfill(9) for i in range(2)]:
            assert seats[wnid] == 1  # n=41 may only lose one member
        two = [wnid for wnid, seat in seats.items() if seat == 2]
        assert len(two) == 56
        assert all(sizes[wnid] - 2 >= 40 for wnid in two)

    def test_tie_breaks_toward_smaller_wnid(self):
        sizes = {str(i).zfill(9): 42 for i in range(2)}
        for i in range(2, 200):
            sizes[str(i).zfill(9)] = 10000
        seats = imagenet_r_seats(sizes)
        two = sorted(wnid for wnid, seat in seats.items() if seat == 2)
        # both 42-classes tie (smallest size); the smaller wnid wins first
        assert two[:2] == [str(0).zfill(9), str(1).zfill(9)]
        assert sizes[two[0]] == 42 and sizes[two[1]] == 42

    def test_deterministic_and_validation_errors(self):
        sizes = {str(i).zfill(9): 41 + i for i in range(200)}
        assert imagenet_r_seats(sizes) == imagenet_r_seats(sizes)
        with pytest.raises(ValueError, match="expects 200"):
            imagenet_r_seats({str(i): 50 for i in range(199)})
        with pytest.raises(ValueError, match="empty class"):
            imagenet_r_seats({str(i).zfill(9): 50 for i in range(199)} | {"x": 0})

    def test_selection_stratified_over_200_classes(self):
        sizes = {str(i).zfill(9): 41 for i in range(2)}
        for i in range(2, 200):
            sizes[str(i).zfill(9)] = 60
        records = []
        for wnid, size in sorted(sizes.items()):
            for index in range(size):
                records.append(conv_record(
                    "ImageNet-R/train/{}/img_{}.jpg".format(wnid, index),
                    "class question", "label-" + wnid, native_value=index,
                ))
        selected, seats = select_imagenet_r(records, random.Random(42))
        assert len(selected) == VALIDATION_TARGET
        per_class = Counter(records[index]["image"].split("/")[2] for index in selected)
        assert per_class == {wnid: seats[wnid] for wnid in sizes}


def _imagenet_r_record(wnid, index):
    return conv_record(
        "ImageNet-R/train/{}/img_{}.jpg".format(wnid, index),
        "class question", "label-" + wnid,
        native_value="{}/{}".format(wnid, index),
    )


class TestWalk:
    def test_selection_dedupe_twin_and_native_id(self):
        records = _question_bank(n=2000, twins=3, dup_qids=2)
        selected = select_walk(records, random.Random(42))
        assert len(selected) == VALIDATION_TARGET
        identities = [_record_identity(records[index]) for index in selected]
        assert len(set(identities)) == len(identities)
        native = [records[index]["id"] for index in selected]
        assert len(set(native)) == len(native)
        # an exact twin never shares validation with its content copy
        twin_identity = _record_identity(records[0])
        assert sum(_record_identity(records[i]) == twin_identity
                   for i in selected) <= 1

    def test_selection_excludes_test_identities(self):
        # self-verifying counterfactual: pick a row the unexcluded walk
        # accepts, then exclude exactly its identity and require the walk to
        # still fill 256 slots without it (identity check, never positional).
        records = _question_bank(n=600, twins=0, dup_qids=0)
        rng = random.Random(42)
        base = select_walk(records, rng)
        victim = base[0]
        excluded = [_record_identity(records[victim])]
        selected = select_walk(records, random.Random(42),
                               excluded_identities=excluded)
        assert len(selected) == VALIDATION_TARGET
        assert victim not in selected

    def test_walk_exhaustion_raises(self):
        records = _question_bank(n=300, twins=0, dup_qids=0)
        with pytest.raises(ValueError, match="selected"):
            select_walk(records, random.Random(1), target=500)


def _caption_records(k_list, task="Flickr30k", dup_caption_image=1):
    """One image per k in ``k_list``; sibling rows share the question string,
    answers differ per row; the group for ``dup_caption_image`` duplicates
    its first caption string on its last row (dedupe exercised)."""
    records = []
    prefix = "Flickr30k/train/{}.jpg" if task == "Flickr30k" else "VizWiz/train/{}.jpg"
    for number, k in enumerate(k_list, start=1):
        if number == dup_caption_image and k < 3:
            k = 3  # ensure room for the duplicate
        for row in range(k):
            caption = "caption {} #{}".format(number, row)
            if number == dup_caption_image and row == k - 1:
                caption = "caption {} #0".format(number)
            records.append(conv_record(
                prefix.format(1000000 + number),
                "What is happening in the image?", caption,
                native_value=1000000 + number,
            ))
    return records


class TestCaptionSelection:
    def test_flickr_ref_rich_first_keeps_whole_groups(self):
        # 8 images k=5, 12 k=4, 40 k=3, 60 k=2, 80 k=1: the 10 selected are
        # the 8 richest plus the 2 smallest-path k=4 images - deterministic
        # by (-k, path), independent of the shuffle.
        k_list = [5] * 8 + [4] * 12 + [3] * 40 + [2] * 60 + [1] * 80
        records = _caption_records(k_list)
        selected = select_caption_images(
            records, random.Random(42), target=10, ref_rich_first=True
        )
        groups = group_records_by_image(records)
        assert len(selected) == 10
        ks = sorted(len(groups[image]) for image in selected)
        assert ks == [4, 4] + [5] * 8
        # path tie-break is ascending: the two k=4 are images 9 and 10
        assert selected == [
            "Flickr30k/train/{}.jpg".format(1000000 + number) for number in range(1, 11)
        ]
        again = select_caption_images(
            records, random.Random(42), target=10, ref_rich_first=True
        )
        assert again == selected

    def test_vizwiz_plain_walk_keeps_whole_groups(self):
        records = _caption_records([1, 2, 3, 4, 5] * 40, task="VizWiz")
        selected = select_caption_images(records, random.Random(7), target=33)
        assert len(selected) == 33  # 33 whole images, never a split group

    def test_caption_group_exclusion(self):
        records = _caption_records([2] * 50)
        excluded = [_record_identity(records[0])]
        selected = select_caption_images(
            records, random.Random(1), target=10, excluded_identities=excluded
        )
        assert "Flickr30k/train/1000001.jpg" not in selected
        assert len(selected) == 10


class TestBuildTaskSplit:
    def _fake_images(self, tmp_path, payload):
        image_folder = tmp_path / "datasets"
        for record in payload["validation"]["records"] + payload["train"]["records"]:
            target = image_folder / record["image"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")

    def _assert_multiset(self, original, payload):
        original_counts = Counter(content_signature(r) for r in original)
        kept_counts = Counter(content_signature(r) for r in payload["train"]["records"])
        removed_sigs = Counter(
            content_signature(original[index]) for index in payload["removed_indexes"]
        )
        assert kept_counts == original_counts - removed_sigs

    def test_walk_task_payload_and_gates(self, tmp_path):
        source = tmp_path / "train.json"
        original = _question_bank(n=4000, twins=2, dup_qids=2)
        dump(original, source)
        test = tmp_path / "test.json"
        dump(_question_bank(n=10, twins=0, dup_qids=0), test)
        payload = build_task_split(
            task_name="ArxivQA", task_index=1,
            source_train_path=str(source), test_path=str(test), seed=42,
        )
        assert payload["validation"]["count"] == VALIDATION_TARGET
        assert payload["removed_count"] >= VALIDATION_TARGET
        self._fake_images(tmp_path, payload)
        validate_built_task(payload, image_folder=str(tmp_path / "datasets"),
                            caption_task=False)
        self._assert_multiset(original, payload)
        question_ids = [r["question_id"] for r in payload["validation"]["records"]]
        assert len(set(question_ids)) == VALIDATION_TARGET
        assert all(
            r["answer"] == r["conversations"][-1]["value"]
            for r in payload["validation"]["records"]
        )
        assert all("text" not in r and "id" not in r
                   for r in payload["validation"]["records"])
        assert all(r["id"].startswith("v7_t1_train_")
                   for r in payload["train"]["records"])

    def test_whole_exact_identity_twin_groups_leave_train(self, tmp_path):
        # base rows 0..199 each have two exact content twins: selection of
        # one member drags all copies out of formal train.
        source = tmp_path / "train.json"
        original = _question_bank(n=400, twins=200, dup_qids=0)
        for index in range(200):
            row = original[index]
            original.append(conv_record(
                row["image"], "question-{}?".format(index),
                row["conversations"][-1]["value"], native_value=200000 + index,
            ))
        dump(original, source)
        payload = build_task_split(
            task_name="CLEVR", task_index=4,
            source_train_path=str(source), test_path=None, seed=42,
        )
        self._fake_images(tmp_path, payload)
        validate_built_task(payload, image_folder=str(tmp_path / "datasets"),
                            caption_task=False)
        # 256 val members + at least 100 extra twin copies removed
        assert payload["removed_count"] >= VALIDATION_TARGET + 100

    def test_caption_task_payload_refs_and_duplicate_caption(self, tmp_path):
        source = tmp_path / "train.json"
        original = _caption_records([1, 2, 3, 4, 5] * 60, task="Flickr30k")
        dump(original, source)
        payload = build_task_split(
            task_name="Flickr30k", task_index=5,
            source_train_path=str(source), test_path=None, seed=42,
        )
        assert payload["validation"]["count"] == VALIDATION_TARGET
        assert payload["stats"]["num_validation_images"] == VALIDATION_TARGET
        assert payload["stats"]["num_validation_questions"] == VALIDATION_TARGET
        assert payload["stats"]["ref_rich_first"] is True
        coco = payload["outputs"]["validation_coco"]
        assert len(coco["images"]) == VALIDATION_TARGET
        # ids are positional and file_names are basenames in record order
        for position, record in enumerate(payload["validation"]["records"], start=1):
            assert coco["images"][position - 1] == {
                "id": position, "file_name": Path(record["image"]).name,
            }
        refs = Counter(item["image_id"] for item in coco["annotations"])
        assert set(refs) == set(range(1, VALIDATION_TARGET + 1))
        assert all(1 <= refs[position] <= 5 for position in refs)
        assert payload["stats"]["num_validation_reference_captions"] == sum(refs.values())
        self._fake_images(tmp_path, payload)
        validate_built_task(payload, image_folder=str(tmp_path / "datasets"),
                            caption_task=True)
        self._assert_multiset(original, payload)
        # every removed caption row is retained as a COCO reference
        ref_captions = {item["caption"] for item in coco["annotations"]}
        for index in payload["removed_indexes"]:
            assert content_signature(original[index])[2] in ref_captions

    def test_vizwiz_whole_group_build(self, tmp_path):
        source = tmp_path / "train.json"
        original = _caption_records([1, 2, 3, 4, 5] * 60, task="VizWiz")
        dump(original, source)
        payload = build_task_split(
            task_name="VizWiz", task_index=2,
            source_train_path=str(source), test_path=None, seed=42,
        )
        assert payload["stats"]["num_validation_images"] == VALIDATION_TARGET
        assert payload["stats"]["num_validation_questions"] == VALIDATION_TARGET
        assert payload["outputs"]["validation_coco"]["categories"] == [
            {"id": 1, "name": "captioning"}
        ]
        self._fake_images(tmp_path, payload)
        validate_built_task(payload, image_folder=str(tmp_path / "datasets"),
                            caption_task=True)

    def test_write_outputs_and_idempotence(self, tmp_path):
        source = tmp_path / "train.json"
        original = _question_bank(n=1000, twins=0, dup_qids=0)
        dump(original, source)
        payload = build_task_split(
            task_name="CLEVR", task_index=4,
            source_train_path=str(source), test_path=None, seed=42,
        )
        self._fake_images(tmp_path, payload)
        validate_built_task(payload, image_folder=str(tmp_path / "datasets"),
                            caption_task=False)
        val_root = tmp_path / "v7_validation"
        train_root = tmp_path / "v7_train"
        provenance = write_task_outputs(payload, str(val_root), str(train_root))
        assert (val_root / "CLEVR" / "validation.json").is_file()
        assert (train_root / "CLEVR" / "train.json").is_file()
        assert (val_root / "CLEVR" / "provenance_build.json").is_file()
        assert provenance["validation"]["record_count"] == VALIDATION_TARGET
        # rebuilding on the same seed reproduces byte-identical artifacts
        payload2 = build_task_split(
            task_name="CLEVR", task_index=4,
            source_train_path=str(source), test_path=None, seed=42,
        )
        provenance2 = write_task_outputs(payload2, str(val_root), str(train_root))
        assert provenance2["validation"]["file_sha256"] == \
            provenance["validation"]["file_sha256"]
        assert provenance2["train"]["file_sha256"] == provenance["train"]["file_sha256"]

    def test_imagenet_r_end_to_end(self, tmp_path):
        sizes = {str(i).zfill(9): 41 for i in range(2)}
        for i in range(2, 200):
            sizes[str(i).zfill(9)] = 50
        records = []
        for wnid, size in sorted(sizes.items()):
            records.extend(_imagenet_r_record(wnid, index) for index in range(size))
        source = tmp_path / "train.json"
        dump(records, source)
        payload = build_task_split(
            task_name="ImageNet-R", task_index=0,
            source_train_path=str(source), test_path=None, seed=42,
        )
        assert payload["validation"]["count"] == VALIDATION_TARGET
        remaining = Counter(
            record["image"].split("/")[2] for record in payload["train"]["records"]
        )
        assert len(remaining) == 200
        assert min(remaining.values()) >= 40
        self._fake_images(tmp_path, payload)
        validate_built_task(payload, image_folder=str(tmp_path / "datasets"),
                            caption_task=False)
