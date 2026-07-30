import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw


FUNCTIONS = ("A_only", "B_only", "C_only", "A_plus_B", "B_plus_C", "A_plus_C")
SHAPES = ("circle", "square", "triangle")
SPLIT_SIZES = {"train": 1600, "val": 200, "test": 400}
PROMPTS = {
    "A_only": (
        "What shape is shown?",
        "Identify the shape in the image.",
        "Name the visible geometric shape.",
        "Which shape do you see?",
    ),
    "B_only": (
        "How many objects are shown?",
        "Count all the objects in the image.",
        "What is the total number of objects?",
        "How many items do you see?",
    ),
    "C_only": (
        "Is the blue marker left or right of the divider?",
        "Which side of the vertical line contains the blue marker?",
        "State the marker's side relative to the divider: left or right.",
        "Where is the blue marker compared with the vertical divider?",
    ),
    "A_plus_B": (
        "How many {query} shapes are shown?",
        "Count the {query} shapes in the image.",
        "What is the number of {query} shapes?",
        "How many of the objects are {query}s?",
    ),
    "B_plus_C": (
        "How many objects are on the {side} side of the divider?",
        "Count the objects located {side} of the vertical line.",
        "What is the number of objects on the {side}?",
        "How many items appear to the {side} of the divider?",
    ),
    "A_plus_C": (
        "What shape is on the {side} side of the divider?",
        "Identify the shape located {side} of the vertical line.",
        "Which shape appears on the {side}?",
        "Name the geometric shape to the {side} of the divider.",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(draw: ImageDraw.ImageDraw, kind: str, center: Tuple[int, int], size: int, fill):
    x, y = center
    if kind == "circle":
        draw.ellipse((x - size, y - size, x + size, y + size), fill=fill, outline="black", width=2)
    elif kind == "square":
        draw.rectangle((x - size, y - size, x + size, y + size), fill=fill, outline="black", width=2)
    elif kind == "triangle":
        draw.polygon(((x, y - size), (x - size, y + size), (x + size, y + size)), fill=fill, outline="black")
    else:
        raise ValueError("unknown shape {}".format(kind))


def _grid_positions(rng: random.Random, count: int, left=None) -> List[Tuple[int, int]]:
    if left is None:
        candidates = [(x, y) for y in (55, 105, 155, 205) for x in (45, 95, 145, 195, 245)]
    elif left:
        candidates = [(x, y) for y in (55, 105, 155, 205) for x in (35, 75, 115)]
    else:
        candidates = [(x, y) for y in (55, 105, 155, 205) for x in (185, 225, 265)]
    rng.shuffle(candidates)
    return candidates[:count]


def _render(function_name: str, rng: random.Random, output: Path):
    image = Image.new("RGB", (300, 260), "white")
    draw = ImageDraw.Draw(image)
    metadata = {}
    if function_name == "A_only":
        kind = rng.choice(SHAPES)
        _shape(draw, kind, (150, 130), rng.randint(38, 55), "#b8b8b8")
        answer = kind
    elif function_name == "B_only":
        count = rng.randint(1, 6)
        for center in _grid_positions(rng, count):
            _shape(draw, "circle", center, 15, "#b8b8b8")
        answer = str(count)
    elif function_name == "C_only":
        draw.line((150, 15, 150, 245), fill="black", width=4)
        side = rng.choice(("left", "right"))
        center = (rng.randint(45, 115), rng.randint(55, 205)) if side == "left" else (rng.randint(185, 255), rng.randint(55, 205))
        _shape(draw, "circle", center, 22, "#3d78d8")
        answer = side
    elif function_name == "A_plus_B":
        query = rng.choice(SHAPES)
        query_count = rng.randint(1, 4)
        total = rng.randint(max(query_count + 1, 3), 7)
        kinds = [query] * query_count
        kinds.extend(rng.choice([kind for kind in SHAPES if kind != query]) for _ in range(total - query_count))
        rng.shuffle(kinds)
        for center, kind in zip(_grid_positions(rng, total), kinds):
            _shape(draw, kind, center, 16, "#b8b8b8")
        metadata["query"] = query
        answer = str(query_count)
    elif function_name == "B_plus_C":
        draw.line((150, 15, 150, 245), fill="black", width=4)
        left_count = rng.randint(1, 5)
        right_count = rng.randint(1, 5)
        for center in _grid_positions(rng, left_count, left=True):
            _shape(draw, "circle", center, 13, "#b8b8b8")
        for center in _grid_positions(rng, right_count, left=False):
            _shape(draw, "circle", center, 13, "#b8b8b8")
        side = rng.choice(("left", "right"))
        metadata["side"] = side
        answer = str(left_count if side == "left" else right_count)
    elif function_name == "A_plus_C":
        draw.line((150, 15, 150, 245), fill="black", width=4)
        left_shape, right_shape = rng.sample(SHAPES, 2)
        _shape(draw, left_shape, (80, 130), 32, "#b8b8b8")
        _shape(draw, right_shape, (220, 130), 32, "#b8b8b8")
        side = rng.choice(("left", "right"))
        metadata["side"] = side
        answer = left_shape if side == "left" else right_shape
    else:
        raise ValueError("unknown function {}".format(function_name))
    image.save(output, format="PNG", optimize=False)
    return answer, metadata


def _question(function_name: str, index: int, metadata: dict) -> str:
    template = PROMPTS[function_name][index % len(PROMPTS[function_name])]
    return template.format(**metadata) + "\nAnswer the question using a single word or phrase."


def generate(root: str, seed: int = 730, split_sizes: Dict[str, int] = None) -> dict:
    root_path = Path(root).resolve()
    split_sizes = dict(split_sizes or SPLIT_SIZES)
    root_path.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "functions": {
            "A": "shape recognition",
            "B": "counting",
            "C": "left/right spatial relation",
        },
        "split_sizes": split_sizes,
        "records": {},
        "template_id_policy": "split/function/index; IDs are disjoint across splits",
        "files": {},
    }
    all_template_ids = {}
    for function_offset, function_name in enumerate(FUNCTIONS):
        manifest["records"][function_name] = {}
        for split_offset, (split, count) in enumerate(split_sizes.items()):
            image_dir = root_path / "images" / function_name / split
            instruction_dir = root_path / "instructions" / function_name
            image_dir.mkdir(parents=True)
            instruction_dir.mkdir(parents=True, exist_ok=True)
            rng = random.Random(seed + function_offset * 100_000 + split_offset * 10_000)
            records = []
            annotations = []
            template_ids = set()
            answer_counts = Counter()
            for index in range(count):
                template_id = "{}:{}:{:06d}".format(split, function_name, index)
                template_ids.add(template_id)
                image_name = "{:06d}.png".format(index)
                image_path = image_dir / image_name
                answer, metadata = _render(function_name, rng, image_path)
                question = _question(function_name, index, metadata)
                sample_id = "controlled/{}/{}/{}".format(function_name, split, index)
                relative_image = str(image_path.relative_to(root_path.parent)).replace(os.sep, "/")
                records.append({
                    "id": sample_id,
                    "question_id": sample_id,
                    "task_id": "Controlled/{}".format(function_name),
                    "template_id": template_id,
                    "image": relative_image,
                    "conversations": [
                        {"from": "human", "value": "<image>\n" + question},
                        {"from": "gpt", "value": answer},
                    ],
                })
                annotations.append({
                    "question_id": sample_id,
                    "task_id": "Controlled/{}".format(function_name),
                    "template_id": template_id,
                    "image": relative_image,
                    "text": question,
                    "answer": answer,
                })
                answer_counts[answer] += 1
            train_path = instruction_dir / (split + ".json")
            eval_path = instruction_dir / (split + "_eval.json")
            train_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            eval_path.write_text(json.dumps(annotations, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            manifest["records"][function_name][split] = {
                "count": count,
                "answer_counts": dict(sorted(answer_counts.items())),
                "instruction_file": str(train_path),
                "evaluation_file": str(eval_path),
            }
            all_template_ids[(function_name, split)] = template_ids
            manifest["files"][str(train_path.relative_to(root_path))] = _sha256(train_path)
            manifest["files"][str(eval_path.relative_to(root_path))] = _sha256(eval_path)
    for function_name in FUNCTIONS:
        splits = [all_template_ids[(function_name, split)] for split in split_sizes]
        for left in range(len(splits)):
            for right in range(left + 1, len(splits)):
                if splits[left] & splits[right]:
                    raise AssertionError("template leakage in {}".format(function_name))
    manifest_path = root_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=730)
    parser.add_argument("--train-size", type=int, default=SPLIT_SIZES["train"])
    parser.add_argument("--val-size", type=int, default=SPLIT_SIZES["val"])
    parser.add_argument("--test-size", type=int, default=SPLIT_SIZES["test"])
    args = parser.parse_args()
    result = generate(args.output_root, args.seed, {
        "train": args.train_size, "val": args.val_size, "test": args.test_size,
    })
    print(json.dumps({"seed": result["seed"], "records": result["records"]}, sort_keys=True))


if __name__ == "__main__":
    main()
