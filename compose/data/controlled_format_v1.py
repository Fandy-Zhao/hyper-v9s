"""
Format-controlled dataset generator v1.

Produces a shared scene pool and five task types (A_only, B_only, C_only,
A_plus_B, B_plus_C) with a unified A/B yes/no answer interface.

Key differences from controlled_functional.py:
- Shared scene pool: all tasks derive from the same scenes
- Scene metadata saved: enables QA regeneration without re-rendering
- Unified answer space: A=Yes, B=No across all tasks
- Scene-level split: train/val/test partition on scene_id, not sample index
- Positive/negative balance: ~50/50 per task per split
- Hard negatives: count_negative, attribute_negative, relation_negative
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANVAS_WIDTH = 300
CANVAS_HEIGHT = 260
MIN_HORIZONTAL_GAP = 30  # minimum x-distance for unambiguous left/right

SHAPES = ("circle", "square", "triangle")
COLORS = {
    "red": "#e74c3c",
    "blue": "#3498db",
    "green": "#2ecc71",
    "yellow": "#f1c40f",
}
# Assign colours only when the scene needs distinct-object identification;
# for count-only images we may use a single neutral grey.
NEUTRAL_GREY = "#b8b8b8"

FUNCTIONS = ("A_only", "B_only", "C_only", "A_plus_B", "B_plus_C")

# Per-function question templates (all yield Yes/No answers).
# {shape}, {color}, {ref_shape}, {ref_color}, {count}, {ref_x}, etc.
PROMPTS: Dict[str, Tuple[str, ...]] = {
    "A_only": (
        "Is the {color} object a {shape}?",
        "Is the {color} shape a {shape}?",
        "Is the {color} object {shape}-shaped?",
        "Is the {color} one a {shape}?",
    ),
    "B_only": (
        "Are there exactly {count} objects in the image?",
        "Does the image contain exactly {count} objects?",
        "Is the total number of objects equal to {count}?",
        "Are there precisely {count} objects shown?",
    ),
    "C_only": (
        "Is the {color} {shape} to the left of the {ref_color} {ref_shape}?",
        "Is the {color} {shape} on the left side of the {ref_color} {ref_shape}?",
        "Is the {color} {shape} positioned to the left of the {ref_color} {ref_shape}?",
        "Is the {color} {shape} located left of the {ref_color} {ref_shape}?",
    ),
    "A_plus_B": (
        "Are there exactly {count} {shape_plural} in the image?",
        "Does the image contain exactly {count} {shape_plural}?",
        "Is the number of {shape_plural} equal to {count}?",
        "Are there precisely {count} {shape_plural}?",
    ),
    "B_plus_C": (
        "Are there exactly {count} objects to the left of the {ref_color} {ref_shape}?",
        "Does the image have exactly {count} objects left of the {ref_color} {ref_shape}?",
        "Is the number of objects to the left of the {ref_color} {ref_shape} equal to {count}?",
        "Are there precisely {count} objects on the left of the {ref_color} {ref_shape}?",
    ),
}

# Answer mapping
ANSWER_LETTER = {"yes": "A", "no": "B"}
ANSWER_TEXT = {"A": "Yes", "B": "No"}

# ---------------------------------------------------------------------------
# Grid / layout helpers
# ---------------------------------------------------------------------------


def _grid_positions(
    rng: random.Random,
    count: int,
    left_only: bool = False,
    right_only: bool = False,
) -> List[Tuple[int, int]]:
    """Return `count` non-overlapping grid positions."""
    # 6 columns × 4 rows
    cols = [40, 85, 130, 175, 220, 265]
    rows = [55, 105, 155, 205]

    if left_only:
        cols = [c for c in cols if c < CANVAS_WIDTH // 2 - 20]
    elif right_only:
        cols = [c for c in cols if c > CANVAS_WIDTH // 2 + 20]

    candidates = [(x, y) for y in rows for x in cols]
    rng.shuffle(candidates)
    return candidates[:count]


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    kind: str,
    center: Tuple[int, int],
    size: int,
    fill: str,
) -> None:
    """Draw a geometric shape on the image."""
    x, y = center
    if kind == "circle":
        draw.ellipse(
            (x - size, y - size, x + size, y + size),
            fill=fill,
            outline="black",
            width=2,
        )
    elif kind == "square":
        draw.rectangle(
            (x - size, y - size, x + size, y + size),
            fill=fill,
            outline="black",
            width=2,
        )
    elif kind == "triangle":
        draw.polygon(
            ((x, y - size), (x - size, y + size), (x + size, y + size)),
            fill=fill,
            outline="black",
        )
    else:
        raise ValueError(f"unknown shape {kind}")


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------


def _generate_scene(
    scene_id: str,
    rng: random.Random,
    images_dir: Path,
    scenes_dir: Path,
) -> Dict[str, Any]:
    """Generate one scene: image + metadata dict.  Returns the metadata."""
    image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    # Number of objects: 3-8 (uniform)
    num_objects = rng.randint(3, 8)

    # Assign shapes and colors
    shape_counts: Dict[str, int] = defaultdict(int)
    objects: List[Dict[str, Any]] = []

    # Ensure at least 2 different shapes present (needed for A_plus_B negatives)
    # and at least 2 different colors (needed for C tasks)
    available_shapes = list(SHAPES)
    rng.shuffle(available_shapes)
    available_colors = list(COLORS.keys())
    rng.shuffle(available_colors)

    # Place objects on grid
    positions = _grid_positions(rng, num_objects)

    for i, (px, py) in enumerate(positions):
        shape = available_shapes[i % len(available_shapes)] if i < 3 else rng.choice(SHAPES)
        color = available_colors[i % len(available_colors)] if i < 2 else rng.choice(list(COLORS.keys()))
        size = rng.randint(14, 22)
        fill = COLORS[color]

        _draw_shape(draw, shape, (px, py), size, fill)
        shape_counts[shape] += 1
        objects.append({
            "id": f"{scene_id}_obj{i:02d}",
            "shape": shape,
            "color": color,
            "x": px,
            "y": py,
            "size": size,
        })

    # Save image
    image_path = images_dir / f"{scene_id}.png"
    image.save(image_path, format="PNG", optimize=False)

    # Build and save scene metadata
    total_count = num_objects
    metadata = {
        "scene_id": scene_id,
        "image": f"images/{scene_id}.png",
        "total_objects": total_count,
        "shape_counts": dict(shape_counts),
        "objects": objects,
        "min_horizontal_gap": MIN_HORIZONTAL_GAP,
        "canvas_size": [CANVAS_WIDTH, CANVAS_HEIGHT],
    }

    scene_path = scenes_dir / f"{scene_id}.json"
    scene_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return metadata


# ---------------------------------------------------------------------------
# Question / answer generation helpers
# ---------------------------------------------------------------------------


def _pluralize(shape: str, count: int) -> str:
    """Return singular or plural form."""
    if count == 1:
        return shape
    return shape + "s"


def _find_object_pairs_with_clear_lr(
    objects: List[Dict[str, Any]],
    rng: random.Random,
) -> List[Tuple[int, int]]:
    """Return pairs of object indices (target, reference) with clear left/right."""
    pairs = []
    for i, obj_i in enumerate(objects):
        for j, obj_j in enumerate(objects):
            if i == j:
                continue
            if abs(obj_i["x"] - obj_j["x"]) >= MIN_HORIZONTAL_GAP:
                pairs.append((i, j))
    return pairs


def _make_a_only_samples(
    scene: Dict[str, Any],
    rng: random.Random,
    template_idx: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate one positive and one negative A_only sample."""
    objects = scene["objects"]
    # Pick a target object
    target = rng.choice(objects)
    true_shape = target["shape"]
    # Pick a false shape different from true shape
    false_shape = rng.choice([s for s in SHAPES if s != true_shape])

    template = PROMPTS["A_only"][template_idx % len(PROMPTS["A_only"])]

    # Positive: ask about true shape → answer Yes (A)
    pos_question = template.format(color=target["color"], shape=true_shape)
    pos = {
        "task": "A_only",
        "required_functions": ["A"],
        "question": pos_question,
        "options": {"A": "Yes", "B": "No"},
        "answer": "A",
        "polarity": "positive",
        "negative_type": None,
        "metadata": {
            "queried_shape": true_shape,
            "queried_count": None,
            "queried_relation": None,
            "true_count": None,
            "target_object_id": target["id"],
            "reference_object_id": None,
            "target_true_shape": true_shape,
        },
    }

    # Negative: ask about wrong shape → answer No (B)
    neg_question = template.format(color=target["color"], shape=false_shape)
    neg = {
        "task": "A_only",
        "required_functions": ["A"],
        "question": neg_question,
        "options": {"A": "Yes", "B": "No"},
        "answer": "B",
        "polarity": "negative",
        "negative_type": "attribute_negative",
        "metadata": {
            "queried_shape": false_shape,
            "queried_count": None,
            "queried_relation": None,
            "true_count": None,
            "target_object_id": target["id"],
            "reference_object_id": None,
            "target_true_shape": true_shape,
        },
    }

    return pos, neg


def _make_b_only_samples(
    scene: Dict[str, Any],
    rng: random.Random,
    template_idx: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate one positive and one negative B_only sample."""
    total = scene["total_objects"]
    template = PROMPTS["B_only"][template_idx % len(PROMPTS["B_only"])]

    # Positive: exact count → Yes (A)
    pos_question = template.format(count=total)
    pos = {
        "task": "B_only",
        "required_functions": ["B"],
        "question": pos_question,
        "options": {"A": "Yes", "B": "No"},
        "answer": "A",
        "polarity": "positive",
        "negative_type": None,
        "metadata": {
            "queried_shape": None,
            "queried_count": total,
            "queried_relation": None,
            "true_count": total,
            "target_object_id": None,
            "reference_object_id": None,
        },
    }

    # Negative: n-1 or n+1 (choose one that is at least 1 and reasonable)
    if total <= 3:
        false_count = total + 1
    elif total >= 8:
        false_count = total - 1
    else:
        false_count = total + rng.choice([-1, 1])

    neg_question = template.format(count=false_count)
    neg = {
        "task": "B_only",
        "required_functions": ["B"],
        "question": neg_question,
        "options": {"A": "Yes", "B": "No"},
        "answer": "B",
        "polarity": "negative",
        "negative_type": "count_negative",
        "metadata": {
            "queried_shape": None,
            "queried_count": false_count,
            "queried_relation": None,
            "true_count": total,
            "target_object_id": None,
            "reference_object_id": None,
        },
    }

    return pos, neg


def _make_c_only_samples(
    scene: Dict[str, Any],
    rng: random.Random,
    template_idx: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate one positive and one negative C_only sample."""
    objects = scene["objects"]
    pairs = _find_object_pairs_with_clear_lr(objects, rng)
    if not pairs:
        # Fallback: no clear left/right pair; skip
        return None, None

    # Pick a pair and ensure target is LEFT of ref for positive samples.
    # All C_only templates ask "Is X to the LEFT of Y?"
    left_idx, right_idx = rng.choice(pairs)
    if objects[left_idx]["x"] > objects[right_idx]["x"]:
        left_idx, right_idx = right_idx, left_idx  # swap to ensure left < right

    target = objects[left_idx]  # always the left object
    ref = objects[right_idx]    # always the right object

    template = PROMPTS["C_only"][template_idx % len(PROMPTS["C_only"])]

    # Positive: "Is left_obj to the LEFT of right_obj?" → Yes (A)
    pos_question = template.format(
        color=target["color"],
        shape=target["shape"],
        ref_color=ref["color"],
        ref_shape=ref["shape"],
    )
    pos = {
        "task": "C_only",
        "required_functions": ["C"],
        "question": pos_question,
        "options": {"A": "Yes", "B": "No"},
        "answer": "A",
        "polarity": "positive",
        "negative_type": None,
        "metadata": {
            "queried_shape": target["shape"],
            "queried_count": None,
            "queried_relation": "left",
            "true_count": None,
            "target_object_id": target["id"],
            "reference_object_id": ref["id"],
            "target_x": target["x"],
            "reference_x": ref["x"],
            "true_relation": "left",
        },
    }

    # Negative: "Is left_obj to the RIGHT of right_obj?" → No (B)
    neg_question = template.format(
        color=target["color"],
        shape=target["shape"],
        ref_color=ref["color"],
        ref_shape=ref["shape"],
    )
    # Replace "left" wording with "right"
    neg_question = (neg_question
        .replace(" to the left of ", " to the right of ")
        .replace(" on the left side of ", " on the right side of ")
        .replace(" left of ", " right of ")
        .replace(" positioned to the left of ", " positioned to the right of ")
        .replace(" located left of ", " located right of "))
    neg = {
        "task": "C_only",
        "required_functions": ["C"],
        "question": neg_question,
        "options": {"A": "Yes", "B": "No"},
        "answer": "B",
        "polarity": "negative",
        "negative_type": "relation_negative",
        "metadata": {
            "queried_shape": target["shape"],
            "queried_count": None,
            "queried_relation": "right",
            "true_count": None,
            "target_object_id": target["id"],
            "reference_object_id": ref["id"],
            "target_x": target["x"],
            "reference_x": ref["x"],
            "true_relation": "left",
        },
    }

    return pos, neg


def _make_a_plus_b_samples(
    scene: Dict[str, Any],
    rng: random.Random,
    template_idx: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Generate one positive and two types of negatives for A_plus_B."""
    shape_counts = scene["shape_counts"]
    total = scene["total_objects"]
    template = PROMPTS["A_plus_B"][template_idx % len(PROMPTS["A_plus_B"])]

    # Pick a query shape that exists in the scene
    present_shapes = [s for s in SHAPES if shape_counts.get(s, 0) > 0]
    if not present_shapes:
        # Edge case: should not happen but just in case
        query_shape = rng.choice(SHAPES)
        true_count = 0
    else:
        query_shape = rng.choice(present_shapes)
        true_count = shape_counts[query_shape]

    # Positive: correct shape + correct count → Yes (A)
    pos_question = template.format(
        shape=query_shape, shape_plural=_pluralize(query_shape, true_count), count=true_count)
    pos = {
        "task": "A_plus_B",
        "required_functions": ["A", "B"],
        "question": pos_question,
        "options": {"A": "Yes", "B": "No"},
        "answer": "A",
        "polarity": "positive",
        "negative_type": None,
        "metadata": {
            "queried_shape": query_shape,
            "queried_count": true_count,
            "queried_relation": None,
            "true_count": true_count,
            "target_object_id": None,
            "reference_object_id": None,
        },
    }

    negatives = []

    # Negative type 1: count_negative — correct shape, wrong count
    if true_count <= 1:
        false_count = true_count + 1
    elif true_count >= total - 1:
        false_count = true_count - 1
    else:
        false_count = true_count + rng.choice([-1, 1])

    neg1_question = template.format(
        shape=query_shape, shape_plural=_pluralize(query_shape, false_count), count=false_count)
    negatives.append({
        "task": "A_plus_B",
        "required_functions": ["A", "B"],
        "question": neg1_question,
        "options": {"A": "Yes", "B": "No"},
        "answer": "B",
        "polarity": "negative",
        "negative_type": "count_negative",
        "metadata": {
            "queried_shape": query_shape,
            "queried_count": false_count,
            "queried_relation": None,
            "true_count": true_count,
            "target_object_id": None,
            "reference_object_id": None,
        },
    })

    # Negative type 2: attribute_negative — use same count as positive (which
    # is a plausible count for SOME shape), but query a DIFFERENT shape.
    # Answer is B because the queried shape's actual count differs.
    wrong_shapes = [s for s in SHAPES if s != query_shape]
    wrong_shape = rng.choice(wrong_shapes)
    wrong_count = shape_counts.get(wrong_shape, 0)

    # Use the positive's true_count (deceptive value), but ensure it differs
    # from the wrong shape's actual count
    deceptive_count = true_count
    if deceptive_count == wrong_count:
        # Fallback: shift by 1 so answer stays No
        deceptive_count = wrong_count + 1 if wrong_count < total else wrong_count - 1
        deceptive_count = max(1, deceptive_count)

    neg2_question = template.format(
        shape=wrong_shape, shape_plural=_pluralize(wrong_shape, deceptive_count), count=deceptive_count)
    negatives.append({
        "task": "A_plus_B",
        "required_functions": ["A", "B"],
        "question": neg2_question,
        "options": {"A": "Yes", "B": "No"},
        "answer": "B",
        "polarity": "negative",
        "negative_type": "attribute_negative",
        "metadata": {
            "queried_shape": wrong_shape,
            "queried_count": deceptive_count,
            "queried_relation": None,
            "true_count": wrong_count,
            "target_object_id": None,
            "reference_object_id": None,
        },
    })

    return pos, negatives


def _make_b_plus_c_samples(
    scene: Dict[str, Any],
    rng: random.Random,
    template_idx: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Generate one positive and two types of negatives for B_plus_C."""
    objects = scene["objects"]
    if len(objects) < 2:
        return None, []

    template = PROMPTS["B_plus_C"][template_idx % len(PROMPTS["B_plus_C"])]

    # Pick a reference object
    ref_obj = rng.choice(objects)
    ref_x = ref_obj["x"]

    # Count objects to the left of the reference
    left_objects = [obj for obj in objects if obj["x"] < ref_x - 5]  # 5px margin
    left_count = len(left_objects)

    # Positive: correct count to the left → Yes (A)
    pos_question = template.format(
        count=left_count,
        ref_color=ref_obj["color"],
        ref_shape=ref_obj["shape"],
    )
    pos = {
        "task": "B_plus_C",
        "required_functions": ["B", "C"],
        "question": pos_question,
        "options": {"A": "Yes", "B": "No"},
        "answer": "A",
        "polarity": "positive",
        "negative_type": None,
        "metadata": {
            "queried_shape": None,
            "queried_count": left_count,
            "queried_relation": "left",
            "true_count": left_count,
            "target_object_id": None,
            "reference_object_id": ref_obj["id"],
            "reference_x": ref_x,
        },
    }

    negatives = []

    # Negative type 1: count_negative — same reference, wrong count
    if left_count <= 0:
        false_count = left_count + 1
    elif left_count >= len(objects) - 1:
        false_count = left_count - 1
    else:
        false_count = left_count + rng.choice([-1, 1])
    false_count = max(0, false_count)

    neg1_question = template.format(
        count=false_count,
        ref_color=ref_obj["color"],
        ref_shape=ref_obj["shape"],
    )
    negatives.append({
        "task": "B_plus_C",
        "required_functions": ["B", "C"],
        "question": neg1_question,
        "options": {"A": "Yes", "B": "No"},
        "answer": "B",
        "polarity": "negative",
        "negative_type": "count_negative",
        "metadata": {
            "queried_shape": None,
            "queried_count": false_count,
            "queried_relation": "left",
            "true_count": left_count,
            "target_object_id": None,
            "reference_object_id": ref_obj["id"],
            "reference_x": ref_x,
        },
    })

    # Negative type 2: relation_negative — use positive's count (deceptive)
    # but query the OPPOSITE relation (right instead of left).
    # Ensure the count does NOT match the opposite side's actual count.
    right_objects = [obj for obj in objects if obj["x"] > ref_x + 5]
    right_count = len(right_objects)

    deceptive_count = left_count
    if deceptive_count == right_count:
        # Adjust so the answer stays No
        deceptive_count = right_count + 1 if right_count < len(objects) else right_count - 1
        deceptive_count = max(0, deceptive_count)

    # Use right-of template
    right_templates = (
        "Are there exactly {count} objects to the right of the {ref_color} {ref_shape}?",
        "Does the image have exactly {count} objects right of the {ref_color} {ref_shape}?",
        "Is the number of objects to the right of the {ref_color} {ref_shape} equal to {count}?",
        "Are there precisely {count} objects on the right of the {ref_color} {ref_shape}?",
    )
    right_template = right_templates[template_idx % len(right_templates)]

    neg2_question = right_template.format(
        count=deceptive_count,
        ref_color=ref_obj["color"],
        ref_shape=ref_obj["shape"],
    )
    negatives.append({
        "task": "B_plus_C",
        "required_functions": ["B", "C"],
        "question": neg2_question,
        "options": {"A": "Yes", "B": "No"},
        "answer": "B",
        "polarity": "negative",
        "negative_type": "relation_negative",
        "metadata": {
            "queried_shape": None,
            "queried_count": deceptive_count,
            "queried_relation": "right",
            "true_count": right_count,
            "target_object_id": None,
            "reference_object_id": ref_obj["id"],
            "reference_x": ref_x,
        },
    })

    return pos, negatives


# ---------------------------------------------------------------------------
# Sample assembly
# ---------------------------------------------------------------------------


def _assemble_sample(
    split: str,
    task: str,
    index: int,
    scene_id: str,
    scene_image: str,
    core: Dict[str, Any],
) -> Dict[str, Any]:
    """Wrap core QA dict into full sample record."""
    sample_id = f"{task}/{split}/{index:06d}"
    return {
        "id": sample_id,
        "scene_id": scene_id,
        "image": scene_image,
        "task": core["task"],
        "required_functions": core["required_functions"],
        "question": core["question"],
        "options": core["options"],
        "answer": core["answer"],
        "polarity": core["polarity"],
        "negative_type": core["negative_type"],
        "metadata": core["metadata"],
    }


def _assemble_eval_sample(core: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap into flat eval record."""
    return {
        "text": core["question"],
        "answer": core["answer"],
        "task": core["task"],
        "required_functions": core["required_functions"],
        "polarity": core["polarity"],
        "negative_type": core["negative_type"],
    }


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------


def generate(
    output_root: str,
    seed: int = 730,
    split_sizes: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Generate the full format-controlled dataset."""
    if split_sizes is None:
        split_sizes = {"train": 1600, "val": 200, "test": 400}

    root_path = Path(output_root).resolve()
    if root_path.exists():
        raise FileExistsError(f"Output root already exists: {root_path}")

    root_path.mkdir(parents=True)

    images_dir = root_path / "images"
    scenes_dir = root_path / "scenes"
    images_dir.mkdir()
    scenes_dir.mkdir()

    # Step 1: Generate shared scene pool
    total_scenes = sum(split_sizes.values())
    master_rng = random.Random(seed)

    print(f"Generating {total_scenes} shared scenes (seed={seed})...")
    print(f"  Split sizes: {split_sizes}")

    all_scenes: Dict[str, Dict[str, Any]] = {}
    scene_splits: Dict[str, str] = {}  # scene_id -> split

    scene_index = 0
    for split_name, split_count in split_sizes.items():
        for i in range(split_count):
            scene_id = f"scene_{scene_index:06d}"
            # Use a per-scene RNG derived from master seed + scene index
            scene_rng = random.Random(seed + scene_index * 1000)
            scene_meta = _generate_scene(scene_id, scene_rng, images_dir, scenes_dir)
            all_scenes[scene_id] = scene_meta
            scene_splits[scene_id] = split_name
            scene_index += 1

        print(f"  {split_name}: {split_count} scenes generated (scene_{scene_index - split_count:06d} - scene_{scene_index - 1:06d})")

    # Step 2: Generate questions from each scene (1 sample per task per scene).
    # Design: each (task, split) generates exactly split_size samples.
    # Even-index scenes contribute POSITIVE, odd-index scenes contribute NEGATIVE.
    # For A_plus_B / B_plus_C negatives cycle through available negative types.
    print("Generating QA pairs from scenes...")

    # Accumulators per task per split
    records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        task: {split: [] for split in split_sizes} for task in FUNCTIONS
    }

    # Global index per task-per-split (one sample per scene per task)
    task_idx: Dict[str, Dict[str, int]] = {
        task: {split: 0 for split in split_sizes} for task in FUNCTIONS
    }

    stats: Dict[str, Any] = {
        "skipped_C_only": 0,
        "skipped_B_plus_C": 0,
    }

    # Pre-compute scene ordering per split for deterministic iteration
    split_scenes: Dict[str, List[str]] = {
        split: [sid for sid, s in scene_splits.items() if s == split]
        for split in split_sizes
    }
    for split in split_sizes:
        split_scenes[split].sort()  # Deterministic order

    # For each split, for each scene in order, generate one sample per task
    for split in split_sizes:
        scenes_in_split = split_scenes[split]
        target = split_sizes[split]
        # We need exactly `target` scenes; if we generated more, truncate
        scene_list = scenes_in_split[:target]

        for scene_id in scene_list:
            scene_meta = all_scenes[scene_id]
            scene_image = scene_meta["image"]
            scene_seed = seed + int(scene_id.split("_")[1]) * 1000
            scene_rng = random.Random(scene_seed + 500)

            # --- A_only (1 sample per scene) ---
            idx = task_idx["A_only"][split]
            template_idx_a = idx
            pos_a, neg_a = _make_a_only_samples(scene_meta, scene_rng, template_idx_a)
            if idx % 2 == 0:
                records["A_only"][split].append(
                    _assemble_sample(split, "A_only", idx, scene_id, scene_image, pos_a))
            else:
                records["A_only"][split].append(
                    _assemble_sample(split, "A_only", idx, scene_id, scene_image, neg_a))
            task_idx["A_only"][split] += 1

            # --- B_only (1 sample per scene) ---
            idx = task_idx["B_only"][split]
            template_idx_b = idx
            pos_b, neg_b = _make_b_only_samples(scene_meta, scene_rng, template_idx_b)
            if idx % 2 == 0:
                records["B_only"][split].append(
                    _assemble_sample(split, "B_only", idx, scene_id, scene_image, pos_b))
            else:
                records["B_only"][split].append(
                    _assemble_sample(split, "B_only", idx, scene_id, scene_image, neg_b))
            task_idx["B_only"][split] += 1

            # --- C_only (1 sample per scene) ---
            idx = task_idx["C_only"][split]
            pos_c, neg_c = _make_c_only_samples(scene_meta, scene_rng, idx)
            if pos_c is None:
                stats["skipped_C_only"] += 1
                # Fallback: use the two objects closest to each other for a
                # relational question, even if the gap is below threshold.
                objects = scene_meta["objects"]
                # Pick any two objects with different x positions
                obj_pairs = [(i, j) for i in range(len(objects)) for j in range(len(objects))
                             if i != j and objects[i]["x"] != objects[j]["x"]]
                if obj_pairs:
                    ai, bi = obj_pairs[0]
                    a, b = objects[ai], objects[bi]
                    if a["x"] < b["x"]:
                        target_obj, ref_obj = a, b
                    else:
                        target_obj, ref_obj = b, a
                    from copy import deepcopy
                    pos_c = {
                        "task": "C_only", "required_functions": ["C"],
                        "question": f"Is the {target_obj['color']} {target_obj['shape']} to the left of the {ref_obj['color']} {ref_obj['shape']}?",
                        "options": {"A": "Yes", "B": "No"}, "answer": "A",
                        "polarity": "positive", "negative_type": None,
                        "metadata": {"queried_shape": target_obj["shape"], "queried_count": None,
                                     "queried_relation": "left", "true_count": None,
                                     "target_object_id": target_obj["id"],
                                     "reference_object_id": ref_obj["id"],
                                     "target_x": target_obj["x"], "reference_x": ref_obj["x"],
                                     "true_relation": "left"},
                    }
                    neg_c = {
                        "task": "C_only", "required_functions": ["C"],
                        "question": f"Is the {target_obj['color']} {target_obj['shape']} to the right of the {ref_obj['color']} {ref_obj['shape']}?",
                        "options": {"A": "Yes", "B": "No"}, "answer": "B",
                        "polarity": "negative", "negative_type": "relation_negative",
                        "metadata": {"queried_shape": target_obj["shape"], "queried_count": None,
                                     "queried_relation": "right", "true_count": None,
                                     "target_object_id": target_obj["id"],
                                     "reference_object_id": ref_obj["id"],
                                     "target_x": target_obj["x"], "reference_x": ref_obj["x"],
                                     "true_relation": "left"},
                    }
                else:
                    # Last resort: all objects at same x (shouldn't happen)
                    pos_c, neg_c = _make_a_only_samples(scene_meta, scene_rng, idx)
                    pos_c["task"] = "C_only"
                    pos_c["required_functions"] = ["C"]
                    neg_c["task"] = "C_only"
                    neg_c["required_functions"] = ["C"]
            if idx % 2 == 0:
                records["C_only"][split].append(
                    _assemble_sample(split, "C_only", idx, scene_id, scene_image, pos_c))
            else:
                records["C_only"][split].append(
                    _assemble_sample(split, "C_only", idx, scene_id, scene_image, neg_c))
            task_idx["C_only"][split] += 1

            # --- A_plus_B (1 sample per scene) ---
            idx = task_idx["A_plus_B"][split]
            pos_ab, negs_ab = _make_a_plus_b_samples(scene_meta, scene_rng, idx)
            if idx % 2 == 0:
                records["A_plus_B"][split].append(
                    _assemble_sample(split, "A_plus_B", idx, scene_id, scene_image, pos_ab))
            else:
                # Cycle negative types for odd indices
                neg_cycle = (idx // 2) % 2  # 0 → count_neg, 1 → attribute_neg
                records["A_plus_B"][split].append(
                    _assemble_sample(split, "A_plus_B", idx, scene_id, scene_image, negs_ab[neg_cycle]))
            task_idx["A_plus_B"][split] += 1

            # --- B_plus_C (1 sample per scene) ---
            idx = task_idx["B_plus_C"][split]
            pos_bc, negs_bc = _make_b_plus_c_samples(scene_meta, scene_rng, idx)
            if pos_bc is None:
                stats["skipped_B_plus_C"] += 1
                # Fallback
                pos_bc = {
                    "task": "B_plus_C", "required_functions": ["B", "C"],
                    "question": "Are there exactly 0 objects to the left of the green circle?",
                    "options": {"A": "Yes", "B": "No"}, "answer": "B",
                    "polarity": "negative", "negative_type": "count_negative",
                    "metadata": {"queried_shape": None, "queried_count": 0,
                                 "queried_relation": "left", "true_count": 99,
                                 "target_object_id": None, "reference_object_id": None,
                                 "reference_x": -1},
                }
                records["B_plus_C"][split].append(
                    _assemble_sample(split, "B_plus_C", idx, scene_id, scene_image, pos_bc))
            elif idx % 2 == 0:
                records["B_plus_C"][split].append(
                    _assemble_sample(split, "B_plus_C", idx, scene_id, scene_image, pos_bc))
            else:
                neg_cycle = (idx // 2) % 2  # 0 → count_neg, 1 → relation_neg
                records["B_plus_C"][split].append(
                    _assemble_sample(split, "B_plus_C", idx, scene_id, scene_image, negs_bc[neg_cycle]))
            task_idx["B_plus_C"][split] += 1

    # Step 3: Deduplicate (question + image) pairs within each task+split.
    # Repeated question text across different images is acceptable.
    print("Deduplicating...")
    dedup_stats = {}
    for task in FUNCTIONS:
        dedup_stats[task] = {}
        for split in split_sizes:
            seen_pairs: set = set()
            deduped = []
            duplicates_found = 0
            for rec in records[task][split]:
                pair = (rec["question"], rec["image"])
                if pair in seen_pairs:
                    duplicates_found += 1
                    # Regenerate by cycling to next template variant
                    # This should be very rare with shared scenes
                    rec = dict(rec)
                    attempt = 0
                    base_q = rec["question"]
                    while (rec["question"], rec["image"]) in seen_pairs and attempt < 20:
                        if attempt == 0:
                            rec["question"] = base_q.replace("?", " in this picture?")
                        elif attempt == 1:
                            rec["question"] = base_q.replace("?", " shown here?")
                        elif attempt < 10:
                            rec["question"] = base_q + f" [v{attempt}]"
                        else:
                            # Last resort: add small random suffix
                            rec["question"] = base_q + f" [idx{attempt}]"
                        attempt += 1
                seen_pairs.add((rec["question"], rec["image"]))
                deduped.append(rec)
            records[task][split] = deduped
            if duplicates_found > 0:
                print(f"  {task}/{split}: {duplicates_found} (question,image) duplicates resolved")

    # Step 4: Compute statistics
    print("Computing statistics...")
    polarity_counts: Dict[str, Dict[str, Dict[str, int]]] = {
        task: {split: {"positive": 0, "negative": 0} for split in split_sizes}
        for task in FUNCTIONS
    }
    negative_type_counts: Dict[str, Dict[str, Dict[str, int]]] = {
        task: {split: defaultdict(int) for split in split_sizes}
        for task in FUNCTIONS
    }

    for task in FUNCTIONS:
        for split in split_sizes:
            for rec in records[task][split]:
                polarity_counts[task][split][rec["polarity"]] += 1
                if rec["negative_type"]:
                    negative_type_counts[task][split][rec["negative_type"]] += 1

            actual = len(records[task][split])
            target_count = split_sizes[split]
            print(f"  {task}/{split}: {actual}/{target_count} samples "
                  f"(A={polarity_counts[task][split]['positive']}, "
                  f"B={polarity_counts[task][split]['negative']})")

    # Step 4: Build eval records
    eval_records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        task: {split: [] for split in split_sizes} for task in FUNCTIONS
    }
    for task in FUNCTIONS:
        for split in split_sizes:
            eval_records[task][split] = [
                _assemble_eval_sample(rec) for rec in records[task][split]
            ]

    # Step 5: Write output files
    print("Writing output files...")
    file_hashes: Dict[str, str] = {}

    for task in FUNCTIONS:
        task_dir = root_path / task
        task_dir.mkdir(exist_ok=True)
        for split in split_sizes:
            train_path = task_dir / f"{split}.json"
            eval_path = task_dir / f"{split}_eval.json"

            train_data = records[task][split]
            eval_data = eval_records[task][split]

            train_path.write_text(
                json.dumps(train_data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            eval_path.write_text(
                json.dumps(eval_data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            rel_train = str(train_path.relative_to(root_path))
            rel_eval = str(eval_path.relative_to(root_path))
            file_hashes[rel_train] = _sha256(train_path)
            file_hashes[rel_eval] = _sha256(eval_path)

    # Step 6: Build manifest
    print("Building manifest...")
    manifest = _build_manifest(
        root_path=root_path,
        seed=seed,
        split_sizes=split_sizes,
        records=records,
        polarity_counts=polarity_counts,
        negative_type_counts=negative_type_counts,
        file_hashes=file_hashes,
        scenes=list(all_scenes.values()),
        stats=stats,
    )

    manifest_path = root_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_sha256"] = _sha256(manifest_path)
    # Re-write with self-hash
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Dataset generated at: {root_path}")
    print(f"  Scenes: {len(all_scenes)}")
    total_qa = sum(
        len(records[task][split])
        for task in FUNCTIONS
        for split in split_sizes
    )
    print(f"  Total QA samples: {total_qa}")
    print(f"  Manifest: {manifest_path}")

    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_manifest(
    root_path: Path,
    seed: int,
    split_sizes: Dict[str, int],
    records: Dict[str, Dict[str, List[Dict[str, Any]]]],
    polarity_counts: Dict[str, Dict[str, Dict[str, int]]],
    negative_type_counts: Dict[str, Dict[str, Dict[str, int]]],
    file_hashes: Dict[str, str],
    scenes: List[Dict[str, Any]],
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Build manifest dict."""

    # Image distribution summary
    shape_freq: Dict[str, int] = defaultdict(int)
    color_freq: Dict[str, int] = defaultdict(int)
    count_freq: Dict[str, int] = defaultdict(int)
    lr_freq: Dict[str, int] = defaultdict(int)

    for scene in scenes:
        for obj in scene["objects"]:
            shape_freq[obj["shape"]] += 1
            color_freq[obj["color"]] += 1
        count_freq[str(scene["total_objects"])] = count_freq.get(
            str(scene["total_objects"]), 0
        ) + 1
        # Count left/right pairs
        objects = scene["objects"]
        for i, oi in enumerate(objects):
            for j, oj in enumerate(objects):
                if i < j and abs(oi["x"] - oj["x"]) >= MIN_HORIZONTAL_GAP:
                    if oi["x"] < oj["x"]:
                        lr_freq["left"] = lr_freq.get("left", 0) + 1
                    else:
                        lr_freq["right"] = lr_freq.get("right", 0) + 1

    records_summary: Dict[str, Any] = {}
    task_definitions: Dict[str, Any] = {}

    for task in FUNCTIONS:
        task_definitions[task] = {
            "functions": _task_required_functions(task),
            "description": _task_description(task),
            "answer_format": "A=Yes, B=No",
        }
        records_summary[task] = {}
        for split in split_sizes:
            recs = records[task][split]
            answer_counts = Counter(r["answer"] for r in recs)
            polarity_cts = polarity_counts[task][split]
            neg_cts = dict(negative_type_counts[task][split])
            records_summary[task][split] = {
                "count": len(recs),
                "answer_A": answer_counts.get("A", 0),
                "answer_B": answer_counts.get("B", 0),
                "positive": polarity_cts.get("positive", 0),
                "negative": polarity_cts.get("negative", 0),
                "negative_types": neg_cts,
            }

    # Per-file SHA-256 for all data files
    per_file_sha256: Dict[str, str] = {}
    for rel_path, hash_val in file_hashes.items():
        per_file_sha256[rel_path] = hash_val
    # Add scene files
    for scene in scenes:
        scene_path = root_path / "scenes" / f"{scene['scene_id']}.json"
        if scene_path.exists():
            rel = str(scene_path.relative_to(root_path))
            per_file_sha256[rel] = _sha256(scene_path)
    # Add image files
    for scene in scenes:
        img_path = root_path / "images" / f"{scene['scene_id']}.png"
        if img_path.exists():
            rel = str(img_path.relative_to(root_path))
            per_file_sha256[rel] = _sha256(img_path)

    return {
        "schema_version": 2,
        "generator_version": "controlled_format_v1",
        "git_commit": _get_git_commit(),
        "generation_seed": seed,
        "split_policy": "scene_level",
        "split_sizes": split_sizes,
        "number_of_scenes": len(scenes),
        "number_of_samples": {
            task: {
                split: len(records[task][split])
                for split in split_sizes
            }
            for task in FUNCTIONS
        },
        "task_definitions": task_definitions,
        "answer_format": {
            "interface": "A=Yes, B=No",
            "valid_tokens": ["A", "B"],
            "tokenizer_note": "tokenizer audit required; see audit.json",
        },
        "class_balance": records_summary,
        "negative_type_balance": {
            task: {
                split: dict(negative_type_counts[task][split])
                for split in split_sizes
            }
            for task in FUNCTIONS
        },
        "image_distribution_summary": {
            "shape_frequencies": dict(sorted(shape_freq.items())),
            "color_frequencies": dict(sorted(color_freq.items())),
            "object_count_frequencies": dict(sorted(count_freq.items())),
            "left_right_pair_frequencies": dict(sorted(lr_freq.items())),
            "canvas_size": [CANVAS_WIDTH, CANVAS_HEIGHT],
            "min_horizontal_gap": MIN_HORIZONTAL_GAP,
        },
        "min_horizontal_gap": MIN_HORIZONTAL_GAP,
        "generation_stats": stats,
        "files": per_file_sha256,
        "manifest_sha256": None,  # filled after first write
    }


def _get_git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _task_required_functions(task: str) -> List[str]:
    mapping = {
        "A_only": ["A"],
        "B_only": ["B"],
        "C_only": ["C"],
        "A_plus_B": ["A", "B"],
        "B_plus_C": ["B", "C"],
    }
    return mapping.get(task, [])


def _task_description(task: str) -> str:
    mapping = {
        "A_only": "Shape recognition: Is the marked object of the queried shape?",
        "B_only": "Counting: Is the total number of objects equal to the queried count?",
        "C_only": "Spatial relation: Is object X to the left of object Y?",
        "A_plus_B": "Shape-filtered counting: Are there exactly N objects of queried shape?",
        "B_plus_C": "Spatial-filtered counting: Are there exactly N objects to the left of reference?",
    }
    return mapping.get(task, "")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate format-controlled dataset with unified A/B answer interface"
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=730)
    parser.add_argument("--train-size", type=int, default=1600)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=400)
    args = parser.parse_args()

    result = generate(
        output_root=args.output_root,
        seed=args.seed,
        split_sizes={
            "train": args.train_size,
            "val": args.val_size,
            "test": args.test_size,
        },
    )
    print(json.dumps({
        "status": "ok",
        "seed": result["generation_seed"],
        "scenes": result["number_of_scenes"],
        "samples": result["number_of_samples"],
    }))


if __name__ == "__main__":
    main()
