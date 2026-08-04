"""Stage 03: data-shortcut probes.

For every dataset, train lightweight logistic-regression classifiers that
predict the A/B answer from non-visual or trivial-visual features, and
compare their accuracy to chance (50%) and to the experts' accuracy on the
same test set. High probe accuracy means the answer is predictable without
performing the queried function -> a shortcut that undermines the purity
conclusion for that task.

Feature sets:
  instruction    : bag of words over the question text
  image_mean     : global RGB mean of the image (trivial visual statistic)
  template       : one-hot of (queried_shape, queried_count, queried_relation)
  scene_attr     : non-target scene attributes (scene object count, image
                   dims) - whether the label can be read off scene state
                   alone

Cross-validated (5 folds, fixed seed) on the test split.

Output: artifacts/dual_lora_stage03/shortcut_probe.json
"""

import argparse
import json
import os
import re
import statistics
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

DATASETS = ("A_only", "B_only", "C_only", "A_plus_B", "B_plus_C")
DATA_ROOT = Path("experiments/data/controlled_format_v1_training/instructions")
IMAGE_ROOT = Path("experiments/data/controlled_format_v1")
EVAL_ROOT = Path("experiments/runs/format_controlled_composition_v1/evaluation/seed42")

NUMBER_RE = re.compile(r"\b\d+\b")
SHAPE_WORDS = ("circle", "square", "triangle")
COLOR_WORDS = ("red", "blue", "green", "yellow")


def template_features(record: dict) -> List[float]:
    meta = (record.get("source") or {}).get("metadata") or {}
    shape = meta.get("queried_shape")
    count = meta.get("queried_count")
    relation = meta.get("queried_relation")
    vector = []
    for word in SHAPE_WORDS:
        vector.append(1.0 if shape == word else 0.0)
    for number in range(6):
        vector.append(1.0 if count == number else 0.0)
    for word in ("left", "right"):
        vector.append(1.0 if relation == word else 0.0)
    return vector


def scene_attr_features(record: dict, image_array: np.ndarray) -> List[float]:
    source = record.get("source") or {}
    meta = source.get("metadata") or {}
    # scene-level stats are not in the question file; use image-level stats
    height, width = image_array.shape[:2]
    return [float(width), float(height), float(meta.get("reference_x") or 0.0),
            float(meta.get("true_count") if meta.get("true_count") is not None else 0.0)]


def run_probe(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Dict[str, float]:
    if X.shape[1] == 0:
        return {"accuracy": 0.5, "n_folds": 0}
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accuracies = []
    for train_idx, test_idx in splitter.split(X, y):
        model = LogisticRegression(max_iter=2000, C=1.0)
        model.fit(X[train_idx], y[train_idx])
        accuracies.append(float(model.score(X[test_idx], y[test_idx])))
    return {"accuracy": statistics.fmean(accuracies), "n_folds": len(accuracies),
            "per_fold": accuracies}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="artifacts/dual_lora_stage03")
    args = parser.parse_args()

    report: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset in DATASETS:
        with (DATA_ROOT / dataset / "test_eval.json").open(encoding="utf-8") as handle:
            records = json.load(handle)
        labels = np.array([1.0 if r["answer"] == "A" else 0.0 for r in records])

        texts = [str(r.get("text") or "") for r in records]
        bow = make_pipeline(CountVectorizer(analyzer="word", token_pattern=r"\w+"), LogisticRegression(max_iter=2000))
        text_vec = bow.named_steps["countvectorizer"].fit_transform(texts)

        images_raw = []
        image_means = []
        for record in records:
            path = IMAGE_ROOT / str(record["image"])
            image = Image.open(path).convert("RGB")
            array = np.asarray(image, dtype=np.float32)
            images_raw.append(array)
            image_means.append(array.mean(axis=(0, 1)) / 255.0)
        image_mean = np.stack(image_means)

        templates = np.array([template_features(r) for r in records], dtype=np.float32)
        scene_attrs = np.array([scene_attr_features(r, img) for r, img in zip(records, images_raw)], dtype=np.float32)

        def evaluate(X, name: str) -> None:
            if X.shape[0] == 0:
                report[dataset][name] = {"accuracy": 0.5}
                return
            report[dataset][name] = run_probe(X, labels)

        report[dataset] = {}
        evaluate(text_vec.toarray(), "instruction_bow")
        evaluate(image_mean, "image_global_mean")
        evaluate(templates, "task_template")
        evaluate(scene_attrs, "scene_attributes")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "shortcut_probe.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
