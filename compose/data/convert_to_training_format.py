"""
Convert controlled_format_v1 data to LLaVA-compatible training format.

Usage:
  python -m compose.data.convert_to_training_format \
    --source-root experiments/data/controlled_format_v1 \
    --output-root experiments/data/controlled_format_v1_training
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

FUNCTIONS = ("A_only", "B_only", "C_only", "A_plus_B", "B_plus_C")
SPLITS = ("train", "val", "test")


_META_FIELDS = (
    "scene_id",
    "required_functions",
    "answer",
    "polarity",
    "negative_type",
    "metadata",
)


def convert_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert new-format sample to LLaVA training format (conversations).

    Adds source metadata fields (scene_id, required_functions, answer, polarity,
    negative_type, metadata) at top level so downstream preflight audits and
    per-sample evaluations can join on them without consulting the source files.
    """
    question = sample["question"]
    answer = sample["answer"]  # "A" or "B"

    converted = {
        "id": sample["id"],
        "question_id": sample["id"],
        "task_id": f"Controlled/{sample['task']}",
        "template_id": f"{sample['id']}",
        "image": sample["image"],
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\n{question}\nAnswer the question using a single word or phrase.",
            },
            {
                "from": "gpt",
                "value": answer,
            },
        ],
    }
    for field in _META_FIELDS:
        if field in sample:
            converted[field] = sample[field]
    return converted


def convert_eval_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert new-format sample to eval format.

    ``text`` carries the FULL human turn (question plus the fixed suffix used
    in training) so that NLL and generation evaluation present exactly the
    training prompt. The full source sample is embedded under "source" so the
    format-controlled evaluation can report scene_id, polarity, negative_type,
    required_functions, queried_* metadata, options, and the true count.
    """
    return {
        "question_id": sample["id"],
        "task_id": f"Controlled/{sample['task']}",
        "template_id": f"{sample['id']}",
        "image": sample["image"],
        "text": f"{sample['question']}\nAnswer the question using a single word or phrase.",
        "answer": sample["answer"],
        "source": sample,
    }


def convert_dataset(source_root: str, output_root: str) -> None:
    source = Path(source_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)

    # Symlink images directory (don't copy)
    img_src = source / "images"
    img_dst = output / "images"
    if not img_dst.exists():
        img_dst.symlink_to(os.path.relpath(img_src, output))

    instructions_dir = output / "instructions"
    instructions_dir.mkdir(exist_ok=True)

    for task in FUNCTIONS:
        task_dir = instructions_dir / task
        task_dir.mkdir(exist_ok=True)

        for split in SPLITS:
            src_file = source / task / f"{split}.json"
            if not src_file.exists():
                print(f"  WARNING: missing {src_file}")
                continue

            with open(src_file, "r", encoding="utf-8") as f:
                samples = json.load(f)

            train_records = [convert_sample(s) for s in samples]
            eval_records = [convert_eval_sample(s) for s in samples]

            train_path = task_dir / f"{split}.json"
            eval_path = task_dir / f"{split}_eval.json"

            train_path.write_text(
                json.dumps(train_records, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            eval_path.write_text(
                json.dumps(eval_records, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            print(f"  {task}/{split}: {len(train_records)} records")

    print(f"\nTraining-format dataset written to: {output}")
    print(f"  Compatible with: --data_path <path>/instructions/<task>/<split>.json")
    print(f"                   --image_folder <path>")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    convert_dataset(args.source_root, args.output_root)


if __name__ == "__main__":
    main()
