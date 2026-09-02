"""Reuse the repository's original UCIT evaluator for V7 validation pruning."""

import argparse
import json
from pathlib import Path

from compose.eval.formal_ucit_eval import _score_answers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metric = _score_answers(
        Path(args.work_root),
        args.task_index,
        args.task_index,
        Path(args.predictions_file),
        annotation_file=args.annotation_file,
    )
    metric["validation_only"] = True
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metric, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metric, sort_keys=True))


if __name__ == "__main__":
    main()
