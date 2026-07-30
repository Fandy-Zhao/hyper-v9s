import argparse
import json
from typing import Dict, Iterable, List

from compose.data.records import answer_text


def imagenet_r_exact_match(
    annotations: Iterable[Dict[str, object]],
    predictions: Iterable[Dict[str, object]],
) -> Dict[str, object]:
    """Replicate the effective UCIT Task1 exact-match metric deterministically."""

    answers = {str(item["question_id"]): answer_text(item) for item in annotations}
    rows: List[Dict[str, object]] = []
    correct = 0
    for prediction in predictions:
        question_id = str(prediction["question_id"])
        if question_id not in answers:
            raise ValueError("prediction has unknown question_id {}".format(question_id))
        predicted = str(prediction["text"]).strip()
        target = answers[question_id]
        is_correct = predicted.upper() == target.upper()
        correct += int(is_correct)
        rows.append(
            {
                "question_id": question_id,
                "prediction": predicted,
                "answer": target,
                "correct": is_correct,
            }
        )
    if len(rows) != len(answers):
        raise ValueError(
            "prediction count does not match annotations; predictions={}, annotations={}".format(
                len(rows), len(answers)
            )
        )
    return {
        "metric": "imagenet_r_case_insensitive_exact_match",
        "samples": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "accuracy_percent": 100.0 * correct / len(rows) if rows else 0.0,
        "rows": rows,
    }


def load_jsonl(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    with open(args.annotation_file, "r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    result = imagenet_r_exact_match(annotations, load_jsonl(args.predictions_file))
    rows = result.pop("rows")
    result["incorrect"] = [row for row in rows if not row["correct"]]
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "incorrect"}, sort_keys=True))


if __name__ == "__main__":
    main()
