import argparse
import json
import os
from typing import List


def build_mixture(input_paths: List[str], task_ids: List[str]):
    if len(input_paths) != len(task_ids) or not input_paths:
        raise ValueError("input paths and task IDs must be non-empty and have equal length")
    output = []
    task_counts = {}
    for path, task_id in zip(input_paths, task_ids):
        with open(path, encoding="utf-8") as handle:
            records = json.load(handle)
        task_counts[task_id] = len(records)
        for index, source in enumerate(records):
            record = dict(source)
            original_id = str(record.get("question_id", record.get("id", index)))
            record["source_sample_id"] = original_id
            record["task_id"] = task_id
            record["question_id"] = "{}:{}".format(task_id, original_id)
            if "id" in record:
                record["id"] = "{}:{}".format(task_id, record["id"])
            output.append(record)
    return output, task_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    records, task_counts = build_mixture(args.input, args.task_id)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"samples": len(records), "task_counts": task_counts}, sort_keys=True))


if __name__ == "__main__":
    main()
