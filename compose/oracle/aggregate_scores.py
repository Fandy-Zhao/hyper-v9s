import argparse
import json
import os

from .cache import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    rows = list(read_jsonl(args.input_file))
    if not rows:
        raise ValueError("selection score cache is empty")
    names = sorted(set(str(row["selection_name"]) for row in rows))
    hashes = sorted(set(str(row["config_hash"]) for row in rows))
    if len(names) != 1 or len(hashes) != 1:
        raise ValueError("selection score cache mixes configurations")
    result = {
        "samples": len(rows),
        "selection_name": names[0],
        "config_hash": hashes[0],
        "mean_nll": sum(float(row["nll"]) for row in rows) / len(rows),
        "mean_target_token_count": sum(int(row["target_token_count"]) for row in rows)
        / len(rows),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
