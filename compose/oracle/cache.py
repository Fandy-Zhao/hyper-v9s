import hashlib
import json
import os
from typing import Dict, Iterable, Iterator


def config_hash(config: Dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_jsonl(path: str, records: Iterable[Dict[str, object]]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: str) -> Iterator[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("cache line {} is not an object".format(line_number))
            yield value
