import argparse
import hashlib
import json
import os
from typing import Dict, Iterable

import torch

from .checkpoint import MANIFEST_NAME, WEIGHTS_NAME


def _expert_state(checkpoint_dir: str, expert_id: int) -> Dict[str, torch.Tensor]:
    path = os.path.join(checkpoint_dir, WEIGHTS_NAME)
    state = torch.load(path, map_location="cpu")
    marker = ".experts.{}.".format(expert_id)
    selected = {key: value for key, value in state.items() if marker in key}
    if len(selected) != 448:
        raise ValueError(
            "expert {} expected 448 tensors in {}, found {}".format(
                expert_id, checkpoint_dir, len(selected)
            )
        )
    return selected


def _digest(state: Dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _metadata(checkpoint_dir: str, expert_id: int) -> Dict[str, object]:
    with open(os.path.join(checkpoint_dir, MANIFEST_NAME), encoding="utf-8") as handle:
        manifest = json.load(handle)
    matches = [
        entry for entry in manifest.get("experts", [])
        if int(entry.get("expert_id", -1)) == expert_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected one metadata entry for expert {}, found {}".format(
                expert_id, len(matches)
            )
        )
    return matches[0]


def verify_expert_isolation(
    before_dir: str, after_dir: str, expert_ids: Iterable[int]
) -> Dict[str, object]:
    results = []
    for expert_id in expert_ids:
        before = _expert_state(before_dir, expert_id)
        after = _expert_state(after_dir, expert_id)
        if set(before) != set(after):
            raise AssertionError("expert {} tensor keys changed".format(expert_id))
        changed = [key for key in sorted(before) if not torch.equal(before[key], after[key])]
        if changed:
            raise AssertionError(
                "expert {} changed in {} tensors: {}".format(
                    expert_id, len(changed), changed[:8]
                )
            )
        before_metadata = _metadata(before_dir, expert_id)
        after_metadata = _metadata(after_dir, expert_id)
        for field in ("expert_id", "name", "source_checkpoint", "trained_steps", "tags"):
            if before_metadata.get(field) != after_metadata.get(field):
                raise AssertionError(
                    "expert {} metadata field {!r} changed".format(expert_id, field)
                )
        before_origin = before_metadata.get("origin_task_id")
        if before_origin is not None and before_origin != after_metadata.get("origin_task_id"):
            raise AssertionError(
                "expert {} origin_task_id changed".format(expert_id)
            )
        results.append(
            {
                "expert_id": expert_id,
                "tensor_count": len(before),
                "sha256": _digest(before),
                "exactly_equal": True,
            }
        )
    return {
        "before_checkpoint": before_dir,
        "after_checkpoint": after_dir,
        "experts": results,
        "exactly_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-dir", required=True)
    parser.add_argument("--after-dir", required=True)
    parser.add_argument("--expert-ids", required=True)
    parser.add_argument("--output-file")
    args = parser.parse_args()
    expert_ids = [int(value) for value in args.expert_ids.split(",") if value.strip()]
    if not expert_ids:
        raise ValueError("expert_ids must not be empty")
    result = verify_expert_isolation(args.before_dir, args.after_dir, expert_ids)
    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
