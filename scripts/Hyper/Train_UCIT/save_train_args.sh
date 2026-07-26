# Save the effective deepspeed command and training arguments next to each checkpoint.
hyper_save_train_args_json() {
    local launcher="$1"
    shift

    local -a args=("$@")
    local output_dir=""
    local i
    for ((i = 0; i < ${#args[@]}; i++)); do
        if [[ "${args[$i]}" == "--output_dir" && $((i + 1)) -lt ${#args[@]} ]]; then
            output_dir="${args[$((i + 1))]}"
            break
        fi
    done

    if [[ -z "$output_dir" ]]; then
        echo "[save_train_args] skip: --output_dir not found" >&2
        return 0
    fi

    mkdir -p "$output_dir"

    local task_script="${BASH_SOURCE[1]:-$0}"
    local json_path="$output_dir/training_args.json"

    PROMPT_VERSION="${PROMPT_VERSION:-}" MODEL_VERSION="${MODEL_VERSION:-}" \
        /home/zhaozhuofan/miniconda3/envs/hyper/bin/python - "$json_path" "$task_script" "$PWD" "$launcher" "${args[@]}" <<'PY'
import getpass
import json
import os
import shlex
import socket
import sys
from datetime import datetime, timezone

json_path, task_script, cwd, launcher, *args = sys.argv[1:]

parsed = {}
i = 0
while i < len(args):
    item = args[i]
    if item.startswith("--"):
        key = item[2:].replace("-", "_")
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            value = args[i + 1]
            i += 2
        else:
            value = True
            i += 1
        if key in parsed:
            if not isinstance(parsed[key], list):
                parsed[key] = [parsed[key]]
            parsed[key].append(value)
        else:
            parsed[key] = value
    else:
        parsed.setdefault("_positionals", []).append(item)
        i += 1

payload = {
    "saved_at": datetime.now(timezone.utc).isoformat(),
    "hostname": socket.gethostname(),
    "user": getpass.getuser(),
    "working_dir": cwd,
    "task_script": os.path.abspath(task_script),
    "launcher": launcher,
    "command": shlex.join([launcher, *args]),
    "argv": [launcher, *args],
    "parsed_args": parsed,
    "env": {
        "PROMPT_VERSION": os.environ.get("PROMPT_VERSION"),
        "MODEL_VERSION": os.environ.get("MODEL_VERSION"),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"),
    },
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(f"[save_train_args] wrote {json_path}")
PY
}

deepspeed() {
    hyper_save_train_args_json deepspeed "$@"
    command deepspeed "$@"
}
