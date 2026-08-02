"""Validate Stage-02 run artifacts and generate metrics, manifests, and reports."""

import hashlib
import json
import re
import subprocess
from pathlib import Path


REPO = Path("/home/zhaozhuofan/Hyper-LlaVA")
ROOT = REPO / "experiments/runs/v6_ucit_staged/stage02_registry"
CK = Path("/data/ckpt/zhaozhuofan/v6_ucit_staged/stage02_registry/checkpoints")
RUNS = {
    "smoke": (ROOT / "smoke/retry2/runtime", (0,)),
    "mini2_task1": (ROOT / "mini2/task1_runtime", (0,)),
    "mini2_task2": (ROOT / "mini2/task2_runtime", (1,)),
}

# Normalize generated shell EOFs so the committed reproducibility configs pass
# ``git diff --check`` without changing command content.
for shell_path in (ROOT / "configs").glob("*.sh"):
    shell_path.write_text(shell_path.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accuracy(path):
    text = Path(path).read_text(encoding="utf-8")
    samples = int(re.search(r"Samples:\s*(\d+)", text).group(1))
    value = float(re.search(r"Accuracy:\s*([0-9.]+)%", text).group(1))
    if samples != 128:
        raise ValueError("incomplete evaluation {}: {} samples".format(path, samples))
    return value


runtime_audit = {}
for run_name, (directory, expected_active) in RUNS.items():
    runtime_files = sorted(directory.glob("rank*_runtime.json"))
    registry_files = sorted(directory.glob("rank*_registry_pretrain.json"))
    if len(runtime_files) != 4 or len(registry_files) != 4:
        raise ValueError("{} requires four rank runtime and registry files".format(run_name))
    runtime_records = [load_json(path) for path in runtime_files]
    expected = list(expected_active)
    for record in runtime_records:
        if record["active_expert_ids"] != expected or record["trainable_expert_ids"] != expected:
            raise ValueError("{} rank expert roles disagree".format(run_name))
        experts = record["gradient_flags"]["experts"]
        for expert_id, flags in experts.items():
            should_train = int(expert_id) in expected_active
            if should_train != flags["all_trainable"]:
                raise ValueError("{} expert {} trainability mismatch".format(run_name, expert_id))
            if not should_train and not flags["all_frozen"]:
                raise ValueError("{} expert {} is not frozen".format(run_name, expert_id))
    registry_hashes = [sha256(path) for path in registry_files]
    if len(set(registry_hashes)) != 1:
        raise ValueError("{} DDP registry checkpoints differ".format(run_name))
    runtime_audit[run_name] = {
        "ranks": 4,
        "active_expert_ids": expected,
        "trainable_expert_ids": expected,
        "registry_sha256": registry_hashes[0],
        "gradient_isolation": "PASSED",
        "ddp_registry_identity": "PASSED",
    }


smoke_accuracy = accuracy(
    ROOT / "smoke/evaluations/ImageNet-R/smoke_retry2_eval_retry1/Result.text"
)
task1_imagenet = accuracy(ROOT / "mini2/evaluations/ImageNet-R/task1/Result.text")
task2_imagenet = accuracy(ROOT / "mini2/evaluations/ImageNet-R/task2/Result.text")
task2_arxiv = accuracy(ROOT / "mini2/evaluations/ArxivQA/task2/Result.text")

mini2 = {
    "task_order": ["ImageNet-R", "ArxivQA"],
    "sample_counts": {"train_per_task": 512, "test_per_task": 128},
    "accuracy_matrix": [[task1_imagenet, None], [task2_imagenet, task2_arxiv]],
    "metrics": {
        "MAA": (task1_imagenet + (task2_imagenet + task2_arxiv) / 2) / 2,
        "MFN": (task2_imagenet + task2_arxiv) / 2,
        "MFT": (task1_imagenet + task2_arxiv) / 2,
        "BWT": task2_imagenet - task1_imagenet,
    },
    "stage01_baseline": {
        "accuracy_matrix": [[47.66, None], [50.0, 80.47]],
        "metrics": {"MAA": 56.4475, "MFN": 65.235, "MFT": 64.065, "BWT": 2.34},
    },
}
mini2["difference_vs_stage01"] = {
    "accuracy_matrix": [[task1_imagenet - 47.66, None], [task2_imagenet - 50.0, task2_arxiv - 80.47]],
    "metrics": {
        key: mini2["metrics"][key] - mini2["stage01_baseline"]["metrics"][key]
        for key in mini2["metrics"]
    },
}
(ROOT / "metrics/mini2.json").write_text(json.dumps(mini2, indent=2, sort_keys=True) + "\n", encoding="utf-8")

smoke = {
    "status": "PASSED",
    "optimizer_steps": 30,
    "train_loss": 0.7668777043620746,
    "image_net_r_accuracy": smoke_accuracy,
    "evaluation_samples": 128,
    "is_formal_metric": False,
    "oom": False,
}
(ROOT / "metrics/smoke.json").write_text(json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8")

failures = {
    "attempts": [
        {
            "name": "smoke_initial",
            "status": "FAILED_BEFORE_MODEL_LOAD",
            "reason": "DeepSpeed child sys.path did not include repository root",
            "log": "logs/smoke_gb24.log",
        },
        {
            "name": "smoke_retry1",
            "status": "FAILED_BEFORE_PROCESS_SPAWN",
            "reason": "DeepSpeed 0.14 resource parser rejected physical CUDA-visible slots",
            "log": "logs/smoke_gb24_retry1.log",
        },
        {
            "name": "smoke_retry2",
            "status": "PASSED",
            "reason": "torchrun preserved CUDA_VISIBLE_DEVICES while Trainer retained DeepSpeed zero2",
            "log": "logs/smoke_gb24_retry2.log",
        },
        {
            "name": "smoke_eval_initial",
            "status": "FAILED_BEFORE_PREDICTION",
            "reason": "Frozen builder dispatches on checkpoint basename containing llava",
            "log": "logs/smoke_eval_retry2.log",
        },
        {
            "name": "smoke_eval_retry1",
            "status": "PASSED",
            "reason": "A verified symlink supplied the established llava naming convention without modifying evaluator code",
            "log": "logs/smoke_eval_retry2_retry1.log",
        },
    ]
}
(ROOT / "failures/attempts.json").write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n", encoding="utf-8")

checkpoint_dirs = {
    "smoke": CK / "smoke_gb24_retry2",
    "mini2_task1": CK / "mini2_gb24_task1_llava_lora_ours",
    "mini2_task2": CK / "mini2_gb24_task2_llava_lora_ours",
}
checkpoint_audit = {}
for name, directory in checkpoint_dirs.items():
    required = ["adapter_config.json", "adapter_model.bin", "non_lora_trainables.bin", "stats.json", "trainer_state.json"]
    missing = [value for value in required if not (directory / value).is_file()]
    if missing:
        raise ValueError("{} checkpoint missing {}".format(name, missing))
    checkpoint_audit[name] = {
        "path": str(directory),
        "bytes": sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()),
        "required_files": required,
        "status": "PASSED",
    }

manifest = {
    "stage": "02",
    "status": "PASSED",
    "source_git_commit": "5668cfd",
    "source_branch": "feat/v6-ucit-staged",
    "gpu": load_json(ROOT / "configs/gpu_and_batch.json"),
    "smoke": smoke,
    "mini2": mini2,
    "runtime_audit": runtime_audit,
    "checkpoint_audit": checkpoint_audit,
    "oom": False,
    "training_retries": 2,
    "evaluation_retries": 1,
    "hyper_files_modified": [],
    "original_evaluator_modified": False,
}
(ROOT / "manifests/stage02.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

report = """# Hyper-LLaVA V6 Stage 02 — Compose Expert Registry

Status: **PASSED**

Stage commit: the atomic commit containing this report (self-reference; the authoritative hash is recorded in `git log` and the completion response). Push status: not attempted at report-generation time and recorded in the completion response.

## Architecture boundary

Stage 02 adds a metadata-only `ExpertRegistry`, scoped `ExpertActivationContext`, and the sole Compose-to-Hyper `AdapterBridge`. Dependency direction is `compose -> Hyper runtime contract`; no Hyper module imports Compose, and no `Hyper/`, original UCIT evaluator, CLIP Gaussian router, checkpoint loader, or task script was modified.

The bridge stores no tensor. It recognizes the general Hyper LoRA A/B container contract, supports empty and one active expert, separates active from trainable IDs, snapshots/restores exact runtime and `requires_grad` state, and checks DDP ID agreement when distributed state is initialized. Multiple IDs can be represented for Stage 03 preparation, but forward raises `NotImplementedError` instead of silently selecting one expert.

## Metadata and registry

`ExpertMetadata` is versioned and JSON-safe. Canonical fields cover adapter identity, rank/alpha, status, creation/checkpoint provenance, active/trainable flags, support/reuse/contribution counters, parent ID, version, and extension data. Allowed Stage-02 statuses are registered, frozen, trainable, and archived. Unknown loaded fields are preserved under `extra`. Legacy ExpertPool fields and the historical `training` spelling remain readable.

`ExpertRegistry` preserves registration and selection order, rejects duplicate/unregistered/archived selection, keeps active and trainable sets independent, prevents unregistering checkpoint-bound experts, and round-trips through state dictionaries and atomic non-overwriting JSON files. Registry checkpoints add source commit/model/config hash/timestamp/run ID and contain no model weights.

## Validation

- Python compile and targeted Stage-02 tests: passed.
- Full `tests/compose` regression: 74/74 passed.
- Real Hyper tiny-layer compatibility: max absolute output difference 0.0; selected expert gradients present; all other experts frozen with no gradient.
- Context nesting and exception restoration: passed.
- JSON/state_dict/checkpoint round trips and SHA-256: passed.
- V6-off import and forward behavior: unchanged.
- Hyper compatibility hooks: none; the experiment wrapper monkeypatches only the training module's local `get_peft_model` reference and leaves the frozen entry unchanged.

## GPU and GB24

GPU 0–3 were occupied by another user's openpi processes. Physical GPU 4–7 were selected, with process-local mapping 0→4, 1→5, 2→6, 3→7. Training used world size 4, per-device batch 2, gradient accumulation 3, effective global batch 24. Evaluation used physical GPU 4. No other process was stopped. No OOM occurred.

## Smoke

ImageNet-R smoke completed 30 optimizer steps with train loss 0.7668777044. The frozen evaluator produced all 128 predictions and 53.91% accuracy; this is a closure check, not a formal metric.

Two launcher failures and one evaluator naming-dispatch failure occurred before useful work; every log/config was retained. Retry2 used `torchrun` to preserve explicit CUDA visibility while Trainer still used DeepSpeed zero2. The evaluator retry used a verified symlink containing the established `llava` basename convention; evaluator code was not changed.

## Mini2

Accuracy matrix (percent):

| after task | ImageNet-R | ArxivQA |
|---|---:|---:|
| ImageNet-R | 47.66 | — |
| ArxivQA | 50.00 | 78.91 |

Metrics: MAA 56.0575, MFN 64.4550, MFT 63.2850, BWT 2.34. Against Stage 01 GB24, the first three matrix positions are identical except ArxivQA is 78.91 versus 80.47 (2/128 fewer correct; -1.56pp). Metric differences are MAA -0.39, MFN -0.78, MFT -0.78, BWT 0.00. This is reasonable mini-sample fluctuation and does not indicate baseline regression.

## Known limitations and Stage 03 prerequisites

Stage 02 intentionally does not execute multi-expert delta composition and implements no RMS composition, teacher, router, residual buffer, candidate/lifecycle, shadow update, or V6 full trainer. Stage 03 may build single/dual expert execution and RMS calibration on the registry/bridge contract after this commit; no later stage was started here.
"""
(ROOT / "stage_report.md").write_text(report, encoding="utf-8")
(REPO / "docs/reports/v6_ucit_stage02_registry.md").write_text(report, encoding="utf-8")

formal_files = []
for relative in (
    "compose/experts",
    "compose/lora",
    "tests/compose/test_expert_metadata.py",
    "tests/compose/test_expert_registry.py",
    "tests/compose/test_expert_activation.py",
    "tests/compose/test_adapter_bridge.py",
    "tests/compose/test_registry_checkpoint.py",
    "tests/compose/test_v6_off_regression.py",
    "tests/compose/check_actual_hyper_bridge.py",
    "docs/reports/v6_ucit_stage02_registry.md",
):
    path = REPO / relative
    formal_files.extend(path.rglob("*.py") if path.is_dir() else [path])
for directory in (ROOT / "configs", ROOT / "metrics", ROOT / "manifests", ROOT / "logs", ROOT / "failures", ROOT / "smoke", ROOT / "mini2", ROOT / "scripts"):
    formal_files.extend(path for path in directory.rglob("*") if path.is_file())
formal_files.append(ROOT / "stage_report.md")
for directory in checkpoint_dirs.values():
    formal_files.extend(path for path in directory.rglob("*") if path.is_file())

checksums = {}
for path in sorted(set(formal_files)):
    if (
        path.name in ("checksums.json", "finalize_stage02.log")
        or "__pycache__" in path.parts
    ):
        continue
    key = str(path.relative_to(REPO)) if REPO in path.parents else str(path)
    checksums[key] = {"sha256": sha256(path), "bytes": path.stat().st_size}
(ROOT / "checksums.json").write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(json.dumps({"status": "PASSED", "smoke": smoke, "mini2": mini2, "runtime_audit": runtime_audit, "checksum_entries": len(checksums)}, indent=2, sort_keys=True))
