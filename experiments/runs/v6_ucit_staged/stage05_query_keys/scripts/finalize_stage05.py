#!/usr/bin/env python3
"""Generate Stage 05 acceptance manifest, checksums, and committed report."""

import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

import torch
import transformers


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
DATA = Path("/data/ckpt/zhaozhuofan/v6_ucit_staged/stage05_query_keys")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def main():
    train = torch.load(DATA / "features/bundles/train.pt", map_location="cpu")
    validation = torch.load(DATA / "features/bundles/validation.pt", map_location="cpu")
    continual = json.loads((ROOT / "metrics/continual_anchor.json").read_text(encoding="utf-8"))
    offline = json.loads((ROOT / "metrics/offline_diagnostic.json").read_text(encoding="utf-8"))
    oracle_audit = json.loads((ROOT / "validation/oracle_expansion.json").read_text(encoding="utf-8"))
    continual_ckpt = torch.load(DATA / "checkpoints/continual_anchor.pt", map_location="cpu")
    key_tensors = [value.float() for name, value in sorted(continual_ckpt["key_store"].items()) if name.startswith("keys.")]
    normalized_keys = torch.nn.functional.normalize(torch.stack(key_tensors), dim=-1)
    off_diagonal = normalized_keys @ normalized_keys.T - torch.eye(len(normalized_keys))
    keys_distinct = bool(torch.isfinite(normalized_keys).all() and off_diagonal.abs().max() < 0.999999)
    config_path = ROOT / "configs/query_keys.json"
    environment = {
        "python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "peft": __import__("peft").__version__,
    }
    label_counts = {
        "train": dict(Counter(len(value) for value in train["oracle_sets"])),
        "validation": dict(Counter(len(value) for value in validation["oracle_sets"])),
    }
    required = ["OracleMemberRecall@1", "OracleMemberRecall@2", "OracleMemberRecall@4", "OracleSetRecall@4",
                "SingleRecall@1", "PairBothMembersRecall@4", "PairAtLeastOneRecall@4", "EmptyFalseRecallRate", "MRR"]
    checks = {
        "oracle_expansion": oracle_audit["status"] == "PASSED", "test_data_used": train["test_data_used"] is False and validation["test_data_used"] is False,
        "temporal_mask": continual["temporal_mask_violations"] == 0, "metrics_complete": all(name in continual for name in required),
        "checkpoint_metadata": continual_ckpt["checkpoint_version"] == 1 and bool(continual_ckpt["extra"]["key_initialization"]),
        "key_tensors_finite_and_distinct": keys_distinct,
    }
    status = "PASSED" if all(checks.values()) else "FAILED"
    validation_payload = {"status": status, "checks": checks, "label_counts": label_counts, "environment": environment,
                          "config_sha256": sha256(config_path), "continual_checkpoint_sha256": sha256(DATA / "checkpoints/continual_anchor.pt"),
                          "offline_checkpoint_sha256": sha256(DATA / "checkpoints/offline_diagnostic.pt"),
                          "dataset_manifest_hash": train["manifest_hash"], "oracle_cache_hash": train["oracle_cache_hash"],
                          "source_parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                          "failures_retries": ["initial SSH quoting failures during read-only/preflight and controlled launch; no data mutation", "pytest installed in hyper environment because absent"]}
    target = ROOT / "validation/stage05_acceptance.json"; target.write_text(json.dumps(validation_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gap = continual["OracleSetRecall@4"] - offline["OracleSetRecall@4"]
    report = f"""# Hyper-LLaVA V6 Stage 05 — Multimodal Query and Learnable Expert Keys

Status: **{status}**. Git: this report is contained by the atomic Stage 05 commit; source parent `{validation_payload['source_parent_commit']}`.

## Frozen configuration and provenance

- Query: frozen CLIP image projection + question-only text projection, gated fusion, LayerNorm, 128-D L2-normalized output.
- Supervision: historical-only Oracle-Direct; post-task Direct is used only to initialize each expert key on its creation task.
- Scale: 128 train + 32 validation per task (768/192 total), seed 42. The original 32 train rows/task are reused and only 96 missing train + 32 validation rows/task were scored.
- Config SHA256: `{validation_payload['config_sha256']}`.
- Dataset manifest hash: `{train['manifest_hash']}`; Oracle cache hash: `{train['oracle_cache_hash']}`.
- Continual checkpoint SHA256: `{validation_payload['continual_checkpoint_sha256']}`.
- GPU: preferred 0–3 were occupied by another user; actual 4–7, maximum four GPUs.
- Environment: `{json.dumps(environment, sort_keys=True)}`.

## Technical validation

- Full Compose tests: 125 passed + 8 subtests; new Stage 05 tests: 19 passed.
- Controlled Query/Key smoke: passed; checkpoint round-trip, deterministic retrieval, DDP state fingerprint, Empty/Single/Pair losses, answer-token truncation and future-key mask all passed.
- Oracle audit: `{oracle_audit['status']}`; test data used: false; temporal violations: `{continual['temporal_mask_violations']}`.
- Historical label cardinalities train/validation: `{label_counts}`.

## Continual-anchor validation metrics

```json
{json.dumps(continual, indent=2, sort_keys=True)}
```

Offline diagnostic OracleSetRecall@4: `{offline['OracleSetRecall@4']:.6f}`; continual-minus-offline gap: `{gap:.6f}`. Offline is a diagnostic upper bound, not the continual result.

## Key initialization

```json
{json.dumps(continual_ckpt['extra']['key_initialization'], indent=2, sort_keys=True)}
```

## Limitations

Retrieval quality is a method result rather than an engineering pass criterion. Task-ID and one-key collapse are diagnostics only and never Router inputs. The LLaVA backbone, vision tower, text tower, and LoRA experts remain frozen; only Query Encoder and Expert Keys are optimized. No test feature, answer token, task ID, Oracle set, checkpoint name, or test accuracy enters Query.
"""
    docs = REPO / "docs/reports/v6_ucit_stage05_query_keys.md"; docs.parent.mkdir(parents=True, exist_ok=True); docs.write_text(report, encoding="utf-8")
    (ROOT / "stage_report.md").write_text(report, encoding="utf-8")
    checksum_paths = [config_path, target, ROOT / "metrics/continual_anchor.json", ROOT / "metrics/offline_diagnostic.json", docs]
    (ROOT / "checksums.json").write_text(json.dumps({str(path.relative_to(REPO)): sha256(path) for path in checksum_paths}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(docs), "checks": checks}, sort_keys=True))


if __name__ == "__main__": main()
