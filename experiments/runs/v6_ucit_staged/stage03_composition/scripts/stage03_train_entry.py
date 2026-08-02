"""Stage-03 single-mode entry coupling Registry, Bridge and Composer.

This wrapper leaves ``llava/train/train_MOE.py`` unchanged.  It replaces only
that module's ``get_peft_model`` reference for this process, then delegates the
entire training loop to the frozen Hyper entry.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

# DeepSpeed launches this file by absolute path, so Python otherwise places the
# artifact directory (not the repository) on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compose.experts import (
    ExpertMetadata,
    ExpertRegistry,
    ExpertStatus,
    save_registry_checkpoint,
)
from compose.lora import AdapterBridge, CompositionRuntime, ExpertComposer
from llava.train import train_MOE


_ORIGINAL_GET_PEFT_MODEL = train_MOE.get_peft_model


def _argument(name, default=None):
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return default


def _config_hash(config):
    payload = {
        "r": int(config.r),
        "lora_alpha": float(config.lora_alpha),
        "lora_dropout": float(config.lora_dropout),
        "expert_num": int(config.expert_num),
        "cur_task": int(config.cur_task),
        "target_modules": sorted(config.target_modules),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return payload, hashlib.sha256(encoded).hexdigest()


def _compose_get_peft_model(model, config, *args, **kwargs):
    peft_model = _ORIGINAL_GET_PEFT_MODEL(model, config, *args, **kwargs)
    current = int(config.cur_task)
    expert_count = int(config.expert_num)
    if not 0 <= current < expert_count:
        raise ValueError("cur_task must identify one physical Hyper expert")

    registry = ExpertRegistry()
    per_expert_rank = int(config.r) // expert_count
    if per_expert_rank <= 0 or int(config.r) % expert_count:
        raise ValueError("Hyper LoRA rank must divide evenly across experts")
    previous_checkpoint = _argument("--previous_task_model_path")
    for expert_id in range(expert_count):
        learned = expert_id <= current
        registry.register(
            ExpertMetadata(
                expert_id=expert_id,
                adapter_name="default",
                rank=per_expert_rank,
                alpha=float(config.lora_alpha),
                status=ExpertStatus.FROZEN if learned else ExpertStatus.REGISTERED,
                creation_task=expert_id if learned else None,
                checkpoint_path=previous_checkpoint if expert_id < current else None,
                extra={"physical_hyper_expert": True},
            )
        )
    registry.set_active_ids([current])
    registry.set_trainable_ids([current])

    bridge = AdapterBridge(peft_model)
    composer = ExpertComposer(bridge)
    runtime = CompositionRuntime(registry, bridge, composer, [current], [current], "single")
    runtime.__enter__()
    peft_model._compose_v6_registry = registry
    peft_model._compose_v6_bridge = bridge
    peft_model._compose_v6_composer = composer
    peft_model._compose_v6_runtime = runtime

    artifact_dir = os.environ.get("V6_REGISTRY_ARTIFACT_DIR")
    if not artifact_dir:
        raise RuntimeError("V6_REGISTRY_ARTIFACT_DIR is required for Stage 02 runs")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    config_payload, config_hash = _config_hash(config)
    target = Path(artifact_dir) / "rank{}_registry_pretrain.json".format(rank)
    checksum = save_registry_checkpoint(
        registry,
        target,
        source_git_commit=os.environ.get("V6_SOURCE_GIT_COMMIT", "unknown"),
        source_model_identifier=_argument("--model_name_or_path", "unknown-model"),
        source_adapter_config_hash=config_hash,
        timestamp=os.environ.get("V6_RUN_TIMESTAMP"),
        run_id=os.environ.get("V6_RUN_ID"),
    )
    runtime = {
        "rank": rank,
        "active_expert_ids": list(bridge.get_active_experts()),
        "trainable_expert_ids": list(bridge.get_trainable_experts()),
        "composition_mode": "single",
        "adapter_config": config_payload,
        "adapter_config_sha256": config_hash,
        "registry_sha256": checksum,
        "gradient_flags": bridge.verify_grad_flags(),
    }
    runtime_path = Path(artifact_dir) / "rank{}_runtime.json".format(rank)
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return peft_model


train_MOE.get_peft_model = _compose_get_peft_model

if __name__ == "__main__":
    train_MOE.train()
