"""Four-rank smoke against the real Hyper LoRA implementation."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from Hyper.peft.tuners.clitmoelora import HyperMOELoraLinear
from compose.experts import ExpertMetadata, ExpertRegistry
from compose.lora import (AdapterBridge, CompositionRuntime, ExpertComposer,
                          RMSStatistics, StatisticKey)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = HyperMOELoraLinear("default", 8, 8, r=8, lora_alpha=16, expert_num=2,
                                       cur_task=0, task_embedding_dim=4, train_signal=True,
                                       layer=0, expert_weight=[1.0, 0.0])

    def forward(self, inputs):
        return self.proj(inputs)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(42)
    model = Tiny().to(device=device, dtype=torch.bfloat16)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bridge = AdapterBridge(model)
    registry = ExpertRegistry()
    for expert_id in (0, 1):
        registry.register(ExpertMetadata(expert_id=expert_id, adapter_name="default"))
    calibration = torch.randn(4, 8, device=device, dtype=torch.bfloat16) + rank * 0.01
    base = torch.nn.functional.linear(calibration, model.proj.weight, model.proj.bias)
    stats = RMSStatistics({"calibration_split": "train_calibration_smoke", "checkpoint_hash": "tiny-fixed",
                           "dataset_manifest_hash": "seed42-rank-sharded", "composition_config_hash": "29eacd7c25718e085a864a742b3c0d8ab343759b2be958202b11cc846923d0f2"})
    for expert_id in (0, 1):
        delta = bridge.compute_expert_delta(model.proj, expert_id, calibration)
        stats.update(StatisticKey(expert_id, "proj", "proj", type(model.proj).__name__), delta, base + delta, base)
    stats.all_reduce_(device)
    state_text = json.dumps(stats.state_dict(), sort_keys=True)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, state_text)
    assert len(set(gathered)) == 1
    output_root = Path(os.environ["PAIR_SMOKE_OUTPUT"])
    output_root.mkdir(parents=True, exist_ok=True)
    stats_path = output_root / "rms_statistics.json"
    if rank == 0:
        stats.save_json(stats_path)
    dist.barrier()
    restored = RMSStatistics.load_json(stats_path, stats.provenance, registry)
    inputs = torch.randn(3, 8, device=device, dtype=torch.bfloat16)
    composer = ExpertComposer(bridge, restored)
    with CompositionRuntime(registry, bridge, composer, [0, 1], [1], "direct_sum"):
        direct = model(inputs)
        direct.float().sum().backward()
        frozen_grad = all(parameter.grad is None or not torch.count_nonzero(parameter.grad) for _, parameter in bridge._expert_parameters[0])
        new_grad = any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for _, parameter in bridge._expert_parameters[1])
    model.zero_grad(set_to_none=True)
    with CompositionRuntime(registry, bridge, composer, [1, 0], [], "rms_calibrated"):
        rms = model(inputs)
    assert frozen_grad and new_grad and torch.isfinite(direct).all() and torch.isfinite(rms).all()
    local = {"rank": rank, "active": [0, 1], "trainable": [1], "direct_finite": True,
             "rms_finite": True, "old_expert_frozen": frozen_grad, "new_expert_gradient": new_grad,
             "statistics_identical": True, "statistics_sample_count": next(iter(restored.entries.values()))["sample_count"]}
    all_rows = [None] * dist.get_world_size()
    dist.all_gather_object(all_rows, local)
    if rank == 0:
        with (output_root / "summary.json").open("x", encoding="utf-8") as handle:
            json.dump({"status": "PASSED", "world_size": dist.get_world_size(), "ranks": all_rows}, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps({"status": "PASSED", "world_size": dist.get_world_size()}, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
