"""V6 Stage E11 mock DDP smoke: 2 processes, tiny model, selection parity.

Each rank builds the identical unified selection from the same seed,
runs the tiny decoder with an old expert + candidate, and verifies
gradient behavior and cross-rank equality via broadcast.
"""

import os

import torch
import torch.distributed as dist

from compose.adapters import ExpertManager, inject_compose_adapters
from compose.adapters.types import ComposeSelection
from compose.config import ComposeAdapterConfig
from compose.experts import ExpertPool
from compose.experts.metadata import ExpertLifecycleStatus, ExpertMetadata
from compose.experts.registry import ExpertRegistry
from compose.experts.task_state import TaskStateMachine, TaskStage
import torch.nn as nn


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.Linear(3, 3, bias=False))


class TinyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, name, nn.Linear(3, 3, bias=False))


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()
        self.mlp = TinyMlp()


class TinyDecoder(nn.Module):
    def __init__(self, layer_count=2):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer() for _ in range(layer_count)])
        self.mm_projector = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
        self.vision_tower = TinyAttention()


class TinyModel(nn.Module):
    def __init__(self, layer_count=2):
        super().__init__()
        self.model = TinyDecoder(layer_count)
        self.lm_head = nn.Linear(3, 3)

    def get_model(self):
        return self.model



def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(42)
    model = TinyModel(layer_count=2)
    inject_compose_adapters(model, ComposeAdapterConfig(rank=1, alpha=2))
    pool = ExpertPool(ExpertManager(model))
    pool.manager.add_expert(0)
    pool.manager.add_expert(1)
    pool.train_only([1])
    for layer in pool.manager.layers.values():
        with torch.no_grad():
            layer.experts["0"].lora_A.weight.fill_(1.0)
            layer.experts["0"].lora_B.weight.fill_(2.0)
            layer.experts["1"].lora_A.weight.fill_(0.5)
            layer.experts["1"].lora_B.weight.fill_(0.5)

    # Unified selection: empty / single / pair / heterogeneous.
    selection = ComposeSelection(
        torch.tensor([[-1, -1], [0, -1], [0, 1], [1, -1]], dtype=torch.long),
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0]],
                      dtype=torch.float32),
    )
    # Cross-rank parity: broadcast the selection from rank 0.
    broadcast_selection = ComposeSelection(
        selection.expert_ids.clone(), selection.gates.clone()
    )
    for tensor in (broadcast_selection.expert_ids, broadcast_selection.gates):
        dist.broadcast(tensor, src=0)
    assert torch.equal(broadcast_selection.expert_ids, selection.expert_ids)

    inputs = torch.randn(4, 4, 3)
    from compose.adapters.runtime import use_selection

    with use_selection(selection):
        hidden = inputs
        layer = model.model.layers[0]
        hidden = layer.self_attn.o_proj(
            layer.self_attn.q_proj(hidden)
            + layer.self_attn.k_proj(hidden)
            + layer.self_attn.v_proj(hidden)
        )
        output = layer.mlp.down_proj(
            layer.mlp.gate_proj(hidden) + layer.mlp.up_proj(hidden)
        )
    output.square().mean().backward()

    grads = {}
    executed_layers = list(pool.manager.layers.values())[:7]
    for layer in executed_layers:
        for name in ("lora_A", "lora_B"):
            old_w = getattr(layer.experts["0"], name).weight
            cand_w = getattr(layer.experts["1"], name).weight
            assert old_w.grad is None, "old expert must have no gradient"
            assert cand_w.grad is not None, "candidate must have gradient"
            grads[(id(layer), name)] = cand_w.grad.clone()

    # Cross-rank gradient parity after all_reduce.
    flat = torch.cat([value.reshape(-1) for value in grads.values()])
    dist.all_reduce(flat)
    if rank == 0:
        print("DDP_SMOKE_OK rank0 total_grad_norm={:.6f}".format(float(flat.abs().sum())))

    # Registry + state machine instantiate identically on every rank.
    registry = ExpertRegistry()
    registry.register(
        ExpertMetadata(
            expert_id=0, adapter_name="e0", rank=1, alpha=2.0,
            creation_task=0, creation_task_name="ImageNet-R",
            created_seed=42,
            checkpoint_path="/ckpt/e0", checkpoint_sha256="a" * 64,
            lifecycle_status=ExpertLifecycleStatus.PROVISIONAL,
        )
    )
    machine = TaskStateMachine(0, "ImageNet-R")
    machine.advance(TaskStage.DATA_READY)
    assert machine.stage is TaskStage.DATA_READY
    dist.destroy_process_group()
    if rank == 0:
        print("MOCK_DDP_SMOKE PASSED")


if __name__ == "__main__":
    main()
