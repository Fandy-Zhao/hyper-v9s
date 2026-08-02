"""Manual L0 check against the real HyperMOELoraLinear implementation."""

import json

import torch
import torch.nn as nn

from Hyper.peft.tuners.clitmoelora import HyperMOELoraLinear
from compose.lora import AdapterBridge


class ActualHyperTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = HyperMOELoraLinear(
            "default",
            4,
            4,
            r=4,
            lora_alpha=8,
            expert_num=2,
            cur_task=0,
            task_embedding_dim=4,
            train_signal=True,
            layer=0,
            expert_weight=[1.0, 0.0],
        )


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActualHyperTiny().to(device)
inputs = torch.randn(2, 4, device=device)
baseline = model.proj(inputs).detach()
bridge = AdapterBridge(model, verify_ddp=False)
unchanged = model.proj(inputs).detach()
torch.testing.assert_close(unchanged, baseline, rtol=0, atol=0)
bridge.set_active_experts([0])
bridge.set_trainable_experts([0])
loss = model.proj(inputs).sum()
loss.backward()
flags = bridge.verify_grad_flags()
assert flags["experts"]["0"]["all_trainable"]
assert flags["experts"]["1"]["all_frozen"]
assert any(parameter.grad is not None for _, parameter in bridge._expert_parameters[0])
assert all(parameter.grad is None for _, parameter in bridge._expert_parameters[1])
print(json.dumps({"status": "PASSED", "device": str(device), "max_abs_diff": float((unchanged - baseline).abs().max()), "flags": flags}, sort_keys=True))
