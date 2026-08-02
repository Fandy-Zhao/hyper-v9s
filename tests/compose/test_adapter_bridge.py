import unittest

import torch
import torch.nn as nn

from compose.lora import AdapterBridge


class Side(nn.Module):
    def __init__(self, label, inputs, outputs, count=2):
        super().__init__()
        setattr(self, label, nn.ModuleList([nn.Linear(inputs, outputs, bias=False) for _ in range(count)]))


class FakeHyperLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(3, 3, bias=False)
        self.lora_A = nn.ModuleDict({"default": Side("loraA", 3, 2)})
        self.lora_B = nn.ModuleDict({"default": Side("loraB", 2, 3)})
        self.active_adapter = "default"
        self.cur_task = 0
        self.disable_adapters = False
        with torch.no_grad():
            self.base.weight.copy_(torch.eye(3))
            for parameter in self.lora_A.parameters():
                parameter.fill_(0.5)
            for parameter in self.lora_B.parameters():
                parameter.fill_(0.25)

    def forward(self, inputs):
        result = self.base(inputs)
        if self.disable_adapters:
            return result
        value = self.lora_A["default"].loraA[self.cur_task](inputs)
        return result + self.lora_B["default"].loraB[self.cur_task](value)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = FakeHyperLayer()

    def forward(self, inputs):
        return self.proj(inputs)


class AdapterBridgeTest(unittest.TestCase):
    def test_empty_frozen_trainable_and_restore_cases(self):
        model, inputs = FakeModel(), torch.ones(2, 3)
        bridge = AdapterBridge(model)
        initial = bridge.snapshot_runtime_state()

        bridge.set_active_experts([])
        bridge.freeze_all_experts()
        torch.testing.assert_close(model(inputs), model.proj.base(inputs))
        self.assertTrue(all(item["all_frozen"] for item in bridge.verify_grad_flags()["experts"].values()))

        bridge.set_active_experts([0])
        bridge.set_trainable_experts([])
        model(inputs).sum().backward()
        self.assertTrue(all(parameter.grad is None for _, parameter in bridge._expert_parameters[0]))

        model.zero_grad(set_to_none=True)
        bridge.set_trainable_experts([0])
        model(inputs).sum().backward()
        self.assertTrue(any(parameter.grad is not None for _, parameter in bridge._expert_parameters[0]))
        self.assertTrue(all(parameter.grad is None for _, parameter in bridge._expert_parameters[1]))

        bridge.restore_runtime_state(initial)
        self.assertEqual(bridge.get_active_experts(), ())

    def test_multi_expert_state_is_recorded_but_forward_is_blocked(self):
        model = FakeModel()
        bridge = AdapterBridge(model)
        bridge.set_active_experts([0, 1])
        bridge.set_trainable_experts([1])
        self.assertEqual(bridge.get_active_experts(), (0, 1))
        self.assertEqual(bridge.get_trainable_experts(), (1,))
        with self.assertRaisesRegex(NotImplementedError, "Stage 02"):
            model(torch.ones(1, 3))


if __name__ == "__main__":
    unittest.main()
