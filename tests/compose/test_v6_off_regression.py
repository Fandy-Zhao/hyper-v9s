import unittest

import torch

from compose.lora import AdapterBridge
from test_adapter_bridge import FakeModel


class V6OffRegressionTest(unittest.TestCase):
    def test_constructing_bridge_does_not_change_existing_forward(self):
        torch.manual_seed(42)
        model = FakeModel()
        inputs = torch.randn(2, 3)
        expected = model(inputs).detach()
        bridge = AdapterBridge(model)
        actual = model(inputs).detach()
        bridge.close()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
