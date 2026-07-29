import subprocess
import sys
import unittest


class ImportIsolationTest(unittest.TestCase):
    def test_compose_import_does_not_load_hyper_modules(self):
        code = """
import sys
import compose.train.train_compose
assert not any(name == 'Hyper.peft' or name.startswith('Hyper.peft.') for name in sys.modules)
assert 'llava.model.llava_arch' not in sys.modules
assert 'llava.model.language_model.llava_llama' not in sys.modules
from llava import LlavaLlamaForCausalLM
assert LlavaLlamaForCausalLM.__name__ == 'LlavaLlamaForCausalLM'
"""
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_compose_model_has_no_hyper_state(self):
        from compose.model import ComposeLlavaConfig, ComposeLlavaForCausalLM

        config = ComposeLlavaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        model = ComposeLlavaForCausalLM(config)
        forbidden = {
            "cur_task",
            "expert_num",
            "image_anchors",
            "text_anchors",
            "image_mean",
            "image_var",
            "text_mean",
            "text_var",
            "instance_router",
            "expert_weight",
        }
        self.assertTrue(forbidden.isdisjoint(vars(model)))
