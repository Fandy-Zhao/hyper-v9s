import json
import tempfile
import unittest
import warnings
from pathlib import Path

from compose.model import load_compose_config


class ConfigConversionTest(unittest.TestCase):
    def test_llava_config_is_explicitly_converted_without_warning(self):
        source = {
            "model_type": "llava",
            "architectures": ["LlavaLlamaForCausalLM"],
            "vocab_size": 32001,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(source), encoding="utf-8")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                config = load_compose_config(directory)
        self.assertEqual(caught, [])
        self.assertEqual(config.model_type, "compose_llava")
        self.assertEqual(config.architectures, ["ComposeLlavaForCausalLM"])
        for field in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "rope_theta",
        ):
            self.assertEqual(getattr(config, field), source[field])
