"""Top-level LLaVA exports.

The model export is lazy so utility modules such as ``llava.constants`` and
``llava.conversation`` do not initialize the Hyper-LLaVA model stack.
"""

__all__ = ["LlavaLlamaForCausalLM"]


def __getattr__(name):
    if name == "LlavaLlamaForCausalLM":
        from .model import LlavaLlamaForCausalLM

        return LlavaLlamaForCausalLM
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
