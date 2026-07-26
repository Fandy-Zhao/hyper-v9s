"""作用：实现 LLaVA/HiDe-LLaVA 的训练入口、数据预处理、Trainer 扩展或注意力加速补丁。"""

# Make it more memory efficient by monkey patching the LLaMA model with xformers attention.

# Need to call this before importing transformers.
from llava.train.llama_xformers_attn_monkey_patch import (
    replace_llama_attn_with_xformers_attn,
)

replace_llama_attn_with_xformers_attn()

from llava.train.train import train

if __name__ == "__main__":
    train()
