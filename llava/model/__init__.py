"""作用：实现 LLaVA/HiDe-LLaVA 模型加载、结构封装、多模态输入整理和权重转换工具。"""

from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
# from .language_model.llava_mpt import LlavaMPTForCausalLM, LlavaMPTConfig
# import os

# AVAILABLE_MODELS = {
#     "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
#     # "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
#     # "llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
#     # "llava_mixtral": "LlavaMixtralForCausalLM, LlavaMixtralConfig",
#     # "llava_qwen_moe": "LlavaQwenMoeForCausalLM, LlavaQwenMoeConfig",    
#     # Add other models as needed
# }

# for model_name, model_classes in AVAILABLE_MODELS.items():
#     try:
#         exec(f"from .language_model.{model_name} import {model_classes}")
#     except Exception as e:
#         print(f"Failed to import {model_name} from llava.language_model.{model_name}. Error: {e}")