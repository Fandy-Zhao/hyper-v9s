"""作用：提供 MPT 语言模型兼容实现，包括配置、注意力、模块结构、初始化和 HuggingFace 适配逻辑。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class SharedEmbedding(nn.Embedding):

    """作用：SharedEmbedding 类封装模型结构、配置或前向传播相关逻辑。"""
    def forward(self, input: Tensor, unembed: bool=False) -> Tensor:
        """
        作用：执行当前模块的前向传播。
        
        在训练模式下通常只使用当前任务 expert；在推理模式下会根据外部写入的 expert_weight 选择或融合对应 LoRA expert。
        """
        if unembed:
            return F.linear(input, self.weight)
        return super().forward(input)