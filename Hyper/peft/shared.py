# -*- encoding: utf-8 -*-
"""作用：提供 Hyper/PEFT 模型封装、自动加载、映射和共享适配逻辑。"""

# here put the import lib
import torch.nn as nn
from .utils import PeftConfig


class Gate(nn.Module):
    """Gate"""
    def __init__(self, peft_config: PeftConfig, adapter_name="default"):

        """作用：初始化对象状态、保存配置参数，并构建后续方法需要使用的成员变量。"""
        super().__init__()

        self.expert_num = peft_config.expert_num
        self.te_dim = peft_config.task_embedding_dim

        #self.lora_task_embedding = nn.Embedding(self.task_num+1, self.te_dim)# 使用embedding来代替线性层
        self.GateL = nn.Linear(self.te_dim, self.expert_num, bias=False)
        self.act = nn.Softmax(dim=0)    # 第0维为batch size
    
    def forward(self, task_em):

        #task_em = self.lora_task_embedding(x)
        """
        作用：执行当前模块的前向传播。
        
        在训练模式下通常只使用当前任务 expert；在推理模式下会根据外部写入的 expert_weight 选择或融合对应 LoRA expert。
        """
        y = self.GateL(task_em)
        y = self.act(y)

        return y