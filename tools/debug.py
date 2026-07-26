"""作用：加载 adapter checkpoint 并打印部分参数 key，用于快速检查保存的 LoRA/adapter 权重内容。"""

import torch
checkpoint = torch.load('runs/checkpoints/HiDe/CoIN/Task1_llava_lora_ours/adapter_model.bin', map_location='cpu')
print('检查点键:', list(checkpoint.keys())[:10])  # 显示前10个键