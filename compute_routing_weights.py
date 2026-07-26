"""作用：训练结束时被 train_MOE.py 调用，根据当前阶段各 expert 的图像/文本高斯统计量生成 adaptive_w_img 路由融合权重。"""

import numpy as np
import torch
import torch.nn.functional as F

def calculate_log_bc_diagonal(mu1, var1, mu2, var2):
    """
    作用：计算两个对角高斯分布的 Log Bhattacharyya Coefficient。
    
    返回值通常小于等于 0，越接近 0 表示两个分布重叠越高；在路由权重计算中会取负数作为分布距离。
    """
    eps = 1e-12
    var1 = np.maximum(var1, eps)
    var2 = np.maximum(var2, eps)
    
    # 1. Mean difference term
    term1 = -0.25 * ((mu1 - mu2)**2) / (var1 + var2)
    
    # 2. Variance difference term
    numerator = 2 * np.sqrt(var1 * var2)
    denominator = var1 + var2
    term2 = 0.5 * np.log(numerator / denominator + eps)
    
    # 3. Sum over dimensions
    log_bc = np.sum(term1 + term2)
    
    return log_bc

def compute_current_stage_weights(expert_data_list, temperature=0.5):
    """
    作用：计算当前训练阶段应保存到 stats.json 的 adaptive_w_img。
    
    输入是已学习 expert 的图像/文本均值和方差列表；输出每个 expert 的图像模态权重，文本权重在推理时由 1 - w_img 得到。该函数是训练流程实际调用的路由权重计算入口。
    """
    stage = len(expert_data_list)
    
    if stage == 0:
        return []
    if stage == 1:
        # Default to 0.5 for the first task
        return [0.5]
        
    # 1. Compute Distance Matrices (-LogBC) for the current stage
    dist_img = np.zeros((stage, stage))
    dist_txt = np.zeros((stage, stage))
    
    for i in range(stage):
        for j in range(stage):
            if i == j:
                dist_img[i, j] = np.inf
                dist_txt[i, j] = np.inf
            else:
                bc_img = calculate_log_bc_diagonal(
                    expert_data_list[i]['img_mean'], expert_data_list[i]['img_var'],
                    expert_data_list[j]['img_mean'], expert_data_list[j]['img_var']
                )
                bc_txt = calculate_log_bc_diagonal(
                    expert_data_list[i]['txt_mean'], expert_data_list[i]['txt_var'],
                    expert_data_list[j]['txt_mean'], expert_data_list[j]['txt_var']
                )
                dist_img[i, j] = -bc_img
                dist_txt[i, j] = -bc_txt
                
    # 2. Minimum distance to any other task (Separability)
    min_dist_img = np.min(dist_img, axis=1)
    min_dist_txt = np.min(dist_txt, axis=1)
    
    # 3. Normalization Strategy
    if stage == 2:
        # Virtual Anchor Strategy for Stage 2
        avg_sep_img = np.mean(min_dist_img)
        avg_sep_txt = np.mean(min_dist_txt)
        
        anchor_scale = np.sqrt(avg_sep_img * avg_sep_txt + 1e-6)
        
        norm_dist_img = min_dist_img / anchor_scale
        norm_dist_txt = min_dist_txt / anchor_scale
        current_T = temperature
    else:
        # Standard Intra-modality Normalization for Stage >= 3
        mean_img = np.mean(min_dist_img[min_dist_img != np.inf])
        mean_txt = np.mean(min_dist_txt[min_dist_txt != np.inf])
        
        norm_dist_img = min_dist_img / (mean_img + 1e-6)
        norm_dist_txt = min_dist_txt / (mean_txt + 1e-6)
        current_T = temperature

    # 4. Softmax Weighting
    w_img_list = []
    for i in range(stage):
        logits = torch.tensor([norm_dist_img[i], norm_dist_txt[i]], dtype=torch.float32)
        logits = logits / current_T
        weights = F.softmax(logits, dim=0)
        w_img_list.append(weights[0].item())
        
    return w_img_list