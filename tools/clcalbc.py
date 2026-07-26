"""作用：按持续学习 stage 逐步模拟已学习 expert 集合，计算每个阶段的自适应图像/文本融合权重。"""

import json
import numpy as np
import os
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
    
    # 1. 均值差异项
    term1 = -0.25 * ((mu1 - mu2)**2) / (var1 + var2)
    
    # 2. 方差差异项
    numerator = 2 * np.sqrt(var1 * var2)
    denominator = var1 + var2
    term2 = 0.5 * np.log(numerator / denominator + eps)
    
    # 3. 求和
    log_bc = np.sum(term1 + term2)
    
    return log_bc

def get_full_log_matrix(modality_name, raw_means, raw_vars):
    """作用：预先计算一个模态完整的 expert LogBC 矩阵，供持续学习阶段切片复用。"""
    num_experts = len(raw_means)
    if num_experts == 0:
        print(f"警告: {modality_name} 数据为空。")
        return None

    # --- 1. 数据解析 ---
    expert_data = []
    for i in range(num_experts):
        try:
            mu = np.array(raw_means[i][0])  
            var = np.array(raw_vars[i][0])
            expert_data.append({"mean": mu, "var": var})
        except IndexError:
            print(f"错误: {modality_name} 第 {i} 个元素结构错误")
            return None

    # --- 2. 计算 Log 矩阵 ---
    log_matrix = np.zeros((num_experts, num_experts))

    for i in range(num_experts):
        for j in range(i, num_experts):
            if i == j:
                log_bc = 0.0 
            else:
                log_bc = calculate_log_bc_diagonal(
                    expert_data[i]["mean"], expert_data[i]["var"],
                    expert_data[j]["mean"], expert_data[j]["var"]
                )
            
            log_matrix[i, j] = log_bc
            log_matrix[j, i] = log_bc

    return log_matrix

def compute_cl_weights(full_img_mat, full_txt_mat, temperature=0.5):
    """
    作用：按持续学习阶段逐步计算自适应模态融合权重。
    
    每个 stage 只使用已经学过的 expert；stage=1 默认 0.5/0.5，stage=2 使用 virtual anchor 做联合归一化，stage>=3 使用模态内平均距离归一化。
    """
    total_tasks = full_img_mat.shape[0]
    
    # 存储所有阶段的结果
    cl_results = {}

    print(f"\n{'='*20} 开始持续学习权重计算 (Temperature={temperature}) {'='*20}")

    # === 增量循环：从第1个任务学到第N个任务 ===
    for stage in range(1, total_tasks + 1):
        print(f"\n>>> Stage {stage}: 已学习任务 [E0 ... E{stage-1}]")
        
        # 1. 切片：只获取当前阶段已知的任务数据
        # 形状变为 (stage, stage)
        curr_img_mat = full_img_mat[:stage, :stage]
        curr_txt_mat = full_txt_mat[:stage, :stage]
        
        # 特殊情况：如果是第1个阶段，只有一个任务，不存在“距离”，默认权重 0.5/0.5
        if stage == 1:
            print(f"    E0    | W_Image: 0.5000   W_Text: 0.5000 (初始阶段)")
            cl_results[stage] = [(0.5, 0.5)]
            continue

        # 2. 转换为距离 (Dist = -LogBC)
        dist_img = -curr_img_mat
        dist_txt = -curr_txt_mat
        
        # 对角线设为无穷大
        np.fill_diagonal(dist_img, np.inf)
        np.fill_diagonal(dist_txt, np.inf)
        
        # 3. 计算当前集合内的最近邻距离
        # 注意：这里 axis=1 是在 (stage, stage) 矩阵上操作
        min_dist_img = np.min(dist_img, axis=1)
        min_dist_txt = np.min(dist_txt, axis=1)

        # ==============================================================================
        # 4. 归一化策略分支
        # ==============================================================================
        if stage == 2:
            # --- Stage 2: 虚拟锚点 (Virtual Anchor) 策略 ---
            # 问题：ScienceQA(E0) 和 TextVQA(E1) 互为最近邻，min_dist_img 均值等于自身，
            #       导致 norm_dist 恒为 1.0，无法体现模态差异。
            # 方案：不再分别归一化。而是建立一个"虚拟锚点"，其距离定义为两个模态距离的几何平均值。
            #       这相当于把两个模态拉到同一个坐标系下比较。
            
            # 计算两个模态的平均分离程度 (Average Separation)
            # 因为是对称矩阵，取非对角线元素均可
            avg_sep_img = np.mean(min_dist_img)
            avg_sep_txt = np.mean(min_dist_txt)
            
            print(f"    [Anchor Info] Image分离度: {avg_sep_img:.2f}, Text分离度: {avg_sep_txt:.2f}")
            
            # 构造虚拟锚点距离 (Global Anchor)
            # 含义：这是一个"基准单位"。
            # 如果某模态的分离度 > 基准，说明该模态区分能力强；反之则弱。
            anchor_scale = np.sqrt(avg_sep_img * avg_sep_txt + 1e-6)
            
            # 联合归一化 (Joint Normalization)
            # 关键点：Image 和 Text 都除以同一个 anchor_scale
            norm_dist_img = min_dist_img / anchor_scale
            norm_dist_txt = min_dist_txt / anchor_scale
            
            # 此时：
            # 如果 Image分离度(20) > Text分离度(5)，则 Anchor=10
            # Norm_Img = 2.0, Norm_Txt = 0.5 -> Image 权重胜出！
            # 这能救 ScienceQA，因为它的图表特征通常比 OCR 文本特征更具区分度（相对于 TextVQA）。
            
            current_T = temperature

        else:
            # --- Stage > 2: 常规模态内归一化 ---
            # 任务多了，各自模态内部的统计规律稳定了，可以各自归一化。
            mean_img = np.mean(min_dist_img[min_dist_img != np.inf])
            mean_txt = np.mean(min_dist_txt[min_dist_txt != np.inf])
            
            norm_dist_img = min_dist_img / (mean_img + 1e-6)
            norm_dist_txt = min_dist_txt / (mean_txt + 1e-6)
            
            current_T = temperature
        
        # 5. Softmax 计算权重
        stage_weights = []
        print(f"    {'Task':<5} {'Img_Dist':<10} {'Txt_Dist':<10} | {'W_Image':<10} {'W_Text':<10}")
        print("    " + "-" * 55)
        
        for i in range(stage):
            logits = torch.tensor([norm_dist_img[i], norm_dist_txt[i]], dtype=torch.float32)
            logits = logits / temperature
            weights = F.softmax(logits, dim=0)
            
            w_i = weights[0].item()
            w_t = weights[1].item()
            stage_weights.append((w_i, w_t))
            
            # 打印当前阶段的计算结果
            print(f"    E{i:<4} {min_dist_img[i]:<10.1f} {min_dist_txt[i]:<10.1f} | \033[92m{w_i:.4f}\033[0m     \033[94m{w_t:.4f}\033[0m")
        
        cl_results[stage] = stage_weights

    return cl_results

def main_cl_analysis(json_path):
    """作用：读取 stats.json，构建完整图像/文本 LogBC 矩阵，并输出每个持续学习阶段的权重结果。"""
    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        return

    print(f"读取文件: {json_path}")
    with open(json_path, 'r') as f:
        data_dict = json.load(f)

    # 1. 预先计算全量 Log 矩阵
    if 'image_mean' in data_dict and 'text_mean' in data_dict:
        full_img_log = get_full_log_matrix("Image", data_dict['image_mean'], data_dict['image_var'])
        full_txt_log = get_full_log_matrix("Text", data_dict['text_mean'], data_dict['text_var'])
    else:
        print("数据不完整")
        return

    # 2. 执行持续学习权重计算
    if full_img_log is not None and full_txt_log is not None:
        # 温度系数 0.5 能较好地区分特征差异
        results = compute_cl_weights(full_img_log, full_txt_log, temperature=0.5)
        
        # 这里 results 字典存储了每个阶段的数据，格式为:
        # {
        #    1: [(w_i, w_t)],
        #    2: [(w_i, w_t), (w_i, w_t)],
        #    ...
        # }
        return results

# --- 运行 ---
json_file_path = "/home/zhangyanqin/project/Hyper-LlaVA/runs/checkpoints/HiDe/UCIT_IFRCAV/Task6_llava_lora_ours/stats.json"

if os.path.exists(json_file_path):
    cl_weights = main_cl_analysis(json_file_path)
else:
    print(f"注意: {json_file_path} 不存在")

# python clcalbc.py
# /home/zhangyanqin/anaconda3/envs/hide1/lib/python3.10/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
#   import pynvml  # type: ignore[import]
# 读取文件: /home/zhangyanqin/project/Hyper-LlaVA/runs/checkpoints/HiDe/UCIT_IFRCAV/Task6_llava_lora_ours/stats.json

# ==================== 开始持续学习权重计算 (Temperature=0.5) ====================

# >>> Stage 1: 已学习任务 [E0 ... E0]
#     E0    | W_Image: 0.5000   W_Text: 0.5000 (初始阶段)

# >>> Stage 2: 已学习任务 [E0 ... E1]
#     [Anchor Info] Image分离度: 52.95, Text分离度: 1686.43
#     Task  Img_Dist   Txt_Dist   | W_Image    W_Text    
#     -------------------------------------------------------
#     E0    52.9       1686.4     | 0.0000     1.0000
#     E1    52.9       1686.4     | 0.0000     1.0000

# >>> Stage 3: 已学习任务 [E0 ... E2]
#     Task  Img_Dist   Txt_Dist   | W_Image    W_Text    
#     -------------------------------------------------------
#     E0    50.1       1329.4     | 0.8199     0.1801
#     E1    19.8       1686.4     | 0.2681     0.7319
#     E2    19.8       1329.4     | 0.3749     0.6251

# >>> Stage 4: 已学习任务 [E0 ... E3]
#     Task  Img_Dist   Txt_Dist   | W_Image    W_Text    
#     -------------------------------------------------------
#     E0    50.1       81.6       | 0.6455     0.3545
#     E1    19.8       1686.4     | 0.0193     0.9807
#     E2    19.8       1329.4     | 0.0462     0.9538
#     E3    408.3      81.6       | 0.9983     0.0017

# >>> Stage 5: 已学习任务 [E0 ... E4]
#     Task  Img_Dist   Txt_Dist   | W_Image    W_Text    
#     -------------------------------------------------------
#     E0    50.1       65.5       | 0.6578     0.3422
#     E1    19.8       1686.4     | 0.0075     0.9925
#     E2    19.8       1329.4     | 0.0223     0.9777
#     E3    408.3      81.6       | 0.9988     0.0012
#     E4    86.8       65.5       | 0.7827     0.2173

# >>> Stage 6: 已学习任务 [E0 ... E5]
#     Task  Img_Dist   Txt_Dist   | W_Image    W_Text    
#     -------------------------------------------------------
#     E0    50.1       65.5       | 0.6922     0.3078
#     E1    19.8       1686.4     | 0.0214     0.9786
#     E2    19.8       1329.4     | 0.0506     0.9494
#     E3    408.3      81.6       | 0.9996     0.0004
#     E4    86.8       65.5       | 0.8212     0.1788
#     E5    32.4       1582.5     | 0.0350     0.9650


# UCIT 标准：
                # [1.0,0.0,0.0,0.0,0.0,0.0],
                # [0.5,0.5,0.0,0.0,0.0,0.0],
                # [0.3396,0.8026,0.3236,0.0,0.0,0.0],
                # [0.0928,0.9635,0.0604,0.8521,0.0,0.0],
                # [0.0199,0.7544,0.0114,0.6316,0.9988,0.0],
                # [0.0000,0.7534,0.6129,0.6056,0.9996,0.5437]
# UCIT AIRFCV：
                # [1.0,0.0,0.0,0.0,0.0,0.0],
                # [0.5,0.5,0.0,0.0,0.0,0.0],
                # [0.9244,0.7966,0.0204,0.0,0.0,0.0],
                # [0.9777,0.9017,0.0651,0.0345,0.0,0.0],
                # [0.7756,0.6615,0.0187,0.0084,0.9989,0.0],
                # [0.7603,0.6236,0.0000,0.5864,0.9994,0.6502]
# UCIT IFRCAV:
                # [1.0,0.0,0.0,0.0,0.0,0.0],
                # [0.5,0.5,0.0,0.0,0.0,0.0],
                # [0.8199,0.2681,0.3749,0.0,0.0,0.0],
                # [0.6455,0.0193,0.0462,0.9983,0.0,0.0],
                # [0.6578,0.0075,0.0223,0.9988,0.7827,0.0],
                # [0.6922,0.0214,0.0506,0.9996,0.8212,0.0350]
