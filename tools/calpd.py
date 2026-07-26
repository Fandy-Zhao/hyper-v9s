"""作用：离线把 expert 的高斯统计量映射到 Poincare ball，计算双曲距离矩阵并保存热力图，用于分析双曲路由空间。"""

import json
import numpy as np
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt

def map_to_poincare_ball(means, vars, scale_coeff=1.0):
    """
    作用：将欧氏空间中的 expert 高斯统计映射到 Poincare ball。
    
    均值方向决定双曲嵌入方向，总方差决定半径：方差越小越靠近边界，方差越大越靠近原点，从而把确定性作为双曲层级信息。
    """
    # 1. 计算欧氏方向 (Direction)
    # 加上 eps 防止零向量除法
    norms = torch.norm(means, dim=1, keepdim=True) + 1e-6
    directions = means / norms
    
    # 2. 计算双曲半径 (Hyperbolic Radius)
    # 逻辑：不确定性(方差)越小，越靠近边界(1.0)；不确定性越大，越靠近中心(0.0)
    # 计算每个 Expert 的总方差 (Trace of Covariance)
    total_variances = torch.sum(vars, dim=1, keepdim=True)
    
    # 自适应归一化：
    # 为了防止指数爆炸或全归0，我们用所有任务的平均方差来归一化 lambda
    # 这样可以保证映射后的半径分布比较均匀
    avg_variance = torch.mean(total_variances)
    lambda_factor = scale_coeff / (avg_variance + 1e-6)
    
    # 半径公式: r = exp(-lambda * var)
    # var=0 -> r=1; var=huge -> r=0
    radii = torch.exp(-lambda_factor * total_variances)
    
    # 3. 数值稳定性截断
    # 庞加莱距离在 r=1 处由于分母为0会无穷大，所以限制最大半径为 0.999
    radii = torch.clamp(radii, min=0.001, max=0.999)
    
    # 4. 组合得到双曲嵌入
    embeddings = directions * radii
    
    return embeddings

def compute_pairwise_poincare_dist(embeddings):
    """
    作用：计算 Poincare ball 中所有点的两两双曲距离。
    
    使用标准 Poincare 距离公式，并对分母和 arccosh 输入做数值裁剪，避免边界点带来的不稳定。
    """
    N, D = embeddings.shape
    dist_matrix = torch.zeros((N, N))
    
    # 为了利用 GPU/Tensor 加速，可以使用广播，但为了逻辑清晰，这里用双重循环
    # (数据量 N=8 很小，循环无性能瓶颈)
    for i in range(N):
        for j in range(i, N):
            u = embeddings[i]
            v = embeddings[j]
            
            # 欧氏距离平方 ||u-v||^2
            sq_dist_euclidean = torch.sum((u - v)**2)
            
            # 模长平方 ||u||^2
            sq_norm_u = torch.sum(u**2)
            sq_norm_v = torch.sum(v**2)
            
            # 庞加莱距离公式:
            # d = arccosh( 1 + 2 * ||u-v||^2 / ((1-||u||^2)(1-||v||^2)) )
            numerator = 2 * sq_dist_euclidean
            denominator = (1 - sq_norm_u) * (1 - sq_norm_v)
            
            # 防止分母过小
            denominator = torch.clamp(denominator, min=1e-7)
            
            arg = 1 + numerator / denominator
            
            # 防止 arccosh 的输入小于 1 (由于浮点误差)
            arg = torch.clamp(arg, min=1.0 + 1e-7)
            
            dist = torch.acosh(arg)
            
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
            
    return dist_matrix

def process_and_plot_hyperbolic(modality_name, raw_means, raw_vars, json_path):
    """
    作用：处理一个模态的 expert 统计量，映射到双曲空间并绘制距离热力图。
    
    该函数主要用于观察不同任务 expert 在双曲路由空间中的可分性。
    """
    print(f"\n{'='*20} 正在计算 {modality_name} 的庞加莱距离 (Hyperbolic) {'='*20}")
    
    num_experts = len(raw_means)
    if num_experts == 0:
        print("数据为空")
        return

    # --- 1. 数据解析 ---
    # 将 list 转为 tensor
    means_list = []
    vars_list = []
    
    for i in range(num_experts):
        try:
            m = raw_means[i][0]
            v = raw_vars[i][0]
            means_list.append(m)
            vars_list.append(v)
        except IndexError:
            print(f"Expert {i} 数据结构错误")
            return

    means_tensor = torch.tensor(means_list, dtype=torch.float32)
    vars_tensor = torch.tensor(vars_list, dtype=torch.float32)
    dim = means_tensor.shape[1]

    print(f"Expert 数量: {num_experts}, 维度: {dim}")

    # --- 2. 映射到庞加莱球 ---
    # scale_coeff 控制方差对半径的影响力度
    # 建议设为 1.0 或 2.0，越大则大方差的任务越会被吸入原点
    embeddings = map_to_poincare_ball(means_tensor, vars_tensor, scale_coeff=1.0)
    
    # 打印一下模长，验证映射逻辑是否符合预期
    # ImageNet (Text方差0) 的模长应该接近 1
    radii = torch.norm(embeddings, dim=1)
    print("--- 映射后的庞加莱半径 (Radius) ---")
    print("半径越接近 1 代表越特化(方差小)，越接近 0 代表越泛化(方差大)")
    for i, r in enumerate(radii):
        print(f"E{i}: {r:.4f}")

    # --- 3. 计算距离矩阵 ---
    dist_matrix = compute_pairwise_poincare_dist(embeddings)
    dist_np = dist_matrix.numpy()

    # --- 4. 打印矩阵 ---
    print(f"\n--- {modality_name} Poincaré Distance Matrix ---")
    print(f"{'':<6}", end="")
    for idx in range(num_experts):
        print(f"E{idx:<7}", end="")
    print()
    for i in range(num_experts):
        print(f"E{i:<5} ", end="")
        for j in range(num_experts):
            print(f"{dist_np[i, j]:>6.2f}  ", end="")
        print()

    # --- 5. 绘制热力图 ---
    plt.figure(figsize=(10, 8))
    # 使用 "Magma_r" 或 "YlOrRd" 配色，颜色越深距离越远
    sns.heatmap(dist_np, 
                xticklabels=[f"E{i}" for i in range(num_experts)], 
                yticklabels=[f"E{i}" for i in range(num_experts)], 
                annot=True, 
                fmt=".2f", 
                cmap="YlGnBu", 
                cbar_kws={'label': 'Poincaré Distance'})
    
    plt.title(f"{modality_name} Hyperbolic Distance (Poincaré Ball)")
    plt.tight_layout()
    
    base_name = os.path.splitext(json_path)[0]
    save_img_path = f"{base_name}_{modality_name.lower()}_poincare.png"
    plt.savefig(save_img_path)
    plt.close()
    print(f"热力图已保存: {save_img_path}")

def main_hyperbolic_analysis(json_path):
    """作用：读取 stats.json，并分别对图像和文本 expert 运行 Poincare 双曲距离分析。"""
    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        return

    print(f"读取文件: {json_path}")
    with open(json_path, 'r') as f:
        data_dict = json.load(f)

    # 处理 Image
    if 'image_mean' in data_dict:
        process_and_plot_hyperbolic(
            "Image", data_dict['image_mean'], data_dict['image_var'], json_path
        )
    
    # 处理 Text
    if 'text_mean' in data_dict:
        process_and_plot_hyperbolic(
            "Text", data_dict['text_mean'], data_dict['text_var'], json_path
        )

# --- 运行配置 ---
json_file_path = "/home/zhangyanqin/project/Hide/runs/checkpoints/HiDe/CoIN/Task8_llava_lora_ours/stats.json"

if os.path.exists(json_file_path):
    main_hyperbolic_analysis(json_file_path)
else:
    print("未找到指定文件，请检查路径。")