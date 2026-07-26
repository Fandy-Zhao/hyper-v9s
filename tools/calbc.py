"""作用：基于 Log Bhattacharyya 系数分析 expert 分布重叠度，并离线计算图像/文本模态的平滑融合权重。"""

import json
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
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

def process_and_get_log_matrix(modality_name, raw_means, raw_vars, json_path):
    """
    作用：计算指定模态下所有 expert 的 LogBC 矩阵并保存重叠度热力图。
    
    矩阵用于后续估计每个 expert 与最近邻任务的距离，从而衡量该模态对任务区分的可靠程度。
    """
    print(f"\n{'='*20} 正在处理: {modality_name} {'='*20}")
    
    num_experts = len(raw_means)
    if num_experts == 0:
        print(f"警告: {modality_name} 数据为空。")
        return None, 0

    # --- 1. 数据解析 ---
    expert_data = []
    dim = 0
    for i in range(num_experts):
        try:
            mu = np.array(raw_means[i][0])  
            var = np.array(raw_vars[i][0])
            dim = mu.shape[0]
            expert_data.append({"mean": mu, "var": var})
        except IndexError:
            print(f"错误: {modality_name} 第 {i} 个元素结构错误")
            return None, 0

    # --- 2. 计算 Log 矩阵 ---
    # 注意：这里存储的是 Log 值 (负数)
    log_matrix = np.zeros((num_experts, num_experts))
    labels = [f"E{i}" for i in range(num_experts)]

    for i in range(num_experts):
        for j in range(i, num_experts):
            if i == j:
                log_bc = 0.0 # log(1.0) = 0
            else:
                log_bc = calculate_log_bc_diagonal(
                    expert_data[i]["mean"], expert_data[i]["var"],
                    expert_data[j]["mean"], expert_data[j]["var"]
                )
            
            log_matrix[i, j] = log_bc
            log_matrix[j, i] = log_bc

    # --- 3. 绘制热力图 (仅用于可视化，转换回 0-1) ---
    # 为了画图好看，我们做 exp 处理
    vis_matrix = np.exp(log_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(vis_matrix, xticklabels=labels, yticklabels=labels, 
                annot=True, fmt=".1e", cmap="YlGnBu", vmin=0, vmax=1)
    plt.title(f"{modality_name} Expert Overlap (Vis Only)")
    plt.tight_layout()
    base_name = os.path.splitext(json_path)[0]
    save_img_path = f"{base_name}_{modality_name.lower()}_heatmap.png"
    plt.savefig(save_img_path)
    plt.close()
    print(f"热力图已保存: {save_img_path}")

    return log_matrix, dim

def compute_and_plot_smooth_weights(img_log_matrix, txt_log_matrix, json_path, temperature=0.5):
    """
    作用：根据图像和文本的 LogBC 距离矩阵计算平滑模态融合权重。
    
    流程是先取每个 expert 到最近邻 expert 的距离，再做模态内归一化，最后对图像/文本距离做 temperature softmax，得到每个 expert 的 image/text 权重。
    """
    print(f"\n{'='*20} 计算平滑融合权重 (Temperature={temperature}) {'='*20}")
    num_experts = img_log_matrix.shape[0]
    
    # 1. 转换为 Bhattacharyya 距离 (正数)
    # D = -log_bc. 值越大，表示分得越开（越独特）
    dist_img = -img_log_matrix
    dist_txt = -txt_log_matrix
    
    # 将对角线(自己和自己)设为无限大，防止干扰求最小值
    np.fill_diagonal(dist_img, np.inf)
    np.fill_diagonal(dist_txt, np.inf)
    
    # 2. 计算最近邻距离 (Minimum Distance)
    # 含义：离我最近的那个捣乱鬼，离我有多远？
    # 距离越小，风险越高；距离越大，越安全。
    min_dist_img = np.min(dist_img, axis=1)
    min_dist_txt = np.min(dist_txt, axis=1)
    
    # 3. 模态内归一化 (Intra-modal Normalization)
    # 关键步骤：解决 ImageNet 这种极端情况。
    # 我们除以该模态的平均距离，将数值拉伸到相对尺度 (Relative Scale)
    # 加 1e-6 防止除以 0
    mean_img = np.mean(min_dist_img[min_dist_img != np.inf])
    mean_txt = np.mean(min_dist_txt[min_dist_txt != np.inf])
    
    norm_dist_img = min_dist_img / (mean_img + 1e-6)
    norm_dist_txt = min_dist_txt / (mean_txt + 1e-6)
    
    print(f"平均距离参考 -> Image: {mean_img:.2f}, Text: {mean_txt:.2f}")
    
    # 4. Softmax 计算权重
    final_w_img = []
    final_w_txt = []
    
    print(f"\n{'Expert':<6} {'Img_Dist':<10} {'Txt_Dist':<10} | {'W_Image':<10} {'W_Text':<10}")
    print("-" * 60)
    
    for i in range(num_experts):
        # 构造 Logits
        logits = torch.tensor([norm_dist_img[i], norm_dist_txt[i]], dtype=torch.float32)
        # 除以温度系数，T 越小差异越明显
        logits = logits / temperature
        
        weights = F.softmax(logits, dim=0)
        
        w_i = weights[0].item()
        w_t = weights[1].item()
        
        final_w_img.append(w_i)
        final_w_txt.append(w_t)
        
        # 打印详细信息
        print(f"E{i:<5} {min_dist_img[i]:<10.1f} {min_dist_txt[i]:<10.1f} | \033[92m{w_i:.4f}\033[0m     \033[94m{w_t:.4f}\033[0m")

    # 5. 绘制权重对比图
    plt.figure(figsize=(12, 6))
    x = np.arange(num_experts)
    width = 0.35
    
    plt.bar(x - width/2, final_w_img, width, label='Image Weight', color='skyblue', alpha=0.9)
    plt.bar(x + width/2, final_w_txt, width, label='Text Weight', color='salmon', alpha=0.9)
    
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Experts')
    plt.ylabel('Computed Fusion Weight')
    plt.title(f'Smoothed Modality Weights (Temp={temperature})\nBased on Relative Bhattacharyya Distance')
    plt.xticks(x, [f"E{i}" for i in x])
    plt.legend()
    plt.ylim(0, 1.05)
    
    base_name = os.path.splitext(json_path)[0]
    save_path = f"{base_name}_smooth_weights.png"
    plt.savefig(save_path)
    plt.close()
    print(f"\n权重分布图已保存: {save_path}")

def main_analysis(json_path):
    """作用：读取 stats.json，计算图像/文本 LogBC 矩阵，并生成离线平滑融合权重图。"""
    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        return

    print(f"读取文件: {json_path}")
    with open(json_path, 'r') as f:
        data_dict = json.load(f)

    # 1. 获取 Image 的 Log 矩阵
    if 'image_mean' in data_dict:
        img_log_mat, _ = process_and_get_log_matrix(
            "Image", data_dict['image_mean'], data_dict['image_var'], json_path
        )
    else:
        print("缺少 Image 数据")
        return

    # 2. 获取 Text 的 Log 矩阵
    if 'text_mean' in data_dict:
        txt_log_mat, _ = process_and_get_log_matrix(
            "Text", data_dict['text_mean'], data_dict['text_var'], json_path
        )
    else:
        print("缺少 Text 数据")
        return
        
    # 3. 执行平滑权重计算
    if img_log_mat is not None and txt_log_mat is not None:
        # temperature 可以调整：
        # 0.1 (非常锐利，非0即1) 
        # 1.0 (标准) 
        # 5.0 (非常平滑，接近0.5)
        compute_and_plot_smooth_weights(img_log_mat, txt_log_mat, json_path, temperature=0.5)

# --- 运行 ---
# 使用你指定的路径
json_file_path = "/home/zhangyanqin/project/Hyper-LlaVA/runs/checkpoints/Hide/UCIT/Task8_llava_lora_ours/stats.json"

# 为了本地运行不报错，加一个简单的路径检查，如果不存在则使用上一级目录模拟
if not os.path.exists(json_file_path):
    # 这里只是为了代码健壮性，实际运行时你应该有这个文件
    print(f"注意: {json_file_path} 不存在，请检查路径。")
else:
    main_analysis(json_file_path)



# Expert Img_Dist   Txt_Dist   | W_Image    W_Text    
# ------------------------------------------------------------
# E0     57.0       36.3       | 0.9992     0.0008
# E1     7.3        36.3       | 0.6529     0.3471
# E2     6.4        1003.5     | 0.0003     0.9997
# E3     0.8        4.4        | 0.5172     0.4828
# E4     22.7       21.5       | 0.9430     0.0570
# E5     1.8        92.9       | 0.3556     0.6444
# E6     0.8        4.4        | 0.5172     0.4828
# E7     24.4       574.4      | 0.1229     0.8771