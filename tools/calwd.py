"""作用：离线读取 stats.json 中各 expert 的图像/文本高斯统计量，计算两两 2-Wasserstein 距离并保存热力图，用于分析任务分布差异。"""

import json
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_wasserstein_diagonal(mu1, var1, mu2, var2):
    """
    作用：计算两个对角高斯分布之间的 2-Wasserstein 距离。
    
    输入为两个分布的均值向量和方差向量，先将方差裁剪为非负，再使用标准差项计算均值差异与不确定性差异；返回值越大表示两个 expert 的统计分布越远。
    """
    # 1. 确保方差非负 (数值稳定性)
    var1 = np.maximum(var1, 0)
    var2 = np.maximum(var2, 0)
    
    # 2. 计算标准差 sigma
    sigma1 = np.sqrt(var1)
    sigma2 = np.sqrt(var2)
    
    # 3. 计算均值项 (Euclidean distance between means)
    term_mu = np.sum((mu1 - mu2)**2)
    
    # 4. 计算方差项 (Euclidean distance between sigmas)
    # 注意公式是 (sigma1 - sigma2)^2，不是 var1 - var2
    term_sigma = np.sum((sigma1 - sigma2)**2)
    
    # 5. 开根号得到 W2
    w2_distance = np.sqrt(term_mu + term_sigma)
    
    return w2_distance

def process_and_plot_wasserstein(modality_name, raw_means, raw_vars, json_path):
    """
    作用：处理一个模态的 expert 统计量并绘制 Wasserstein 距离热力图。
    
    该函数解析 stats.json 中形如 [expert][1][hidden_dim] 的 mean/var，构造 expert 两两距离矩阵，打印表格并把热力图保存到 stats 同目录。
    """
    print(f"\n{'='*20} 正在计算 {modality_name} 的 Wasserstein 距离 {'='*20}")
    
    num_experts = len(raw_means)
    if num_experts == 0:
        print(f"警告: {modality_name} 数据为空")
        return

    # --- 1. 数据解析 (适配 [8][1][768] 结构) ---
    expert_data = []
    dim = 0
    for i in range(num_experts):
        try:
            # 提取内部数据
            mu = np.array(raw_means[i][0])
            var = np.array(raw_vars[i][0])
            dim = mu.shape[0]
            expert_data.append({"mean": mu, "var": var})
        except IndexError:
            print(f"错误: {modality_name} Expert {i} 数据结构不符合预期")
            return

    print(f"检测到 {num_experts} 个 Expert，特征维度: {dim}")

    # --- 2. 计算距离矩阵 ---
    dist_matrix = np.zeros((num_experts, num_experts))
    labels = [f"E{i}" for i in range(num_experts)]

    for i in range(num_experts):
        for j in range(i, num_experts):
            if i == j:
                dist = 0.0
            else:
                dist = calculate_wasserstein_diagonal(
                    expert_data[i]["mean"], expert_data[i]["var"],
                    expert_data[j]["mean"], expert_data[j]["var"]
                )
            
            # 距离是对称的
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # --- 3. 打印控制台结果 ---
    print(f"--- {modality_name} Wasserstein Distance Matrix ---")
    print(f"{'':<6}", end="")
    for idx in range(num_experts):
        print(f"E{idx:<7}", end="")
    print()

    for i in range(num_experts):
        print(f"E{i:<5} ", end="")
        for j in range(num_experts):
            val = dist_matrix[i, j]
            # 距离越小越高亮 (表示越相似)
            # 这里简单打印一位小数，因为 Wasserstein 距离通常较大
            print(f"{val:>6.1f}  ", end="")
        print()

    # --- 4. 绘制热力图 ---
    plt.figure(figsize=(10, 8))
    
    # 颜色映射：使用 "Viridis" 或 "Blues"
    # 注意：Wasserstein 是距离，值越大颜色越深，表示差异越大
    # 如果想反过来(越红表示越近)，可以用 "dist_matrix.max() - dist_matrix" 绘图，或者反转 cmap
    sns.heatmap(dist_matrix, 
                xticklabels=labels, 
                yticklabels=labels, 
                annot=True, 
                fmt=".1f", # 保留1位小数
                cmap="YlGnBu", 
                cbar_kws={'label': 'Wasserstein Distance'})
    
    plt.title(f"{modality_name} pairwise Wasserstein Distance (dim={dim})")
    plt.tight_layout()
    
    # 保存图片
    base_name = os.path.splitext(json_path)[0]
    save_img_path = f"{base_name}_{modality_name.lower()}_wasserstein.png"
    plt.savefig(save_img_path)
    plt.close()
    print(f"热力图已保存: {save_img_path}")


def main_wasserstein_analysis(json_path):
    """
    作用：读取 stats.json 并分别触发图像模态和文本模态的 Wasserstein 分析。
    
    stats.json 需要包含 image_mean/image_var 或 text_mean/text_var；缺失的模态会被跳过。
    """
    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        return

    print(f"读取文件: {json_path}")
    with open(json_path, 'r') as f:
        data_dict = json.load(f)

    # 处理 Image
    if 'image_mean' in data_dict:
        process_and_plot_wasserstein(
            "Image", data_dict['image_mean'], data_dict['image_var'], json_path
        )
    
    # 处理 Text
    if 'text_mean' in data_dict:
        process_and_plot_wasserstein(
            "Text", data_dict['text_mean'], data_dict['text_var'], json_path
        )

# --- 运行配置 ---
# 请替换为你的实际路径
json_file_path = "/home/zhaozhuofan/hyper-llava/runs/checkpoints/HiDe/CoIN/Task8_llava_lora_ours/stats.json"

if os.path.exists(json_file_path):
    main_wasserstein_analysis(json_file_path)
else:
    print("未找到指定路径的文件，请检查路径配置。")
    # 生成测试数据以便验证代码逻辑
    # (此段逻辑仅在文件不存在时运行)
    print("生成测试数据演示...")
    dummy_path = "./dummy_stats.json"
    dim = 768
    dummy_data = {'image_mean': [], 'image_var': [], 'text_mean': [], 'text_var': []}
    for _ in range(8):
        dummy_data['image_mean'].append([np.random.randn(dim).tolist()])
        dummy_data['image_var'].append([np.abs(np.random.randn(dim)).tolist()])
        dummy_data['text_mean'].append([np.random.randn(dim).tolist()])
        dummy_data['text_var'].append([np.abs(np.random.randn(dim)).tolist()])
    with open(dummy_path, 'w') as f:
        json.dump(dummy_data, f)
    main_wasserstein_analysis(dummy_path)