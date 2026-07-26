"""作用：读取保存的图像/文本特征，绘制 PCA、高斯边缘分布和相关性图，验证对角高斯建模假设。"""

import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA

def load_features_for_task(feature_dir, task_id, mod='img'):
    """作用：读取指定任务和模态保存的 .pt 特征文件，并合并为一个 numpy 特征矩阵。"""
    files = glob.glob(os.path.join(feature_dir, f"task_{task_id}_{mod}_*.pt"))
    features =[]
    for f in files:
        features.append(torch.load(f).float().numpy()) # 转回 float32 用于分析
    if not features:
        return None
    return np.concatenate(features, axis=0)

def plot_pca_with_ellipse(features, title, save_path):
    """作用：将高维特征 PCA 到二维，绘制采样散点和拟合高斯置信椭圆。"""
    pca = PCA(n_components=2)
    feats_2d = pca.fit_transform(features)
    
    mean_2d = np.mean(feats_2d, axis=0)
    cov_2d = np.cov(feats_2d, rowvar=False)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 1. 画真实的散点 (取样画，防止点太多看不清)
    sample_size = min(2000, feats_2d.shape[0])
    idx = np.random.choice(feats_2d.shape[0], sample_size, replace=False)
    ax.scatter(feats_2d[idx, 0], feats_2d[idx, 1], alpha=0.3, s=10, label='Actual Features', color='#4C72B0')
    
    # 2. 画拟合的高斯置信椭圆 (2-sigma, 约涵盖 95% 数据)
    pearson = cov_2d[0, 1] / np.sqrt(cov_2d[0, 0] * cov_2d[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    lambda_, v = np.linalg.eig(cov_2d)
    lambda_ = np.sqrt(lambda_)
    angle = np.rad2deg(np.arccos(v[0, 0]))
    
    ell = Ellipse(xy=mean_2d, width=lambda_[0]*2*2, height=lambda_[1]*2*2, 
                  angle=angle, edgecolor='#C44E52', fc='None', lw=2, label='Fitted 2$\sigma$ Gaussian')
    ax.add_patch(ell)
    
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def normality_test_and_plot(features, title, save_path):
    """作用：选择方差最大的若干维度，绘制真实边缘分布与高斯拟合曲线，用于检查正态性假设。"""
    # 找到方差最大的 3 个维度
    variances = np.var(features, axis=0)
    top3_dims = np.argsort(variances)[-3:]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{title} - Marginal Distributions of Top 3 Variance Dimensions", fontsize=14)
    
    for i, dim_idx in enumerate(top3_dims):
        data1d = features[:, dim_idx]
        mu, std = stats.norm.fit(data1d)
        
        # 画真实直方图
        sns.histplot(data1d, bins=50, stat='density', alpha=0.5, color='gray', ax=axes[i], label='Actual Dist')
        
        # 画理论高斯曲线
        xmin, xmax = axes[i].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        axes[i].plot(x, p, 'k', linewidth=2, label=f'Gaussian Fit\n$\mu$={mu:.2f}, $\sigma$={std:.2f}')
        
        # 计算 Skewness 和 Kurtosis (偏度和峰度，越接近0越像正态分布)
        skew = stats.skew(data1d)
        kurt = stats.kurtosis(data1d)
        axes[i].set_title(f"Dim {dim_idx} (Skew: {skew:.2f}, Kurt: {kurt:.2f})", fontsize=10)
        axes[i].legend(fontsize=8)
        
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def independence_test_and_plot(features, title, save_path):
    """作用：计算特征维度间 Pearson 相关性，绘制相关矩阵和非对角相关系数分布，用于检查对角协方差假设。"""
    print(f"Running independence test for {title}...")
    
    # 1. 计算 Pearson 相关系数矩阵 R (768 x 768)
    # np.corrcoef 默认按行(rowvar=True)计算变量，我们需要按列(特征维度)计算
    # 返回的 R 矩阵中，R[i, j] 是第 i 维和第 j 维的相关系数 [-1, 1]
    # 对角线元素 R[i, i] 恒为 1.0
    corr_matrix = np.corrcoef(features, rowvar=False)
    
    # --- 准备画图 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{title} - Feature Independence Analysis", fontsize=16, fontweight='bold', y=1.05)
    
    # ==========================================
    # 子图 1: 局部相关系数热力图 (方案 A)
    # ==========================================
    # 768维全画出来会是一团糊，我们截取前 100 维 (或随机 100 维) 来展示清晰的对角线
    display_dim = min(100, corr_matrix.shape[0])
    sub_corr = corr_matrix[:display_dim, :display_dim]
    
    # 使用发散型调色板，0 为白色，正相关偏红，负相关偏蓝
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    sns.heatmap(sub_corr, ax=axes[0], cmap=cmap, center=0, 
                vmin=-0.5, vmax=0.5, # 限制色域以突出微小的非对角相关
                square=True, cbar_kws={"shrink": .8, "label": "Pearson Correlation ($r$)"},
                xticklabels=False, yticklabels=False) # 隐藏繁杂的坐标轴标签
    
    axes[0].set_title(f"Correlation Matrix Heatmap (First {display_dim} Dims)", fontsize=13)
    axes[0].set_xlabel("Feature Dimensions", fontsize=12)
    axes[0].set_ylabel("Feature Dimensions", fontsize=12)
    
    # ==========================================
    # 子图 2: 非对角相关系数分布直方图 (方案 B)
    # ==========================================
    # 提取所有非对角线元素 (即真正的特征间相关系数)
    # 使用 np.triu_indices 提取上三角部分(不含对角线, k=1)，避免重复计算对称部分
    upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
    off_diagonal_corrs = corr_matrix[upper_tri_indices]
    
    # 计算绝对值，审稿人只关心有多"相关"，不关心正负
    abs_corrs = np.abs(off_diagonal_corrs)
    
    # 计算统计指标：多少比例的相关系数绝对值 < 0.1 或 0.2
    # 这就是你写在 Rebuttal 里的强有力证据
    prop_lt_01 = np.mean(abs_corrs < 0.1) * 100
    prop_lt_02 = np.mean(abs_corrs < 0.2) * 100
    
    print(f"  - Total feature pairs: {len(abs_corrs)}")
    print(f"  - Pairs with |r| < 0.1: {prop_lt_01:.2f}%")
    print(f"  - Pairs with |r| < 0.2: {prop_lt_02:.2f}%")
    
    # 画直方图
    sns.histplot(abs_corrs, bins=50, color='#4C72B0', ax=axes[1], stat='percent')
    
    # 添加阈值参考线
    axes[1].axvline(x=0.1, color='red', linestyle='--', linewidth=2, 
                    label=f'|r| < 0.1: {prop_lt_01:.1f}% pairs')
    axes[1].axvline(x=0.2, color='orange', linestyle='--', linewidth=2, 
                    label=f'|r| < 0.2: {prop_lt_02:.1f}% pairs')
    
    axes[1].set_title("Distribution of Absolute Pairwise Correlations", fontsize=13)
    axes[1].set_xlabel("Absolute Pearson Correlation ($|r|$)", fontsize=12)
    axes[1].set_ylabel("Percentage of Feature Pairs (%)", fontsize=12)
    axes[1].set_xlim(0, 1.0)
    axes[1].legend(fontsize=11)
    
    # 优化布局并保存
    sns.despine(ax=axes[1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved independence test plot to {save_path}\n")

if __name__ == "__main__":
    feature_dir = "./runs/rebuttal_features" # 修改为你的实际路径
    for task_id in range(6):
    # task_id = 0 # 分析第一个任务
    
        # 分析图像特征
        img_feats = load_features_for_task(feature_dir, task_id, mod='img')
        if img_feats is not None:
            print(f"Loaded Task {task_id} Image features: {img_feats.shape}")
            plot_pca_with_ellipse(img_feats, f"Task {task_id} Visual Features (PCA)", f"./figure/gaussianverify/task{task_id}_img_pca.svg")
            normality_test_and_plot(img_feats, f"Task {task_id} Visual", f"./figure/gaussianverify/task{task_id}_img_marginal.svg")
        
        # 分析文本特征
        txt_feats = load_features_for_task(feature_dir, task_id, mod='txt')
        if txt_feats is not None:
            print(f"Loaded Task {task_id} Text features: {txt_feats.shape}")
            plot_pca_with_ellipse(txt_feats, f"Task {task_id} Instruction Features (PCA)", f"./figure/gaussianverify/task{task_id}_txt_pca.svg")
            normality_test_and_plot(txt_feats, f"Task {task_id} Textual", f"./figure/gaussianverify/task{task_id}_txt_marginal.svg")
        
        # --- 新增: 调用独立性检验 ---
        if img_feats is not None:
            independence_test_and_plot(img_feats, f"Task {task_id} Visual", f"./figure/gaussianverify/task{task_id}_img_independence.svg")
            
        if txt_feats is not None:
            independence_test_and_plot(txt_feats, f"Task {task_id} Textual", f"./figure/gaussianverify/task{task_id}_txt_independence.svg")