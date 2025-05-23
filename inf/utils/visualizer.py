# Filename: visualizer.py
# Path: inf/utils/visualizer.py
# Description: 所有结构化推理结果的图像绘制（分布图、频谱图、PCA 等）
# Author: msy
# Date: 2025

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.decomposition import PCA

def draw_visualizations(detailed_results, summary, output_dir):
    """
    给定推理结构化结果，绘制所有图像（分类分布图、频谱图、PCA、熵图等）

    参数:
        detailed_results: 每条流的详细结构化记录列表
        summary: 总体 summary 统计（含标签分布）
        output_dir: 图像输出目录（如 result/inf/infer/feature_vis）
    """
    os.makedirs(output_dir, exist_ok=True)
    spectrum_dir = os.path.join(output_dir, 'global_spectrum')
    os.makedirs(spectrum_dir, exist_ok=True)

    # 分类柱状图
    labels = summary['label_distribution']
    plt.figure(figsize=(10, 5))
    plt.bar(labels.keys(), labels.values(), color='skyblue')
    plt.title("Predicted Label Distribution")
    plt.ylabel("Flow Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_distribution.png'))
    plt.close()

    # 提取特征
    mean_energies = [rec["mean_energy"] for rec in detailed_results]
    mean_centroids = [rec["mean_centroid"] for rec in detailed_results]
    spectra = [np.array(rec["spectrum"]) for rec in detailed_results]
    all_features = np.stack(spectra)

    # 能量直方图
    plt.figure()
    plt.hist(mean_energies, bins=30, color='skyblue', edgecolor='black')
    plt.title('Mean Energy Histogram')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_energy_hist.png'))
    plt.close()

    # 谱心直方图
    plt.figure()
    plt.hist(mean_centroids, bins=30, color='salmon', edgecolor='black')
    plt.title('Mean Centroid Histogram')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_centroid_hist.png'))
    plt.close()

    # 能量 vs 谱心散点图
    plt.figure()
    plt.scatter(mean_energies, mean_centroids, alpha=0.5, color='purple')
    plt.title('Mean Energy vs. Mean Centroid')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_vs_centroid_scatter.png'))
    plt.close()

    # PCA 可视化
    X = [x.flatten() for x in spectra]
    labels = [rec["label"] for rec in detailed_results]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for label in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(X_pca[idxs, 0], X_pca[idxs, 1], label=label, alpha=0.6, s=30)
    plt.title("PCA of Spectrum Features")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(spectrum_dir, 'pca_spectrum_clusters.png'))
    plt.close()

    # Normal vs Abnormal 平均谱
    normal, abnormal = [], []
    entropy_vals, entropy_labels = [], []

    for rec in detailed_results:
        spec = np.array(rec['spectrum'])
        mean_chan = np.mean(spec, axis=0)
        prob = mean_chan / np.sum(mean_chan)
        entropy_vals.append(entropy(prob))
        entropy_labels.append('Abnormal' if rec['is_abnormal'] else 'Normal')
        (abnormal if rec['is_abnormal'] else normal).append(spec)

    if normal:
        mean_norm = np.mean(np.stack(normal), axis=0)
        norm_curve = np.mean(mean_norm, axis=0)
        plt.plot(norm_curve, label='Normal', color='green')
    if abnormal:
        mean_abn = np.mean(np.stack(abnormal), axis=0)
        abn_curve = np.mean(mean_abn, axis=0)
        plt.plot(abn_curve, label='Abnormal', color='red')

    plt.title("Mean Spectrum: Normal vs Abnormal")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(spectrum_dir, 'normal_vs_abnormal_spectrum.png'))
    plt.close()

    # 谱熵分布图
    plt.figure(figsize=(8, 4))
    plt.hist(entropy_vals, bins=30, color='gray', edgecolor='black')
    plt.title("Spectral Entropy Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(spectrum_dir, 'spectral_entropy_hist.png'))
    plt.close()
