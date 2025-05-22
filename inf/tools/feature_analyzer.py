# Filename: feature_analyzer.py
# Path: inf/tools/feature_analyzer.py
# Description: 对提取的 payload 特征进行频谱可视化与能量中心分析，用于解释模型输入
# Author: msy
# Date: 2025

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from inf.payload.feature import extract_feature_from_bytes

# === 频谱热力图绘制函数 ===
def visualize_spectrogram(feature_matrix, title="Feature Spectrogram", save_path=None):
    plt.figure(figsize=(10, 5))
    plt.imshow(feature_matrix, aspect='auto', cmap='viridis')
    plt.title(title)
    plt.xlabel("Frequency Bin")
    plt.ylabel("Feature × Scale")
    plt.colorbar(label='Amplitude')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

# === 能量谱质心（Spectral Centroid）分析 ===
def spectral_centroid(feature_matrix):
    """
    计算每个通道的频谱质心（中心频率）
    返回：均值 + 各通道列表
    """
    freqs = np.arange(feature_matrix.shape[1])
    centroids = np.sum(feature_matrix * freqs, axis=1) / (np.sum(feature_matrix, axis=1) + 1e-6)
    return float(np.mean(centroids)), centroids.tolist()

# === 分析主函数 ===
def analyze_feature_dir(bin_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = {}

    for fname in os.listdir(bin_dir):
        if not fname.endswith('.bin'):
            continue
        bin_path = os.path.join(bin_dir, fname)
        with open(bin_path, 'rb') as f:
            payload_bytes = f.read()

        feature = extract_feature_from_bytes(payload_bytes)
        title = fname
        img_path = os.path.join(output_dir, fname.replace('.bin', '.png'))
        visualize_spectrogram(feature, title=title, save_path=img_path)

        mean_centroid, centroid_list = spectral_centroid(feature)
        report[fname] = {
            "mean_energy": float(np.sum(feature)),
            "mean_centroid": mean_centroid,
            "centroids": centroid_list
        }

    # 保存分析报告
    report_path = os.path.join(output_dir, 'feature_analysis.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] 分析完成，图像和报告已保存至: {output_dir}")

# === 示例入口 ===
if __name__ == '__main__':
    BIN_DIR = 'output/inf/infer/'            # 包含 .bin 的目录
    OUTPUT_DIR = 'output/inf/analysis/'       # 分析输出图像与报告目录
    analyze_feature_dir(BIN_DIR, OUTPUT_DIR)
