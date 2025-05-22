# Filename: infer_runner.py
# Path: inf/runner/infer_runner.py
# Description: FT-Encoder++ 推理流程主控脚本（最终优化版），仅保留结构化结果与统计图。
# Author: msy
# Date: 2025

import os
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from inf.infer.infer import load_models, predict_class
from inf.payload.extract.tcp import extract_payloads_from_pcap
from inf.payload.feature import extract_feature_from_bytes
from inf.utils.normalize import normalize_log1p
from inf.utils.vis import spectral_centroid

# ========== 阶段 1：加载全局配置 ==========
print("[阶段] 加载推理配置")
CONFIG_PATH = 'inf/runner/config/ustc2016/tcp/multiclass/infer.yaml'
with open(CONFIG_PATH, 'r') as f:
    global_cfg = yaml.safe_load(f)

PCAP_PATH = global_cfg['pcap_path']
OUTPUT_DIR = global_cfg['output_dir']
MAX_PAYLOAD_LEN = global_cfg['max_payload_len']
MAX_FLOWS = global_cfg.get('max_flows', None)
TRAIN_CFG_PATH = global_cfg['config_path']
NORMAL_RANGE = range(global_cfg['normal_class_range'][0], global_cfg['normal_class_range'][1] + 1)

print(f"[路径确认] PCAP_PATH: {PCAP_PATH}")
print(f"[路径确认] OUTPUT_DIR: {OUTPUT_DIR}")
print(f"[路径确认] TRAIN_CFG_PATH: {TRAIN_CFG_PATH}")
if MAX_FLOWS:
    print(f"[配置] 最大提取流数量: {MAX_FLOWS}")

# ========== 阶段 2：加载模型配置 ==========
with open(TRAIN_CFG_PATH, 'r') as f:
    config = yaml.safe_load(f)

input_channels = config['input_channels']
freq_dim = config['freq_dim']
hidden_dim = config['hidden_dim']
device = config.get('device', 'cuda')
encoder_path = config['pretrained_encoder_path']
classifier_path = config['pretrained_classifier_path']
label_map_path = config['label_map_path']

print(f"[路径确认] encoder_path: {encoder_path}")
print(f"[路径确认] classifier_path: {classifier_path}")

# ========== 阶段 3：加载类别标签 ==========
with open(label_map_path, 'r') as f:
    name_to_idx = json.load(f)
class_names = {v: k for k, v in name_to_idx.items()}
num_classes = len(class_names)
name_to_idx = {v: k for k, v in class_names.items()}
normal_class_ids = set(NORMAL_RANGE)

# ========== 阶段 4：加载模型 ==========
print("[阶段] 加载模型中...")
encoder, classifier = load_models(
    input_channels=input_channels,
    freq_dim=freq_dim,
    hidden_dim=hidden_dim,
    num_classes=num_classes,
    encoder_path=encoder_path,
    classifier_path=classifier_path,
    device=device
)
print("[状态] 模型加载完成")

# ========== 阶段 5：提取载荷 ==========
print("[阶段] 提取 TCP 流载荷中...")
bin_files = extract_payloads_from_pcap(
    pcap_path=PCAP_PATH,
    output_dir=OUTPUT_DIR,
    max_len=MAX_PAYLOAD_LEN,
    max_flows=MAX_FLOWS
)
print(f"[统计] 实际提取流数量: {len(bin_files)}，输出路径: {OUTPUT_DIR}")

meta_path = os.path.join(OUTPUT_DIR, 'meta.json')
with open(meta_path, 'r') as f:
    meta = json.load(f)

# ========== 阶段 6：特征提取与推理 ==========
print("[阶段] 开始推理与特征分析...")
detailed_results = []
abnormal_count = 0
label_counter = Counter()

for i, bin_path in enumerate(bin_files):
    fname = os.path.basename(bin_path)
    with open(bin_path, 'rb') as f:
        payload_bytes = f.read()

    feature = extract_feature_from_bytes(payload_bytes)
    feature = normalize_log1p(feature)

    pred = predict_class(feature, encoder, classifier, class_names, device)
    mean_energy = float(np.sum(feature))
    mean_centroid, centroids = spectral_centroid(feature)
    mean_per_channel = feature.mean(axis=1).tolist()
    std_per_channel = feature.std(axis=1).tolist()

    record = meta.get(fname, {})
    is_abnormal = name_to_idx[pred] not in normal_class_ids
    if is_abnormal:
        abnormal_count += 1

    record.update({
        "filename": fname,
        "label": pred,
        "is_abnormal": is_abnormal,
        "mean_energy": mean_energy,
        "mean_centroid": mean_centroid,
        "centroids": centroids,
        "mean_per_channel": mean_per_channel,
        "std_per_channel": std_per_channel
    })

    detailed_results.append(record)
    label_counter[pred] += 1

    os.remove(bin_path)  # 删除 bin 文件
    print(f"[{i+1}/{len(bin_files)}] {fname} → {pred} | abnormal={is_abnormal}")
    # 缓存频谱特征以便后续绘图使用
    record['spectrum'] = feature.tolist()  # 直接存入 JSON-safe 格式

# ========== 阶段 7：保存结果 ==========
print("[阶段] 保存推理结果")
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, 'detailed_results.json'), 'w') as f:
    json.dump(detailed_results, f, indent=2)

summary = {
    "total_flows": len(detailed_results),
    "abnormal_flows": abnormal_count,
    "label_distribution": dict(label_counter)
}
with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

# ========== 阶段 8：绘制分布图 ==========
print("[阶段] 绘制分类分布图")
plt.figure(figsize=(10, 5))
plt.bar(label_counter.keys(), label_counter.values(), color='skyblue')
plt.title("Predicted Label Distribution")
plt.ylabel("Flow Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'label_distribution.png'))
plt.close()

print(f"[完成] 共推理 {len(detailed_results)} 条流，异常 {abnormal_count} 条")
print(f"[输出] summary.json, detailed_results.json, label_distribution.png")

# ========== 阶段 9：统计并保存特征分布信息 ==========
print("[阶段] 统计并保存特征分布信息")

# 提取所有流的 mean_energy 和 mean_centroid
mean_energies = [record["mean_energy"] for record in detailed_results]
mean_centroids = [record["mean_centroid"] for record in detailed_results]

# 计算统计信息
feature_stats = {
    "mean_energy": {
        "min": float(np.min(mean_energies)),
        "max": float(np.max(mean_energies)),
        "mean": float(np.mean(mean_energies)),
        "std": float(np.std(mean_energies))
    },
    "mean_centroid": {
        "min": float(np.min(mean_centroids)),
        "max": float(np.max(mean_centroids)),
        "mean": float(np.mean(mean_centroids)),
        "std": float(np.std(mean_centroids))
    }
}

# 保存统计信息为 JSON 文件
feature_stats_path = os.path.join(OUTPUT_DIR, 'feature_stats.json')
with open(feature_stats_path, 'w') as f:
    json.dump(feature_stats, f, indent=2)

print(f"[输出] 特征统计信息保存至: {feature_stats_path}")

# ========== 阶段 10：特征可视化 ==========
print("[阶段] 绘制特征可视化图")

# 创建可视化输出路径
fig_dir = os.path.join(OUTPUT_DIR, 'feature_vis')
os.makedirs(fig_dir, exist_ok=True)

# 绘制 mean_energy 直方图
plt.figure()
plt.hist(mean_energies, bins=30, color='skyblue', edgecolor='black')
plt.title('Mean Energy Histogram')
plt.xlabel('Mean Energy')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'mean_energy_hist.png'))
plt.close()

# 绘制 mean_centroid 直方图
plt.figure()
plt.hist(mean_centroids, bins=30, color='salmon', edgecolor='black')
plt.title('Mean Centroid Histogram')
plt.xlabel('Mean Centroid')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'mean_centroid_hist.png'))
plt.close()

# 绘制能量与谱心的散点图
plt.figure()
plt.scatter(mean_energies, mean_centroids, alpha=0.5, color='purple')
plt.title('Mean Energy vs. Mean Centroid')
plt.xlabel('Mean Energy')
plt.ylabel('Mean Centroid')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'energy_vs_centroid_scatter.png'))
plt.close()

# 绘制箱线图（均值和标准差）
plt.figure()
plt.boxplot([mean_energies, mean_centroids], tick_labels=['Mean Energy', 'Mean Centroid'])
plt.title('Boxplot of Energy and Centroid')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'boxplot_energy_centroid.png'))
plt.close()

print(f"[完成] 特征图保存至: {fig_dir}")


# ========== 阶段 11：绘制频谱代表图与统计图 ==========
print("[阶段] 绘制频谱代表图与统计图")

global_spectrum_dir = os.path.join(OUTPUT_DIR, 'feature_vis', 'global_spectrum')
os.makedirs(global_spectrum_dir, exist_ok=True)

# 收集所有频谱数据
all_features = []
label_to_example = {}

for record in detailed_results:
    label = record['label']
    spectrum = np.array(record['spectrum'])
    all_features.append(spectrum)

    # 每个标签只保留一条代表样本
    if label not in label_to_example:
        label_to_example[label] = spectrum

all_features = np.stack(all_features)

# ========== 绘制代表性频谱图 ==========
rep_fig_dir = os.path.join(global_spectrum_dir, 'representative')
os.makedirs(rep_fig_dir, exist_ok=True)

for label, spectrum in label_to_example.items():
    plt.figure(figsize=(6, 3))
    plt.imshow(spectrum, aspect='auto', origin='lower', cmap='viridis')
    plt.title(f"Spectrogram (Representative) - {label}")
    plt.xlabel("Time")
    plt.ylabel("Channel")
    plt.colorbar(label='Magnitude')
    plt.tight_layout()
    plt.savefig(os.path.join(rep_fig_dir, f"spectrogram_{label}.png"))
    plt.close()

# ========== 绘制总体统计图 ==========
# 计算通道均值和标准差
mean_spectrum = np.mean(all_features, axis=0)
std_spectrum = np.std(all_features, axis=0)

# 绘制通道均值曲线
plt.figure(figsize=(8, 4))
for i in range(mean_spectrum.shape[0]):
    plt.plot(mean_spectrum[i], label=f'Ch {i}')
plt.title("Mean Spectrum Per Channel")
plt.xlabel("Frequency Bin")
plt.ylabel("Mean Magnitude")
plt.legend(loc='upper right', fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(global_spectrum_dir, 'mean_spectrum.png'))
plt.close()

# 绘制标准差热图
plt.figure(figsize=(6, 4))
plt.imshow(std_spectrum, aspect='auto', origin='lower', cmap='magma')
plt.title("Spectral Std Heatmap")
plt.xlabel("Frequency Bin")
plt.ylabel("Channel")
plt.colorbar(label='Std Dev')
plt.tight_layout()
plt.savefig(os.path.join(global_spectrum_dir, 'std_heatmap.png'))
plt.close()

print(f"[完成] 总体频谱统计图与代表谱图保存至: {global_spectrum_dir}")


# ========== 阶段 12：类别均值频谱图与 PCA 可视化 ==========
print("[阶段] 绘制类别频谱均值图与 PCA 聚类图")

from sklearn.decomposition import PCA

class_spectra = {}
flattened_features = []
flattened_labels = []

# 收集每类频谱并构造平铺数据用于 PCA
for record in detailed_results:
    label = record['label']
    spectrum = np.array(record['spectrum'])

    if label not in class_spectra:
        class_spectra[label] = []
    class_spectra[label].append(spectrum)

    flattened_features.append(spectrum.flatten())
    flattened_labels.append(label)

# ========== 类别频谱均值图 ==========
plt.figure(figsize=(10, 5))
for label, spectra_list in class_spectra.items():
    stacked = np.stack(spectra_list)
    mean_spectrum = np.mean(stacked, axis=0)
    mean_per_channel = np.mean(mean_spectrum, axis=0)
    plt.plot(mean_per_channel, label=label)

plt.title("Class-wise Mean Spectrum")
plt.xlabel("Frequency Bin")
plt.ylabel("Mean Magnitude (Across Channels)")
plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(global_spectrum_dir, 'classwise_mean_spectrum.png'))
plt.close()

# ========== PCA 聚类图 ==========
pca = PCA(n_components=2)
X_pca = pca.fit_transform(flattened_features)

plt.figure(figsize=(8, 6))
unique_labels = sorted(set(flattened_labels))
for label in unique_labels:
    indices = [i for i, l in enumerate(flattened_labels) if l == label]
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=label, alpha=0.6, s=30)

plt.title("PCA of Flattened Spectrum Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(global_spectrum_dir, 'pca_spectrum_clusters.png'))
plt.close()

print(f"[完成] 类别频谱均值图与 PCA 图保存至: {global_spectrum_dir}")


# ========== 阶段 13：异常对比频谱图与谱熵直方图 ==========
print("[阶段] 绘制异常对比频谱图与谱熵图")

from scipy.stats import entropy

normal_spectra = []
abnormal_spectra = []
entropy_vals = []
entropy_labels = []

# 分组统计谱数据和熵
for record in detailed_results:
    spectrum = np.array(record['spectrum'])
    label = record['label']
    is_abnormal = record['is_abnormal']

    mean_per_channel = np.mean(spectrum, axis=0)
    prob = mean_per_channel / np.sum(mean_per_channel)
    spec_entropy = entropy(prob)

    entropy_vals.append(spec_entropy)
    entropy_labels.append('Abnormal' if is_abnormal else 'Normal')

    if is_abnormal:
        abnormal_spectra.append(spectrum)
    else:
        normal_spectra.append(spectrum)

# ========== 正常 vs 异常频谱均值图 ==========
if normal_spectra and abnormal_spectra:
    mean_normal = np.mean(np.stack(normal_spectra), axis=0)
    mean_abnormal = np.mean(np.stack(abnormal_spectra), axis=0)

    norm_curve = np.mean(mean_normal, axis=0)
    abnorm_curve = np.mean(mean_abnormal, axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(norm_curve, label='Normal', color='green')
    plt.plot(abnorm_curve, label='Abnormal', color='red')
    plt.title("Mean Spectrum: Normal vs Abnormal")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Mean Magnitude (Across Channels)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(global_spectrum_dir, 'normal_vs_abnormal_spectrum.png'))
    plt.close()

# ========== 谱熵直方图 ==========
plt.figure(figsize=(8, 4))
colors = ['green' if l == 'Normal' else 'red' for l in entropy_labels]
plt.hist(entropy_vals, bins=30, color='gray', edgecolor='black')
plt.title("Spectral Entropy Distribution")
plt.xlabel("Spectral Entropy")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(global_spectrum_dir, 'spectral_entropy_hist.png'))
plt.close()

print(f"[完成] 异常对比频谱图与谱熵图保存至: {global_spectrum_dir}")
