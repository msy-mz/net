# scripts/normalize_npz_log1p.py
# description: 对现有 spectrogram_dataset.npz 中的 X 做 log1p 或归一化处理

import numpy as np
import os

# === 参数配置 ===
INPUT_FILE = "inf/data/npz/tcp/npz_all/multiclass/USTC-TFC2016_all_mulClass.npz"
OUTPUT_FILE = "inf/data/npz/tcp/npz_all/multiclass/USTC-TFC2016_all_mulClass_normalized.npz"  # 可与原始区分
USE_LOG1P = True  # 或 False 使用归一化

# === 加载数据 ===
data = np.load(INPUT_FILE)
X = data["X"]
y = data["y"]

# === 处理 ===
if USE_LOG1P:
    X = np.log1p(X)
else:
    # 按样本归一化（标准化每张图）
    mean = X.mean(axis=(1, 2), keepdims=True)
    std = X.std(axis=(1, 2), keepdims=True) + 1e-8
    X = (X - mean) / std

# === 保存 ===
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
np.savez(OUTPUT_FILE, X=X, y=y)

print(f"保存处理后的数据集: {OUTPUT_FILE}")
print(f"特征形状: {X.shape}, 标签形状: {y.shape}")
