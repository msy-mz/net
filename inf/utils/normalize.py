# Filename: normalize.py
# Path: inf/utils/normalize.py
# Description: 提供 log1p 和样本归一化两种特征标准化方法
# Author: msy
# Date: 2025

import numpy as np

def normalize_log1p(feature: np.ndarray) -> np.ndarray:
    """
    按 log1p 对谱图做非线性压缩（与训练对齐）
    """
    return np.log1p(feature)

def normalize_per_sample(feature: np.ndarray) -> np.ndarray:
    """
    对每个样本单独做标准化（均值为0，方差为1）
    """
    mean = feature.mean(axis=(0, 1), keepdims=True)
    std = feature.std(axis=(0, 1), keepdims=True) + 1e-8
    return (feature - mean) / std
