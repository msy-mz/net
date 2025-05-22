# Filename: feature.py
# Path: inf/payload/feature.py
# Description: 从任意载荷字节序列中提取多尺度统计 + 频谱特征。
#              不依赖协议，适用于 TCP/TLS/QUIC 等任意原始 payload。
# Author: msy
# Date: 2025

import numpy as np
from scipy.stats import entropy
from numpy.fft import fft

# === 默认参数配置 ===
WINDOW_SIZES = [32, 64, 128]  # 多尺度滑动窗口大小
STEP_RATIO = 0.5              # 每个窗口的步长占比（例如 32 步长为 16）
FFT_LEN = 64                  # FFT 序列长度，决定输出频谱维度

# === 滑动窗口划分函数 ===
def sliding_window_view(arr, window_size, step):
    """
    将一维字节序列 arr 按指定窗口大小和步长划分为多个窗口
    返回：二维数组 [num_windows, window_size]
    """
    num_windows = (len(arr) - window_size) // step + 1
    if num_windows <= 0:
        return np.empty((0, window_size), dtype=arr.dtype)
    return np.stack([arr[i * step:i * step + window_size] for i in range(num_windows)])

# === 批量提取每个窗口的统计特征 ===
def batch_extract_window_features(windows):
    """
    输入：多个窗口序列，形状为 [N, W]
    输出：每个窗口的统计特征 [N, 4]（熵、可打印率、均值、标准差）
    """
    if len(windows) == 0:
        return np.empty((0, 4), dtype=np.float32)

    W = windows.shape[1]
    freq = np.apply_along_axis(lambda x: np.bincount(x, minlength=256) / W, 1, windows)
    ents = np.apply_along_axis(lambda f: entropy(f, base=2), 1, freq)
    printables = np.sum((windows >= 32) & (windows <= 126), axis=1) / W
    means = np.mean(windows, axis=1)
    stds = np.std(windows, axis=1)

    return np.stack([ents, printables, means, stds], axis=1)

# === 提取多尺度统计特征 ===
def extract_multiscale_features(payload_bytes, window_sizes=WINDOW_SIZES, step_ratio=STEP_RATIO):
    """
    从原始字节中提取多尺度滑窗统计特征
    返回：List of [4, T] 特征矩阵（每个尺度一个矩阵）
    """
    byte_array = np.frombuffer(payload_bytes, dtype=np.uint8)
    all_scale_features = []

    for w in window_sizes:
        step = int(w * step_ratio)
        if len(byte_array) < w:
            continue
        windows = sliding_window_view(byte_array, w, step)
        features = batch_extract_window_features(windows)
        if features.shape[0] == 0:
            continue
        all_scale_features.append(features.T)  # 转为 [4, T]

    return all_scale_features

# === 将特征序列进行频谱变换（FFT） ===
def compute_fft_spectrogram(features_per_scale, fft_len=FFT_LEN):
    """
    对多个尺度的特征序列做 FFT，得到频谱图
    返回：拼接后矩阵 [4×scales, fft_len//2]
    """
    spectrum = []
    for feature_matrix in features_per_scale:  # shape: [4, T]
        for seq in feature_matrix:  # 每个特征维度做一条 FFT
            padded = seq.astype(np.float32)
            fft_vals = np.abs(fft(padded, n=fft_len))[:fft_len // 2]
            spectrum.append(fft_vals)

    return np.array(spectrum, dtype=np.float32)

# === 封装高层接口 ===
def extract_feature_from_bytes(payload_bytes):
    """
    顶层封装：输入 byte 字节流 → 输出最终频谱矩阵（模型输入）
    """
    multiscale_feat = extract_multiscale_features(payload_bytes)
    return compute_fft_spectrogram(multiscale_feat)
