# feature_extraction.py

# 该模块用于从单个网络流的加密载荷中提取内容结构行为特征：
# 1. 使用多个尺度（如 32B, 64B, 128B）进行滑动窗口划分；
# 2. 对每个窗口批量提取熵、可打印比、均值、标准差等统计特征（矢量化）；
# 3. 对每条特征序列做傅里叶变换（FFT），转化为频域特征；
# 4. 输出拼接后的频谱矩阵 S ∈ ℝ^{d × K}，用于后续 Transformer 编码。

# 作者：msy
# 时间：2025


import numpy as np
from scipy.stats import entropy
from numpy.fft import fft

# === 默认参数配置 ===
WINDOW_SIZES = [32, 64, 128]  # 多尺度滑窗
STEP_RATIO = 0.5              # 滑窗步长比例
FFT_LEN = 64                  # 每条特征序列的 FFT 长度

# === 高效滑窗函数（向量化） ===
def sliding_window_view(arr, window_size, step):
    """
    将 1D 数组 arr 划分为多个滑窗段
    返回形状为 [N, window_size]
    """
    num_windows = (len(arr) - window_size) // step + 1
    if num_windows <= 0:
        return np.empty((0, window_size), dtype=arr.dtype)
    return np.stack([arr[i * step:i * step + window_size] for i in range(num_windows)])

# === 批量提取窗口统计特征（向量化） ===
def batch_extract_window_features(windows):
    """
    输入：滑窗字节序列，shape: [N, W]
    输出：每个窗口的4维统计特征，shape: [N, 4]
    """
    if len(windows) == 0:
        return np.empty((0, 4), dtype=np.float32)

    W = windows.shape[1]
    freq = np.apply_along_axis(lambda x: np.bincount(x, minlength=256) / W, 1, windows)
    ents = np.apply_along_axis(lambda f: entropy(f, base=2), 1, freq)
    printables = np.sum((windows >= 32) & (windows <= 126), axis=1) / W
    means = np.mean(windows, axis=1)
    stds = np.std(windows, axis=1)

    return np.stack([ents, printables, means, stds], axis=1)  # shape: [N, 4]

# === 多尺度滑窗特征提取 ===
def extract_multiscale_features(payload_bytes, window_sizes=WINDOW_SIZES, step_ratio=STEP_RATIO):
    """
    输入：原始字节（bytes）
    输出：多尺度特征序列 list of [4, T]
    """
    byte_array = np.frombuffer(payload_bytes, dtype=np.uint8)
    all_scale_features = []

    for w in window_sizes:
        step = int(w * step_ratio)
        if len(byte_array) < w:
            continue
        windows = sliding_window_view(byte_array, w, step)  # shape: [N, w]
        features = batch_extract_window_features(windows)   # shape: [N, 4]
        if features.shape[0] == 0:
            continue
        all_scale_features.append(features.T)  # 转置为 [4, T]

    return all_scale_features  # list of [4, T]

# === 特征序列的 FFT 变换 ===
def compute_fft_spectrogram(features_per_scale, fft_len=FFT_LEN):
    """
    输入：list of [4, T] 多尺度特征序列
    输出：拼接后的频谱图谱，shape: [4×scales, fft_len//2]
    """
    spectrum = []
    for feature_matrix in features_per_scale:  # [4, T]
        for seq in feature_matrix:  # 遍历每一行（每个特征）
            padded = seq.astype(np.float32)
            fft_vals = np.abs(fft(padded, n=fft_len))[:fft_len // 2]
            spectrum.append(fft_vals)

    return np.array(spectrum, dtype=np.float32)  # shape: [4×scales, fft_len//2]
