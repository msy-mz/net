# Filename: vis.py
# Path: inf/utils/vis.py
# Description: 特征可视化与能量分析工具模块
# Author: msy
# Date: 2025

import numpy as np
import matplotlib.pyplot as plt

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

def spectral_centroid(feature_matrix):
    freqs = np.arange(feature_matrix.shape[1])
    centroids = np.sum(feature_matrix * freqs, axis=1) / (np.sum(feature_matrix, axis=1) + 1e-6)
    return float(np.mean(centroids)), centroids.tolist()
