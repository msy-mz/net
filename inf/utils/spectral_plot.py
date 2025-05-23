# Filename: spectral_plot.py
# Path: inf/utils/spectral_plot.py
# Description: 将频谱特征数组渲染为图像并保存（异常兼容版）
# Author: msy
# Date: 2025

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# === 简化配置 ===
SPECTRUM_INPUT_DIR = '/tmp/realtime_spectrum'
PLOT_OUTPUT_DIR = '/tmp/realtime_spectrum_plot'
COLORMAP = 'viridis'

os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

def plot_spectrum_from_file(json_path):
    with open(json_path, 'r') as f:
        spectrum = np.array(json.load(f))

    fname = os.path.splitext(os.path.basename(json_path))[0]
    output_path = os.path.join(PLOT_OUTPUT_DIR, f"{fname}.png")

    plt.figure(figsize=(6, 4))
    plt.imshow(spectrum, aspect='auto', cmap=COLORMAP)
    plt.colorbar(label='Intensity')
    plt.title(f"Spectrum: {fname}")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Channel")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_all_recent():
    if not os.path.isdir(SPECTRUM_INPUT_DIR):
        return  # 如果目录不存在，跳过绘图
    json_files = sorted(
        [f for f in os.listdir(SPECTRUM_INPUT_DIR) if f.endswith('.json')],
        reverse=True
    )
    for fname in json_files[:30]:
        fpath = os.path.join(SPECTRUM_INPUT_DIR, fname)
        plot_spectrum_from_file(fpath)
