# Filename: extract_bin_to_npz.py
# Description: 简化版主控脚本，处理单个 bin 文件夹并保存为 npz（路径与标签手动指定）
# Author: msy
# Date: 2025

import os
import numpy as np
from inf.payload.feature.payload_feature_extract import extract_multiscale_features, compute_fft_spectrogram  # type: ignore

# === 手动配置参数 ===
BIN_FOLDER_PATH = r"data/bin/payload/encrypted/CIC-IDS2017/Monday_1GB"
OUTPUT_NPZ_PATH = r"data/npz/payload_feature/encrypted/CIC-IDS2017/Monday_1GB.npz"
LABEL = 0  # 自定义标签，如 0 表示 Benign，1 表示 Malware

# === 主处理函数：读取 bin，提取特征，保存 npz ===
def process_bin_folder(bin_folder_path, output_path, label):
    print(f"\n[开始处理] 输入目录: {bin_folder_path}")
    print(f"[输出位置] 输出文件: {output_path}")
    print(f"[标签信息] 当前标签: {label}")

    spectrograms, labels, filenames = [], [], []
    total_files = 0
    valid_files = 0

    for fname in os.listdir(bin_folder_path):
        if not fname.endswith(".bin"):
            continue
        total_files += 1
        fpath = os.path.join(bin_folder_path, fname)
        try:
            with open(fpath, "rb") as f:
                payload = f.read()

            feats = extract_multiscale_features(payload)
            S = compute_fft_spectrogram(feats)

            if S.shape != (12, 32):
                print(f" [跳过] 无效形状: {fname} {S.shape}")
                continue

            spectrograms.append(S)
            labels.append(label)
            filenames.append(fname)
            valid_files += 1

        except Exception as e:
            print(f" [异常] 跳过 {fpath}，原因: {e}")

    if spectrograms:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(output_path,
                 spectrograms=np.array(spectrograms),
                 labels=np.array(labels),
                 filenames=np.array(filenames))
        print(f"\n[完成] 已保存: {output_path}")
        print(f"[统计] 总文件数: {total_files}，有效样本: {valid_files}")
    else:
        print(f"\n[失败] 无有效数据可保存: {bin_folder_path}")

# === 主程序入口 ===
if __name__ == "__main__":
    process_bin_folder(BIN_FOLDER_PATH, OUTPUT_NPZ_PATH, LABEL)
