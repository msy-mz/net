# Filename: merge_npz_with_label_map.py
# Description: 合并多个 .npz 文件，统一字符串标签为整数，输出最终训练数据
# Author: msy
# Date: 2025

import json
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 硬编码路径
INPUT_DIR = "data/cic2017/print/normalized/Malware"
OUTPUT_PATH = "data/cic2017/print/normalized/all.npz"

def merge_npz_folder(input_dir, output_path):
    all_features = []
    all_labels_str = []

    # 先读取所有标签（字符串）和特征
    for file in os.listdir(input_dir):
        if file.endswith(".npz"):
            path = os.path.join(input_dir, file)
            data = np.load(path, allow_pickle=True)
            features = data["features"]
            labels = data["labels"]  # 假设是 ["Benign", "Benign", ...]
            all_features.append(features)
            all_labels_str.extend(labels)
            print(f"[读取] {file} | 样本数: {len(labels)} | 标签示例: {labels[0] if len(labels) > 0 else '空'}")

    # 统一编码字符串标签为整数
    label_encoder = LabelEncoder()
    all_labels = label_encoder.fit_transform(all_labels_str)

    # 保存标签映射
    label_names = label_encoder.classes_
    label_map = {name: int(i) for i, name in enumerate(label_names)}
    print(f"[标签映射] {label_map}")

    # 合并特征
    merged_features = np.concatenate(all_features, axis=0)
    merged_labels = np.array(all_labels)

    # 保存为 .npz
    np.savez(output_path, features=merged_features, labels=merged_labels)
    print(f"[完成] 合并样本数: {len(merged_labels)} 类别数: {len(label_map)}")
    print(f"[保存] 文件写入: {output_path}")

    # 保存 label map 到 JSON 文件
    label_map_path = output_path.replace(".npz", "_label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"[保存] 标签映射写入: {label_map_path}")
if __name__ == "__main__":
    merge_npz_folder(INPUT_DIR, OUTPUT_PATH)
