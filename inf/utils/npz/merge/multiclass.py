# merge_npz_multiclass_grouped.py
# author: msy
# 功能：将每个 npz 文件归类为统一标签（如 Weibo-1/Weibo-2 → Weibo）

import os
import numpy as np
import json
from tqdm import tqdm

ROOT_DIR = "inf/data/npz/tcp/USTC-TFC2016"
OUTPUT_FILE = "inf/data/npz/tcp/USTC-TFC2016_all_mulClass.npz"
LABEL_MAP_FILE = "inf/data/npz/tcp/USTC-TFC2016_all_mulClass_label.json"

# 自定义归一化函数：例如将 SMB-1、SMB-2 合并为 SMB
def normalize_class_name(name):
    if "-" in name:
        return name.split("-")[0]
    return name

def load_npz_with_label(npz_path, label_id):
    data = np.load(npz_path)
    if "spectrograms" not in data:
        raise ValueError(f"文件缺少 'X' 数组: {npz_path}")
    X = data["spectrograms"]
    y = np.full((X.shape[0],), label_id, dtype=np.int64)
    return X, y

def main():
    all_X, all_y = [], []
    label_map = {}  # 规范化类别名 → 整数标签
    current_label_id = 0

    for class_dir in ["Benign", "Malware"]:
        class_path = os.path.join(ROOT_DIR, class_dir)
        if not os.path.exists(class_path):
            continue

        print(f"\n 正在处理目录：{class_path}")
        for fname in sorted(os.listdir(class_path)):
            if not fname.endswith(".npz"):
                continue

            fpath = os.path.join(class_path, fname)
            raw_name = os.path.splitext(fname)[0]  # 原始名如 Weibo-2
            norm_name = normalize_class_name(raw_name)  # 统一名如 Weibo

            if norm_name not in label_map:
                label_map[norm_name] = current_label_id
                current_label_id += 1

            label_id = label_map[norm_name]

            try:
                X, y = load_npz_with_label(fpath, label_id)
                all_X.append(X)
                all_y.append(y)
                print(f"   {fname} → 类别：{norm_name}（标签ID：{label_id}，样本数：{X.shape[0]}）")
            except Exception as e:
                print(f"   加载失败：{fname}，错误：{e}")

    if not all_X:
        print(" 未加载到任何样本。")
        return

    # 合并所有数据
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    # 保存合并数据
    np.savez_compressed(OUTPUT_FILE, X=X_all, y=y_all)
    print(f"\n 合并完成，总样本数：{X_all.shape[0]}，保存至：{OUTPUT_FILE}")

    # 保存类别映射表
    with open(LABEL_MAP_FILE, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f" 类别映射表已保存至：{LABEL_MAP_FILE}")

    print("\n 标签映射预览：")
    for name, idx in label_map.items():
        print(f"  {idx}: {name}")

if __name__ == "__main__":
    main()
