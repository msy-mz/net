# Filename: feature_dict.py
# Path: inf/scripts/payload/feature_dict.py
# Description: 批量提取 .bin 文件频谱特征，按文件名组织为字典形式保存为 .npz，同时输出 meta.json。
#              保证 .bin → 特征 → 元信息一一对应，结构稳定可靠。
# Author: msy
# Date: 2025

import os
import json
import argparse
import numpy as np
from inf.payload.feature import extract_feature_from_bytes

def extract_feature_dict_from_dir(input_dir, output_dir, min_bytes=32):
    """
    遍历输入目录下所有 .bin 文件，提取频谱特征并保存为字典结构 .npz，同时输出 meta.json
    """
    os.makedirs(output_dir, exist_ok=True)
    bin_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".bin"))

    if not bin_files:
        print(f"[警告] 未在目录 {input_dir} 中发现 .bin 文件")
        return

    meta_path = os.path.join(input_dir, "meta.json")
    input_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            input_meta = json.load(f)

    feature_dict = {}
    output_meta = {}
    success, failed = 0, 0

    for fname in bin_files:
        fpath = os.path.join(input_dir, fname)
        try:
            with open(fpath, "rb") as f:
                payload = f.read()

            if len(payload) < min_bytes:
                print(f"[跳过] {fname} 长度不足 {min_bytes} 字节")
                continue

            feature = extract_feature_from_bytes(payload)
            if feature.size == 0:
                print(f"[跳过] {fname} 提取失败（空特征）")
                continue

            feature_dict[fname] = feature.astype(np.float32)
            output_meta[fname] = input_meta.get(fname, {})

            success += 1

        except Exception as e:
            print(f"[错误] {fname} 处理失败：{e}")
            failed += 1

    if not feature_dict:
        print("[结果] 未成功提取任何特征")
        return

    # 保存为字典格式的 .npz
    out_npz_path = os.path.join(output_dir, "features_dict.npz")
    np.savez_compressed(out_npz_path, **feature_dict)

    # 保存对应 meta.json
    out_meta_path = os.path.join(output_dir, "meta.json")
    with open(out_meta_path, "w") as f:
        json.dump(output_meta, f, indent=4)

    print(f"\n[完成] 成功 {success} 个，失败 {failed} 个，输出到：")
    print(f"  - 特征字典文件：{out_npz_path}")
    print(f"  - 元信息映射：{out_meta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 .bin 文件提取特征为字典 .npz，同时输出元信息 meta.json")
    parser.add_argument("--input-dir", required=True, help="输入目录（含 .bin 和 meta.json）")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--min-bytes", type=int, default=32, help="最小有效载荷字节数（默认 32）")
    args = parser.parse_args()

    extract_feature_dict_from_dir(args.input_dir, args.output_dir, args.min_bytes)
