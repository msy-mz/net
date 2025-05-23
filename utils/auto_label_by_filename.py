# Filename: auto_label_by_filename.py
# Description: 批量处理目录下 CSV 文件，将文件名作为标签写入 label 列
# Author: msy
# Date: 2025

import os
import pandas as pd
from tqdm import tqdm

def auto_label_csv_folder(input_dir, output_dir=None, inplace=False):
    """
    为目录下所有 CSV 文件添加标签列，标签来自文件名（不含扩展名）

    :param input_dir: 输入 CSV 文件夹路径
    :param output_dir: 输出路径（若为 None 且 inplace=False，则使用 input_dir）
    :param inplace: 是否原地修改（默认 False，输出到新目录）
    """
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    for file in tqdm(files, desc="标注标签中"):
        input_path = os.path.join(input_dir, file)
        df = pd.read_csv(input_path)

        label = os.path.splitext(file)[0]  # 文件名即标签
        df["label"] = label  # 直接替换原 label 列或新建 label 列

        save_path = os.path.join(output_dir, file)
        df.to_csv(save_path, index=False)

        print(f"[完成] {file} → 标签: {label}")

# 示例使用
if __name__ == "__main__":
    input_folder = "data/cic2017/print/Malware"                # 原始连接 CSV 所在目录
    output_folder = "data/cic2017/print/train"           # 输出目录，可为 None 表示覆盖
    auto_label_csv_folder(input_folder, output_folder, inplace=False)
