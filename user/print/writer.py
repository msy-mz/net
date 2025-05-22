# Filename: writer.py
# Path: user/print/writer.py
# Description: 将结构化数据（如连接指纹）写入 CSV 文件，支持字段自定义。
# Author: msy
# Date: 2025

import os
import csv

# 将数据写入 CSV 文件
def write_csv(data, output_path, fieldnames):
    """
    :param data: List[Dict] 结构化数据
    :param output_path: str 输出文件路径
    :param fieldnames: List[str] 要写入的字段名
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
