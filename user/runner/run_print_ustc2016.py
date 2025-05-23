# Filename: run_batch_print.py
# Path: user/scripts/run_batch_print.py
# Description: 批量执行 USTC2016 所有 PCAP 文件的指纹提取，自动分类保存 CSV
# Author: msy
# Date: 2025

import os
import sys

sys.path.append(os.path.abspath("."))  # 确保可导入模块
from user.print.extract import extract_fingerprint

# 固定路径设置
ROOT_PCAP_DIR = "data/ustc2016/pcap"
ROOT_OUTPUT_DIR = "data/ustc2016/print"
MAX_PACKETS = None  # 可设为整数以限制每个流提取的最大包数

def batch_extract_all():
    for category in ["Benign", "Malware"]:
        input_dir = os.path.join(ROOT_PCAP_DIR, category)
        output_dir = os.path.join(ROOT_OUTPUT_DIR, category)
        os.makedirs(output_dir, exist_ok=True)

        for file in sorted(os.listdir(input_dir)):
            if file.endswith(".pcap"):
                input_path = os.path.join(input_dir, file)
                output_csv = os.path.join(output_dir, file.replace(".pcap", ".csv"))
                print(f"[{category}] 提取：{file} → {output_csv}")

                try:
                    extract_fingerprint(input_path, output_csv, label=None, max_packets=MAX_PACKETS)
                except Exception as e:
                    print(f"[错误] 处理 {file} 时出错：{e}")

if __name__ == "__main__":
    batch_extract_all()
