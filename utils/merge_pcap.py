# Filename: merge_all_pcap_linux.py
# Description: 在 Linux 系统下合并指定目录中所有 pcap 文件为一个完整 pcap（分批）
# Author: msy
# Date: 2025

import os
import subprocess
import math

# 参数配置
PCAP_FOLDER = "data/ustc2016/pcap/Benign/Weibo"
OUTPUT_FILE = "data/ustc2016/pcap/Benign/Weibo.pcap"
INTERMEDIATE_DIR = "data/merged/tmp"
MERGECAP_CMD = "mergecap"  # Linux 下 mergecap 命令通常在系统 PATH 中
BATCH_SIZE = 100  # 每批最大文件数

def merge_batch(pcap_files, output_file):
    cmd = [MERGECAP_CMD, "-w", output_file] + pcap_files
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def merge_all_batches(pcap_folder, final_output, batch_size):
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(final_output), exist_ok=True)

    # 获取所有 pcap 文件路径
    pcap_files = []
    for root, _, files in os.walk(pcap_folder):
        for f in sorted(files):
            if f.endswith(".pcap"):
                pcap_files.append(os.path.join(root, f))

    if not pcap_files:
        print("未找到任何 pcap 文件。")
        return

    print(f"共找到 {len(pcap_files)} 个 pcap 文件，开始分批合并...")
    intermediate_files = []
    total_batches = math.ceil(len(pcap_files) / batch_size)

    for i in range(total_batches):
        batch = pcap_files[i * batch_size:(i + 1) * batch_size]
        batch_output = os.path.join(INTERMEDIATE_DIR, f"batch_{i}.pcap")
        print(f"  [Batch {i+1}/{total_batches}] 合并 {len(batch)} 个文件 → {batch_output}")
        merge_batch(batch, batch_output)
        intermediate_files.append(batch_output)

    print("所有批次合并完成，开始最终合并...")
    merge_batch(intermediate_files, final_output)
    print(f"最终合并完成：{final_output}")

if __name__ == "__main__":
    merge_all_batches(PCAP_FOLDER, OUTPUT_FILE, BATCH_SIZE)
