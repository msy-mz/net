# Filename: tcp_batch.py
# Module: scripts.payload.extract.tcp_batch
# Description: 命令行工具，批量提取目录下所有 PCAP 文件的 TCP 流载荷为 .bin 文件
# Author: msy
# Date: 2025

"""
使用示例：
    python tcp_batch.py --input-dir ./pcap_dir --output-dir ./tcp_bin_batch --max-len 1024

参数说明：
    --input-dir   包含多个 pcap 文件的输入目录
    --output-dir  输出 .bin 文件保存主目录（每个 pcap 建一个子目录）
    --max-len     每条 TCP 流最大截断字节数（默认 1024）
"""

import os
import argparse
from inf.payload.extract.tcp import extract_payloads_from_pcap

def batch_extract(input_dir, output_dir, max_len=1024):
    os.makedirs(output_dir, exist_ok=True)
    pcap_files = [f for f in os.listdir(input_dir) if f.endswith(".pcap")]

    if not pcap_files:
        print(f"[警告] 未在 {input_dir} 中发现 .pcap 文件")
        return

    total_files = 0
    total_payloads = 0
    total_bytes = 0

    for fname in sorted(pcap_files):
        pcap_path = os.path.join(input_dir, fname)
        name_prefix = os.path.splitext(fname)[0]
        out_subdir = os.path.join(output_dir, name_prefix)
        os.makedirs(out_subdir, exist_ok=True)

        print(f"\n[处理] {fname}")
        count, size = extract_payloads_from_pcap(pcap_path, out_subdir)
        print(f"[结果] 生成 {count} 个流，总字节数 {size}")
        total_files += 1
        total_payloads += count
        total_bytes += size

    print(f"\n==> 批处理完成：共处理 {total_files} 个 PCAP 文件，生成 {total_payloads} 个流，总字节 {total_bytes}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量提取目录下所有 .pcap 文件的 TCP 载荷")
    parser.add_argument("--input-dir", required=True, help="PCAP 文件所在目录")
    parser.add_argument("--output-dir", required=True, help="输出 .bin 文件保存目录")
    parser.add_argument("--max-len", type=int, default=1024, help="每条流最大截断字节数")
    args = parser.parse_args()

    batch_extract(args.input_dir, args.output_dir, args.max_len)
