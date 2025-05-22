# Filename: tls_batch.py
# Path: inf/scripts/payload/extract/tls_batch.py
# Description: 批量提取目录下所有 PCAP 文件中的 TLS 加密流载荷为 .bin 文件
# Author: msy
# Date: 2025

"""
使用示例：
    python tls_batch.py --input-dir ./pcap_tls --output-dir ./tls_bin_batch --max-len 1024

参数说明：
    --input-dir   输入目录，包含多个 .pcap 文件
    --output-dir  输出目录，每个 PCAP 文件提取一个子目录保存 .bin 文件
    --max-len     每条 TLS 流最大截断字节数（默认 1024）
"""

import os
import argparse
from inf.payload.extract.tls import extract_payloads_from_pcap

def batch_extract_tls(input_dir, output_dir, max_len=1024):
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
        count, size = extract_payloads_from_pcap(pcap_path, out_subdir, max_len=max_len)
        print(f"[结果] 生成 {count} 个流，总字节数 {size}")
        total_files += 1
        total_payloads += count
        total_bytes += size

    print(f"\n==> 批处理完成：共处理 {total_files} 个 PCAP 文件，生成 {total_payloads} 个流，总字节数 {total_bytes}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量提取目录下所有 .pcap 文件的 TLS 加密流")
    parser.add_argument("--input-dir", required=True, help="输入 PCAP 文件目录")
    parser.add_argument("--output-dir", required=True, help="输出 .bin 文件保存主目录")
    parser.add_argument("--max-len", type=int, default=1024, help="每条流最大截断长度（默认 1024 字节）")
    args = parser.parse_args()

    batch_extract_tls(args.input_dir, args.output_dir, args.max_len)
