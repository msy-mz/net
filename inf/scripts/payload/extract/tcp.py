# Filename: tcp.py
# Module: inf.scripts.payload.extract.tcp
# Description: 命令行工具，提取 PCAP 文件中的 TCP 流并保存为 .bin 文件
# Author: msy
# Date: 2025

"""
使用示例：
    python tcp.py --input sample.pcap --output ./tcp_bin --max-len 1024

参数说明：
    --input      输入的 pcap 文件路径
    --output     保存提取结果的目录（自动创建）
    --max-len    每条 TCP 流最多截取多少字节（默认 1024）
"""

import argparse
from inf.payload.extract.tcp import extract_payloads_from_pcap

def main():
    parser = argparse.ArgumentParser(description="提取 TCP 载荷为 .bin 文件")
    parser.add_argument("--input", required=True, help="输入的 PCAP 文件路径")
    parser.add_argument("--output", required=True, help="输出目录，保存 .bin 文件")
    parser.add_argument("--max-len", type=int, default=1024, help="每条流最大截断字节数（默认 1024）")
    args = parser.parse_args()

    file_count, total_bytes = extract_payloads_from_pcap(args.input, args.output, max_len=args.max_len)
    print(f"\n完成提取：生成 {file_count} 个 TCP 流文件，总字节数 {total_bytes}")

if __name__ == "__main__":
    main()
