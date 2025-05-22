# Filename: tls.py
# Module: inf.scripts.payload.extract.tls
# Description: 命令行工具，提取 PCAP 文件中的 TLS 加密流并保存为 .bin 文件
# Author: msy
# Date: 2025

"""
使用示例：
    python tls.py --input sample.pcap --output ./tls_bin --max-len 1024

参数说明：
    --input      输入的 pcap 文件路径
    --output     保存提取结果的目录（自动创建）
    --max-len    每条 TLS 流最多截取多少字节（默认 1024）
"""

import argparse
from inf.payload.extract.tls import extract_flows_from_pcap, save_payloads

def main():
    parser = argparse.ArgumentParser(description="提取 TLS 加密载荷为 .bin 文件")
    parser.add_argument("--input", required=True, help="输入的 PCAP 文件路径")
    parser.add_argument("--output", required=True, help="输出目录，保存 .bin 文件")
    parser.add_argument("--max-len", type=int, default=1024, help="每条流最大截断字节数（默认 1024）")
    args = parser.parse_args()

    # 提取 TLS 流并保存为 bin 文件
    flows = extract_flows_from_pcap(args.input)
    file_count, total_bytes = save_payloads(flows, args.output, max_payload_len=args.max_len)

    print(f"\n完成提取：生成 {file_count} 个 TLS 流文件，总字节数 {total_bytes}")

if __name__ == "__main__":
    main()
