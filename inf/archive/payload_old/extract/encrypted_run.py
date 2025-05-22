# Filename: encrypted_payload_extract_run.py
# Description: 主控模块，配置参数并调用加密流量提取与保存逻辑
# Author: msy
# Date: 2025

import os
from inf.payload.extract.encrypted import extract_encrypted_flows, save_payloads

# 可配置参数区域
PCAP_PATH = r'data/pcap/CIC-IDS2017/Monday_1GB.pcap'           # 输入PCAP文件路径
OUTPUT_FOLDER = r'data/bin/payload/encrypted/CIC-IDS2017/Monday_1GB'         # 输出目录
MAX_PAYLOAD_LENGTH = None                                 # 每条流最大输出字节数
MIN_PAYLOAD_LENGTH = None                                   # 最小有效载荷字节（过滤短数据）
ENCRYPTED_PORTS = {
    'tcp': [443, 22],         # HTTPS、SSH
    'udp': [443, 500, 4500],  # QUIC、IPSec 等
}

# 执行流程
if __name__ == '__main__':
    print(f'[开始] 加载PCAP文件：{PCAP_PATH}')
    flows = extract_encrypted_flows(
        pcap_path=PCAP_PATH,
        encrypted_ports=ENCRYPTED_PORTS,
        min_len=MIN_PAYLOAD_LENGTH
    )

    print(f'[提取完成] 总流数：{len(flows)}，开始保存载荷...')
    file_count, total_bytes = save_payloads(
        flows=flows,
        out_folder=OUTPUT_FOLDER,
        max_payload_len=MAX_PAYLOAD_LENGTH
    )

    print(f'[完成] 共保存文件数：{file_count}，总字节数：{total_bytes}')
