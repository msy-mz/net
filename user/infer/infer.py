# Filename: infer.py
# Path: user/infer/infer.py
# Description: 从原始 PCAP 开始进行推理，当前实现第一步：调用指纹特征提取。
# Author: msy
# Date: 2025

import os
import sys

# 添加路径，确保模块可导入
sys.path.append(os.path.abspath("."))

from user.print.extract import extract_fingerprint

# 推理主控函数（当前实现前置提取部分）
def infer_from_pcap(config):
    pcap_path = config["pcap_path"]
    temp_csv = config["temp_csv"]
    label = config.get("label", None)
    max_packets = config.get("max_packets", None)

    print("【1】从原始 PCAP 提取指纹特征...")
    extract_fingerprint(pcap_path, temp_csv, label, max_packets)

    # 后续：加载 temp_csv，执行模型推理，输出结果...
