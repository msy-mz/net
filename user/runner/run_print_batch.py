# Filename: run_batch_by_config.py
# Description: 批量处理一个目录下所有 pcap 文件，读取配置文件执行
# Author: msy
# Date: 2025

import os
import sys
import yaml

sys.path.append(os.path.abspath("."))  # 添加模块路径
from user.print.extract import extract_fingerprint

# 加载 YAML 配置
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def batch_process_pcap(config):
    pcap_folder = config["pcap_folder"]
    output_folder = config["output_folder"]
    os.makedirs(output_folder, exist_ok=True)

    label = config.get("label", None)
    raw = config.get("max_packets", None)
    max_packets = None if str(raw).lower() in ("none", "null", "~") else int(raw)

    files = [f for f in os.listdir(pcap_folder) if f.endswith(".pcap")]
    print(f"共发现 {len(files)} 个 pcap 文件，开始批量处理...")

    for file in sorted(files):
        input_path = os.path.join(pcap_folder, file)
        output_csv = os.path.join(output_folder, file.replace(".pcap", ".csv"))
        print(f"[处理] {file} → {output_csv}")

        try:
            extract_fingerprint(input_path, output_csv, label, max_packets)
        except Exception as e:
            print(f"[错误] {file} 处理失败：{e}")

if __name__ == "__main__":
    CONFIG_PATH = "user/runner/config/print_batch.yaml"
    config = load_config(CONFIG_PATH)
    print(f"读取配置文件：{CONFIG_PATH}")
    batch_process_pcap(config)
