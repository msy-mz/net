# Filename: run_print_batch_parallel.py
# Path: user/runner/run_print_batch_parallel.py
# Description: 并行处理一个目录下所有 pcap 文件，读取配置文件执行
# Author: msy
# Date: 2025

import os
import sys
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath("."))  # 添加模块路径
from user.print.extract import extract_fingerprint

# 加载 YAML 配置
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# 包装函数用于并行任务
def extract_wrapper(args):
    input_path, output_csv, label, max_packets = args
    try:
        extract_fingerprint(input_path, output_csv, label, max_packets)
        return f"[完成] {os.path.basename(input_path)}"
    except Exception as e:
        return f"[错误] {os.path.basename(input_path)} 失败：{e}"

def batch_process_pcap_parallel(config, max_workers=8):
    pcap_folder = config["pcap_folder"]
    output_folder = config["output_folder"]
    os.makedirs(output_folder, exist_ok=True)

    label = config.get("label", None)
    raw = config.get("max_packets", None)
    max_packets = None if str(raw).lower() in ("none", "null", "~") else int(raw)

    files = [f for f in os.listdir(pcap_folder) if f.endswith(".pcap")]
    print(f"共发现 {len(files)} 个 pcap 文件，开始并行处理...")

    tasks = []
    for file in sorted(files):
        input_path = os.path.join(pcap_folder, file)
        output_csv = os.path.join(output_folder, file.replace(".pcap", ".csv"))
        tasks.append((input_path, output_csv, label, max_packets))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_wrapper, task) for task in tasks]
        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    CONFIG_PATH = "user/runner/config/print_batch.yaml"
    config = load_config(CONFIG_PATH)
    print(f"读取配置文件：{CONFIG_PATH}")
    batch_process_pcap_parallel(config, max_workers=8)
