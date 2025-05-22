# Filename: run_print.py
# Path: user/scripts/run_print.py
# Description: 启动网络流量指纹特征提取流程，读取 PCAP 并生成 CSV。
# Author: msy
# Date: 2025

import os
import sys
import yaml

# 添加模块路径（便于导入项目内部模块）
sys.path.append(os.path.abspath("."))

from user.print.extract import extract_fingerprint

# ========================== 可配置参数 ==========================
CONFIG_PATH = "user/scripts/config/print.yaml"
# ============================================================

# 加载 YAML 配置
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# 主控入口
if __name__ == "__main__":
    config = load_yaml(CONFIG_PATH)
    print(f"提取配置：{CONFIG_PATH}")

    pcap_path = config["pcap_path"]
    output_csv = config["output_csv"]
    label = config.get("label", None)

    # 兼容 None/null/"None"/"null"/"~"
    raw = config.get("max_packets", None)
    max_packets = None if str(raw).lower() in ("none", "null", "~") else int(raw)

    extract_fingerprint(pcap_path, output_csv, label, max_packets)
