# Filename: run_infer.py
# Path: user/scripts/run_infer.py
# Description: 启动推理流程，调用模块化推理函数处理原始 PCAP 文件。
# Author: msy
# Date: 2025

import os
import sys
import yaml

# 添加项目路径，确保可以导入模块
sys.path.append(os.path.abspath("."))

from user.infer.infer import infer_from_pcap

# 加载配置文件
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config_path = "config/infer.yaml"
    config = load_yaml(config_path)
    print(f"加载推理配置：{config_path}")
    results = infer_from_pcap(config)
    print(f"推理完成，共处理连接数：{len(results)}")
