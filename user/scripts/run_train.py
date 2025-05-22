# Filename: run_train.py
# Path: user/scripts/run_train.py
# Description: 启动身份建模训练流程，加载配置并执行训练主控逻辑。
# Author: msy
# Date: 2025

import os
import sys
import yaml

# 添加项目路径，确保可以导入模块
sys.path.append(os.path.abspath("."))

from user.model.train import train

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config_path = "config/train.yaml"
    config = load_yaml(config_path)
    print(f"加载配置文件：{config_path}")
    train(config)
