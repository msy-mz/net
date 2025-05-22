# Filename: config.py
# Module: inf.utils.config
# Description: 加载 YAML 配置，支持默认值合并与回退提示
# Author: msy
# Date: 2025

import yaml

# 加载 YAML 配置，并合并默认配置
def load_config(path, default=None):
    with open(path, 'r') as f:
        loaded = yaml.safe_load(f) or {}

    # 如果未提供默认项，直接返回 YAML 中内容
    if not default:
        return loaded

    # 启动默认值合并机制
    config = default.copy()
    config.update(loaded)

    # 检查哪些默认值被启用，打印提示
    for key in default:
        if key not in loaded:
            print(f"[配置提示] '{key}' 未在 YAML 中指定，已使用默认值: {default[key]}")

    return config
