# Filename: run_infer.py
# Path: inf/runner/run_infer.py
# Description: 对多个 PCAP 文件批量执行推理，并聚合结果
# Author: msy
# Date: 2025

import os
from inf.infer.infer_single import run_single_infer

def run_batch_infer(config_path, pcap_paths):
    """
    批量处理多个 PCAP 文件，返回合并后的结果。

    参数:
        config_path: infer.yaml 配置路径
        pcap_paths: PCAP 文件路径列表

    返回:
        final_result: {
            "summary": {...},
            "detailed": [...],
        }
    """
    all_details = []
    final_summary = {
        "total_flows": 0,
        "abnormal_flows": 0,
        "label_distribution": {}
    }

    for path in pcap_paths:
        result = run_single_infer(config_path, path)
        all_details.extend(result['detailed'])

        # 合并 summary
        final_summary["total_flows"] += result["summary"]["total_flows"]
        final_summary["abnormal_flows"] += result["summary"]["abnormal_flows"]

        for label, count in result["summary"]["label_distribution"].items():
            if label not in final_summary["label_distribution"]:
                final_summary["label_distribution"][label] = 0
            final_summary["label_distribution"][label] += count

    return {
        "summary": final_summary,
        "detailed": all_details
    }
