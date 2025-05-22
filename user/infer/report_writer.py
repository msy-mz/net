# Filename: report_writer.py
# Path: user/infer/report_writer.py
# Description: 记录推理结果到 CSV，输出攻击日志与类别分布可视化。
# Author: msy
# Date: 2025

import csv
import matplotlib.pyplot as plt
from collections import Counter

# 写入预测结果 CSV
def write_prediction_csv(output_csv, result_list):
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['conn_id', 'predicted_label'])
        for conn_id, label in result_list:
            writer.writerow([conn_id, label])

# 写入攻击连接详情 CSV
def write_attack_log_csv(attack_csv, attack_log):
    if not attack_log:
        return
    with open(attack_csv, "w", newline='', encoding="utf-8") as f:
        fieldnames = ["conn_id", "label", "src_ip", "dst_ip", "src_port", "dst_port", "proto", "timestamp"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(attack_log)

# 输出分类结果统计到终端
def print_class_distribution(preds):
    total = len(preds)
    counter = Counter(preds)
    print("\n推理结果统计：")
    for cls in sorted(counter.keys()):
        count = counter[cls]
        ratio = count / total * 100
        print(f"类别 {cls}: {count} 条 ({ratio:.2f}%)")

# 保存分类分布柱状图
def plot_prediction_distribution(preds, output_path):
    counter = Counter(preds)
    labels = sorted(counter.keys())
    values = [counter[k] for k in labels]

    plt.figure(figsize=(6, 4))
    plt.bar([str(l) for l in labels], values, color='skyblue')
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")
    plt.title("Prediction Distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
