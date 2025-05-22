# Filename: report_utils.py
# Path: inf/model/report_utils.py
# Description: 控制台格式化输出 sklearn 分类报告（支持对齐、平均指标、accuracy）
# Author: msy
# Date: 2025

def format_classification_report(report_dict, class_names):
    """
    将 classification_report 字典格式化为对齐良好的文本输出

    参数：
        report_dict: classification_report(output_dict=True) 的返回结果
        class_names: 所有类别名称列表（按 index 对应）

    返回：
        lines: 字符串，可直接打印或保存
    """
    headers = ["precision", "recall", "f1-score", "support"]
    row_fmt = "{:<24s}" + "{:>10.4f}" * 3 + "{:>10}"  # 数值行
    head_fmt = "{:<24s}" + "{:>10}" * 4               # 标题行

    lines = []
    lines.append(head_fmt.format("", *headers))
    lines.append("=" * 60)

    for name in class_names:
        row = report_dict.get(name, {})
        support = int(row.get("support", 0))
        lines.append(row_fmt.format(
            name,
            row.get("precision", 0.0),
            row.get("recall", 0.0),
            row.get("f1-score", 0.0),
            support
        ))

    # accuracy 独立一行
    accuracy = report_dict.get("accuracy", 0.0)
    total_support = sum(report_dict.get(name, {}).get("support", 0) for name in class_names)
    lines.append("")
    lines.append(row_fmt.format("accuracy", 0.0, 0.0, accuracy, total_support))

    # macro avg 和 weighted avg
    for avg in ["macro avg", "weighted avg"]:
        row = report_dict.get(avg, {})
        support = int(row.get("support", 0))
        lines.append(row_fmt.format(
            avg,
            row.get("precision", 0.0),
            row.get("recall", 0.0),
            row.get("f1-score", 0.0),
            support
        ))

    return "\n".join(lines)

def save_classification_report(report_dict, class_names, save_dir, auc_macro=None, auc_micro=None):
    """
    保存分类报告为 .txt 和 .json 文件，可附加 AUC 分数

    参数：
        report_dict: classification_report(output_dict=True) 的结果
        class_names: 所有类别名（顺序对齐）
        save_dir: 目标保存目录
        auc_macro: 可选，macro 平均 AUC
        auc_micro: 可选，micro 平均 AUC
    """
    import os
    import json

    text_path = os.path.join(save_dir, "classification_report.txt")
    json_path = os.path.join(save_dir, "classification_report.json")

    row_fmt = "{:<22}{:>10}{:>10}{:>10}{:>10}"
    with open(text_path, "w") as f:
        # 标题行
        header = row_fmt.format("", "precision", "recall", "f1-score", "support")
        f.write(header + "\n")
        f.write("=" * len(header) + "\n")

        # 每类指标
        for name in class_names:
            stats = report_dict.get(name, {})
            f.write(row_fmt.format(
                name,
                f"{stats.get('precision', 0):.4f}",
                f"{stats.get('recall', 0):.4f}",
                f"{stats.get('f1-score', 0):.4f}",
                f"{int(stats.get('support', 0))}"
            ) + "\n")

        # accuracy
        acc = report_dict.get("accuracy", 0)
        total = sum(report_dict.get(name, {}).get("support", 0) for name in class_names)
        f.write("\n" + row_fmt.format("accuracy", "", "", f"{acc:.4f}", f"{total}") + "\n")

        # macro / weighted avg
        for avg_type in ["macro avg", "weighted avg"]:
            stats = report_dict.get(avg_type, {})
            f.write(row_fmt.format(
                avg_type,
                f"{stats.get('precision', 0):.4f}",
                f"{stats.get('recall', 0):.4f}",
                f"{stats.get('f1-score', 0):.4f}",
                f"{int(stats.get('support', 0))}"
            ) + "\n")

        # AUC
        if auc_macro is not None and auc_micro is not None:
            f.write("\n")
            f.write(f"AUC (macro): {auc_macro:.4f}\n")
            f.write(f"AUC (micro): {auc_micro:.4f}\n")

    # 保存 JSON 结构
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2)
