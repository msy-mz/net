# Filename: error_analysis.py
# Path: inf/model/error_analysis.py
# Description: 多分类模型错误样本分析模块，记录错误样本索引及其预测与真实标签
# Author: msy
# Date: 2025

import os
import numpy as np

# 记录预测错误的样本索引与标签
def show_misclassified(y_true, y_pred, class_names, save_dir, max_errors_display):
    """
    保存前 max_errors_display 条分类错误样本信息至文本文件

    参数：
        y_true: 真实标签数组
        y_pred: 预测标签数组
        class_names: 类别名称列表
        save_dir: 输出文件保存目录
        max_errors_display: 显示最多多少条错误样本
    """
    errors = np.where(y_true != y_pred)[0]
    lines = []
    for i in errors[:max_errors_display]:
        lines.append(f"Sample {i}: True = {class_names[y_true[i]]}, Pred = {class_names[y_pred[i]]}")

    if save_dir:
        with open(os.path.join(save_dir, "sample_errors.txt"), "w") as f:
            for line in lines:
                f.write(line + "\n")
