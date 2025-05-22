# Filename: base.py
# Path: user/model/eval/base.py
# Description: 对身份分类模型进行基础指标评估：准确率、精确率、召回率、F1 和混淆矩阵。
# Author: msy
# Date: 2025

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 分类指标评估函数
def eval_classification_metrics(dataloader, encoder, model, classifier, label_names=None):
    encoder.eval()
    model.eval()
    classifier.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in dataloader:
            x_encoded = encoder(x)
            h = model(x_encoded)
            logits = classifier(h)
            pred = torch.argmax(logits, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    # 输出分类报告
    print("\n分类报告：")
    print(classification_report(y_true, y_pred, target_names=label_names, digits=4) if label_names else classification_report(y_true, y_pred, digits=4))

    # 混淆矩阵可视化
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names if label_names else "auto",
                yticklabels=label_names if label_names else "auto")
    plt.title("身份分类混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.tight_layout()
    plt.show()
