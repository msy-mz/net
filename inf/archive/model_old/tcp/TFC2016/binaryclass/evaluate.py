"""
evaluate.py

评估 FT-Encoder++ 模型的分类性能，计算标准指标，
包括 Accuracy、Precision、Recall、F1、AUC 等。
可选支持 t-SNE 可视化、混淆矩阵显示等。

作者：msy
时间：2025
"""

import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.multiclass import unique_labels
import numpy as np

def evaluate_model(encoder, classifier, dataloader, class_names=None, device='cuda'):
    """
    输入：
        encoder: FT-Encoder++ 编码器（已训练）
        classifier: 分类器（MLP）
        dataloader: 测试数据 DataLoader
        class_names: 类别标签列表（用于打印）
    """
    encoder.eval()
    classifier.eval()
    all_preds, all_labels, all_proba = [], [], []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)

            out = classifier(encoder(xb))
            prob = torch.softmax(out, dim=1)
            preds = torch.argmax(prob, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

            # 仅用于二分类 AUC
            if prob.shape[1] == 2:
                all_proba.extend(prob[:, 1].cpu().numpy())

    # === 分类报告 ===
    print("\n 分类报告:")

    labels_present = sorted(list(unique_labels(all_labels, all_preds)))
    print(classification_report(
        all_labels,
        all_preds,
        labels=labels_present,
        target_names=[class_names[i] for i in labels_present] if class_names else None,
        digits=4,
        zero_division=0
    ))

    # === 混淆矩阵可视化 ===
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[class_names[i] for i in labels_present] if class_names else labels_present,
                yticklabels=[class_names[i] for i in labels_present] if class_names else labels_present)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # === AUC（仅支持二分类） ===
    if len(set(all_labels)) == 2:
        try:
            auc = roc_auc_score(all_labels, all_proba)
            print(f" AUC (binary): {auc:.4f}")
        except Exception as e:
            print(f" 无法计算 AUC：{e}")
