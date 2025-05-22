# Filename: visualization.py
# Path: inf/model/visualization.py
# Description: 多分类模型可视化模块，支持混淆矩阵、类别分布、F1、t-SNE、PCA 等绘图与保存
# Author: msy
# Date: 2025

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# 可视化总控函数
def visualize_all(
    y_true,
    y_pred,
    features,
    class_names,
    report_dict,
    save_dir,
    show,
    max_errors_display,
    tsne_perplexity,
    pca_components
):
    draw_confusion_matrix(y_true, y_pred, class_names, save_dir, show)
    draw_support_bar(y_true, class_names, save_dir, show)
    draw_f1_bar(report_dict, class_names, save_dir, show)
    draw_tsne(features, y_true, class_names, save_dir, show, tsne_perplexity)
    draw_pca(features, y_true, class_names, save_dir, show, pca_components)

# 绘制混淆矩阵
def draw_confusion_matrix(y_true, y_pred, class_names, save_dir, show):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    if show:
        plt.show()
    plt.close()

# 绘制支持度条形图
def draw_support_bar(y_true, class_names, save_dir, show):
    counter = Counter(y_true)
    indices = sorted(counter.keys())
    counts = [counter[i] for i in indices]
    labels = [class_names[i] for i in indices]

    plt.figure(figsize=(10, 4))
    plt.bar(labels, counts)
    plt.title("Support Distribution")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Sample Count")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "support_distribution.png"), dpi=300)
    if show:
        plt.show()
    plt.close()

# 绘制 F1 分数条形图
def draw_f1_bar(report_dict, class_names, save_dir, show):
    f1s = [report_dict.get(name, {}).get("f1-score", 0) for name in class_names]
    plt.figure(figsize=(10, 4))
    plt.bar(class_names, f1s)
    plt.title("F1 Score per Class")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("F1-score")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "f1_scores.png"), dpi=300)
    if show:
        plt.show()
    plt.close()

# t-SNE 可视化
def draw_tsne(features, labels, class_names, save_dir, show, perplexity):
    tsne_result = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(features)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1],
                    hue=[class_names[i] for i in labels],
                    palette="tab20", legend='full', s=20)
    plt.title("t-SNE Feature Projection")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "tsne.png"), dpi=300)
    if show:
        plt.show()
    plt.close()

# PCA 可视化
def draw_pca(features, labels, class_names, save_dir, show, components):
    pca_result = PCA(n_components=components).fit_transform(features)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1],
                    hue=[class_names[i] for i in labels],
                    palette="tab20", legend='full', s=20)
    plt.title("PCA Feature Projection")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "pca.png"), dpi=300)
    if show:
        plt.show()
    plt.close()
