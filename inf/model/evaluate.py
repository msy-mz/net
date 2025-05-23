# Filename: evaluate.py
# Path: inf/model/evaluate.py
# Description: 
#   FT-Encoder++ 多分类模型评估模块，支持加载模型参数、分类报告、混淆矩阵、
#   t-SNE/PCA 可视化、AUC 计算、错误分析与图像保存（依赖分离模块）。
# Author: msy
# Date: 2025

import os
import torch
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import label_binarize

from inf.model.model_loader import load_model_weights
from inf.model.report_utils import format_classification_report,save_classification_report
from inf.model.visualization import visualize_all
from inf.model.error_analysis import show_misclassified

# 主评估函数（支持自动加载最优模型）
def evaluate_model(
    encoder,
    classifier,
    dataloader,
    class_names=None,
    device='cuda',
    save_dir=None,
    show_plot=True,
    max_errors_display=20,
    tsne_perplexity=30,
    pca_components=2,
    model_path_pair=None
):
    """
    多分类模型评估（支持外部传入模型路径并自动加载）

    参数：
        encoder: FT 编码器
        classifier: MLP 分类器
        dataloader: 验证数据集
        class_names: 类别名称列表
        device: 推理设备
        save_dir: 图像/报告保存目录
        show_plot: 是否显示图像
        max_errors_display: 错误样本最大显示数
        tsne_perplexity: t-SNE 可视化参数
        pca_components: PCA 可视化主成分数
        model_path_pair: 可选，(encoder_path, classifier_path)，若提供将自动加载
    """
    # 自动加载模型
    if model_path_pair:
        load_model_weights(encoder, classifier, model_path_pair, device)

    # 创建保存目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    encoder.eval()
    classifier.eval()
    all_preds, all_labels, all_proba, all_features = [], [], [], []

    # 推理与收集输出
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            feature = encoder(x_batch)
            logits = classifier(feature)
            prob = torch.softmax(logits, dim=1)
            preds = torch.argmax(prob, dim=1)

            all_features.append(feature.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_proba.extend(prob.cpu().numpy())

    # 转换为 numpy
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_proba = np.array(all_proba)
    all_features = np.concatenate(all_features, axis=0)

    labels_present = sorted(list(unique_labels(all_labels, all_preds)))
    names = [class_names[i] for i in labels_present] if class_names else labels_present

    # 分类报告
    report = classification_report(
        all_labels,
        all_preds,
        labels=labels_present,
        target_names=names,
        digits=4,
        zero_division=0,
        output_dict=True
    )
    print("\n分类报告:")
    print(format_classification_report(report, names))

    # AUC 计算
    auc_macro = None
    auc_micro = None
    if len(labels_present) >= 2:
        try:
            Y_true = label_binarize(all_labels, classes=labels_present)
            auc_macro = roc_auc_score(Y_true, all_proba, average='macro', multi_class='ovr')
            auc_micro = roc_auc_score(Y_true, all_proba, average='micro', multi_class='ovr')
            print(f"AUC (macro): {auc_macro:.4f}")
            print(f"AUC (micro): {auc_micro:.4f}")
        except Exception as e:
            print(f"无法计算多分类 AUC：{e}")

    # 保存分类报告
    if save_dir:
        save_classification_report(report, names, save_dir, auc_macro, auc_micro)

    # 可视化与错误分析
    visualize_all(
        y_true=all_labels,
        y_pred=all_preds,
        features=all_features,
        class_names=names,
        report_dict=report,
        save_dir=save_dir,
        show=show_plot,
        max_errors_display=max_errors_display,
        tsne_perplexity=tsne_perplexity,
        pca_components=pca_components
    )
    show_misclassified(all_labels, all_preds, names, save_dir, max_errors_display)
