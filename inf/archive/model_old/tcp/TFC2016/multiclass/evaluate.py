# evaluate_multiclass.py
# 作者：msy
# 日期：2025
# 说明：评估 FT-Encoder++ 多分类模型，自动保存可视化图（混淆矩阵、support、F1、t-SNE、PCA、错误样本）

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.utils.multiclass import unique_labels

# ======== 主函数 ========
def evaluate_model(encoder, classifier, dataloader, class_names=None, device='cuda', save_dir=None):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    encoder.eval()
    classifier.eval()
    all_preds, all_labels, all_proba, all_features = [], [], [], []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)

            feature = encoder(xb)
            logits = classifier(feature)
            prob = torch.softmax(logits, dim=1)
            preds = torch.argmax(prob, dim=1)

            all_features.append(feature.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            all_proba.extend(prob.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_proba = np.array(all_proba)
    all_features = np.concatenate(all_features, axis=0)

    labels_present = sorted(list(unique_labels(all_labels, all_preds)))
    names = [class_names[i] for i in labels_present] if class_names else labels_present

    # 分类报告
    report = classification_report(
        all_labels, all_preds, labels=labels_present,
        target_names=names, digits=4, zero_division=0, output_dict=True
    )
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, labels=labels_present,
          target_names=names, digits=4, zero_division=0))

    # === 添加保存文本和JSON报告 ===
    if save_dir:
        with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
            f.write(classification_report(
                all_labels, all_preds, labels=labels_present,
                target_names=names, digits=4, zero_division=0
            ))
        import json
        with open(os.path.join(save_dir, "classification_report.json"), "w") as f:
            json.dump(report, f, indent=2)

    # 可视化
    plot_confusion_matrix(all_labels, all_preds, names, save_path=path(save_dir, "confusion_matrix.png"))
    plot_support_distribution(all_labels, names, save_path=path(save_dir, "support_distribution.png"))
    plot_f1_scores(report, names, save_path=path(save_dir, "f1_scores.png"))
    visualize_tsne(all_features, all_labels, names, save_path=path(save_dir, "tsne.png"))
    visualize_pca(all_features, all_labels, names, save_path=path(save_dir, "pca.png"))
    print_sample_errors(all_labels, all_preds, names, path(save_dir, "sample_errors.txt"))

    # AUC
    if len(labels_present) > 2:
        try:
            Y_true = label_binarize(all_labels, classes=labels_present)
            auc_macro = roc_auc_score(Y_true, all_proba, average='macro', multi_class='ovr')
            auc_micro = roc_auc_score(Y_true, all_proba, average='micro', multi_class='ovr')
            print(f"AUC (macro): {auc_macro:.4f}")
            print(f"AUC (micro): {auc_micro:.4f}")
        except Exception as e:
            print(f"无法计算多分类 AUC：{e}")


# ======== 工具函数 ========
def path(folder, filename):
    return os.path.join(folder, filename) if folder else None

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_support_distribution(y_true, class_names, save_path=None):
    counts = Counter(y_true)
    keys = sorted(counts.keys())
    values = [counts[k] for k in keys]
    labels = [class_names[k] for k in keys]

    plt.figure(figsize=(10, 4))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha='right')
    plt.title("Support Distribution")
    plt.ylabel("Sample Count")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_f1_scores(report_dict, class_names, save_path=None):
    f1s = [report_dict[name]["f1-score"] for name in class_names if name in report_dict]
    plt.figure(figsize=(10, 4))
    plt.bar(class_names, f1s)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.title("F1 Score per Class")
    plt.ylabel("F1-score")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def visualize_tsne(features, labels, class_names, save_path=None):
    tsne_result = TSNE(n_components=2, random_state=42).fit_transform(features)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1],
                    hue=[class_names[i] for i in labels],
                    palette="tab20", legend='full', s=20)
    plt.title("Feature Space (t-SNE)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def visualize_pca(features, labels, class_names, save_path=None):
    pca_result = PCA(n_components=2).fit_transform(features)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1],
                    hue=[class_names[i] for i in labels],
                    palette="tab20", legend='full', s=20)
    plt.title("Feature Space (PCA)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def print_sample_errors(y_true, y_pred, class_names, save_path=None, max_display=20):
    print("\nMisclassified Samples (Top %d):" % max_display)
    errors = np.where(y_true != y_pred)[0]
    lines = []
    for i in errors[:max_display]:
        line = f"Sample {i}: True = {class_names[y_true[i]]}, Pred = {class_names[y_pred[i]]}"
        print(line)
        lines.append(line)
    if save_path:
        with open(save_path, "w") as f:
            for l in lines:
                f.write(l + "\n")

