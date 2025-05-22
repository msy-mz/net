# Filename: embedding.py
# Path: user/model/eval/embedding.py
# Description: 对监督身份模型的嵌入向量结构进行可视化和相似性评估（如 T-SNE、类间相似度）。
# Author: msy
# Date: 2025

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# 身份嵌入 T-SNE 可视化
def visualize_embedding(dataloader, encoder, model, label_names=None):
    encoder.eval()
    model.eval()

    embeddings = []
    labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x_encoded = encoder(x)
            h = model(x_encoded)
            embeddings.append(h.cpu())
            labels.extend(y.cpu().numpy())

    embeddings = torch.cat(embeddings, dim=0).numpy()
    labels = np.array(labels)

    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels,
                    palette="tab10", style=labels,
                    legend="full", s=60)
    plt.title("身份嵌入 T-SNE 可视化")
    plt.tight_layout()
    plt.show()

# 类内类间相似度分析（平均）
def analyze_embedding_similarity(dataloader, encoder, model):
    encoder.eval()
    model.eval()

    vectors = []
    labels = []

    with torch.no_grad():
        for x, y in dataloader:
            h = model(encoder(x))
            vectors.append(h.cpu())
            labels.extend(y.cpu().numpy())

    vectors = torch.cat(vectors, dim=0).numpy()
    labels = np.array(labels)
    sims = cosine_similarity(vectors)

    same_class = []
    diff_class = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            sim = sims[i][j]
            if labels[i] == labels[j]:
                same_class.append(sim)
            else:
                diff_class.append(sim)

    print(f"平均同类相似度: {np.mean(same_class):.4f}")
    print(f"平均异类相似度: {np.mean(diff_class):.4f}")

    # 可选：可视化相似度分布
    sns.kdeplot(same_class, label="同类", fill=True)
    sns.kdeplot(diff_class, label="异类", fill=True)
    plt.title("身份嵌入相似度分布")
    plt.xlabel("余弦相似度")
    plt.legend()
    plt.tight_layout()
    plt.show()
