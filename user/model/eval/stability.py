# Filename: stability.py
# Path: user/model/eval/stability.py
# Description: 评估同一身份用户连接在不同上下文下的嵌入稳定性。
# Author: msy
# Date: 2025

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# 稳定性评估函数：同类样本向量距离离散度
def eval_stability(dataloader, encoder, model):
    encoder.eval()
    model.eval()

    label_vectors = defaultdict(list)

    with torch.no_grad():
        for x, y in dataloader:
            h = model(encoder(x))  # [B, D]
            for vec, label in zip(h, y):
                label_vectors[label.item()].append(vec.cpu().numpy())

    std_list = []
    for label, vecs in label_vectors.items():
        if len(vecs) < 2:
            continue
        sims = cosine_similarity(vecs)
        upper_tri = sims[np.triu_indices(len(vecs), k=1)]
        std_list.append(np.std(upper_tri))

    if std_list:
        print(f"平均同类相似度方差（越小越稳定）：{np.mean(std_list):.6f}")
    else:
        print("样本不足，无法评估稳定性")
