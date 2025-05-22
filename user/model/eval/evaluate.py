# Filename: evaluate.py
# Path: user/model/eval/evaluate.py
# Description: 综合调用多个评估模块，生成身份模型完整性能评估报告。
# Author: msy
# Date: 2025

from user.model.eval.base import eval_classification_metrics
from user.model.eval.embedding import visualize_embedding, analyze_embedding_similarity
from user.model.eval.stability import eval_stability

# 综合身份评估函数
def evaluate_all(dataloader, encoder, model, classifier, label_names=None):
    print("\n========== 评估：分类准确性 ==========")
    eval_classification_metrics(dataloader, encoder, model, classifier, label_names)

    print("\n========== 评估：嵌入结构可视化 ==========")
    visualize_embedding(dataloader, encoder, model, label_names)

    print("\n========== 评估：嵌入相似度分布分析 ==========")
    analyze_embedding_similarity(dataloader, encoder, model)

    print("\n========== 评估：同类样本嵌入稳定性 ==========")
    eval_stability(dataloader, encoder, model)
