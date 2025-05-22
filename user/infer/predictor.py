# Filename: predictor.py
# Path: user/infer/predictor.py
# Description: 执行推理操作，将特征向量输入模型，输出预测结果。
# Author: msy
# Date: 2025

import torch

# 执行推理，返回预测类别列表与分类日志（含攻击连接）
def run_inference_on_tensor(features_tensor, conn_ids, encoder, model, classifier, conn_meta=None):
    """
    :param features_tensor: Tensor, [N, D]
    :param conn_ids: List[str], 连接标识
    :param encoder: FingerprintEncoder
    :param model: IDTCNModel
    :param classifier: IdentityClassifier
    :param conn_meta: dict, 可选，包含连接元数据
    :return: 预测结果列表 [(conn_id, label)], 攻击日志列表
    """
    results = []
    preds = []
    attack_log = []

    with torch.no_grad():
        for i, feature in enumerate(features_tensor):
            x = feature.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
            x_encoded = encoder(x)
            h = model(x_encoded)
            logits = classifier(h)
            predicted_class = torch.argmax(logits, dim=1).item()

            conn_id = conn_ids[i]
            preds.append(predicted_class)
            results.append((conn_id, predicted_class))

            if predicted_class != 0 and conn_meta and conn_id in conn_meta:
                entry = {"conn_id": conn_id, "label": predicted_class}
                entry.update(conn_meta[conn_id])
                attack_log.append(entry)

    return results, preds, attack_log
