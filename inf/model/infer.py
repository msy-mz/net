# Filename: inference.py
# Path: inf/inference/inference.py
# Description: FT-Encoder++ 多分类推理模块，封装模型加载与预测，完全参数化以支持模块复用。
# Author: msy
# Date: 2025

import os
import torch
from inf.model.model import FTEncoder, MLPClassifier

# 模型加载函数（参数化）
def load_models(
    input_channels,
    freq_dim,
    hidden_dim,
    num_classes,
    encoder_path,
    classifier_path,
    device='cuda'
):
    """
    加载 FT-Encoder++ 编码器与 MLP 分类器

    参数：
        input_channels: 输入通道数
        freq_dim: 频谱维度
        hidden_dim: 编码器输出维度
        num_classes: 类别数量
        encoder_path: 编码器权重路径
        classifier_path: 分类器权重路径
        device: 运行设备（默认 'cuda'）

    返回：
        encoder, classifier：已加载权重并设为 eval 的模型
    """
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder model not found: {encoder_path}")
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Classifier model not found: {classifier_path}")

    encoder = FTEncoder(input_shape=(input_channels, freq_dim), d_model=hidden_dim).to(device)
    classifier = MLPClassifier(input_dim=hidden_dim, num_classes=num_classes).to(device)

    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))

    encoder.eval()
    classifier.eval()
    return encoder, classifier

# 推理函数（参数化）
def predict_class(
    feature_vector,
    encoder,
    classifier,
    class_names=None,
    device='cuda'
):
    """
    对单个样本向量进行分类预测

    参数：
        feature_vector: 形状为 (C, F) 的二维特征张量（C 为通道数）
        encoder: 编码器模型
        classifier: 分类器模型
        class_names: 可选，类别名称映射字典（如 {0: 'FTP', 1: 'Skype', ...}）
        device: 推理使用设备

    返回：
        类别名称（若提供 class_names），否则类别索引
    """
    with torch.no_grad():
        x = torch.tensor(feature_vector, dtype=torch.float32).to(device)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # 添加 batch 维度
        z = encoder(x)
        logits = classifier(z)
        pred_idx = torch.argmax(logits, dim=1).item()
        return class_names[pred_idx] if class_names else pred_idx
