# Filename: model.py
# Path: inf/model/model.py
# Description:
#   定义 FT-Encoder++ 网络结构模块，包含频谱注意力（Spectral Attention）、
#   Transformer 编码器（TransformerEncoder）和 MLP 分类器（MLPClassifier）。
#   模型结构设计支持通用配置驱动构造，可适用于 TCP/TLS 等协议流量的多分类或二分类任务。
#   同时提供 build_models 函数，实现统一模型初始化，支持多数据集/任务解耦。
# Author: msy
# Date: 2025

import torch
import torch.nn as nn
import torch.nn.functional as F

# ======== 频谱注意力机制（对输入频谱特征做通道内注意力建模） ========
class SpectralAttention(nn.Module):
    def __init__(self, input_shape, d_model):
        super().__init__()
        C, F = input_shape
        self.query = nn.Linear(F, d_model)
        self.key = nn.Linear(F, d_model)
        self.value = nn.Linear(F, d_model)

    def forward(self, x):
        # x: [B, C, F]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.shape[-1] ** 0.5)
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        return output  # [B, C, d_model]

# ======== 位置编码模块（为频谱通道添加序列位置信息） ========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]  # [B, C, d_model]

# ======== Transformer 编码器模块（捕获通道间时序结构） ========
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=256, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.pos_enc = PositionalEncoding(d_model)

    def forward(self, x):
        x = self.pos_enc(x)      # 添加位置编码
        x = x.transpose(0, 1)    # 转换为 Transformer 输入格式：[C, B, d_model]
        for layer in self.layers:
            x = layer(x)
        return x.transpose(0, 1)  # 返回原格式：[B, C, d_model]

# ======== FTEncoder 主模块：频谱注意力 + Transformer + 平均池化 ========
class FTEncoder(nn.Module):
    def __init__(self, input_shape=(12, 32), d_model=128):
        super().__init__()
        self.attn = SpectralAttention(input_shape, d_model)
        self.encoder = TransformerEncoder(d_model=d_model)
        self.pooling = nn.AdaptiveAvgPool1d(1)  # 在通道维做平均池化

    def forward(self, x):
        # x: [B, C, F]
        x = self.attn(x)         # [B, C, d_model]
        x = self.encoder(x)      # [B, C, d_model]
        x = x.transpose(1, 2)    # [B, d_model, C]
        x = self.pooling(x).squeeze(-1)  # [B, d_model]
        return x

# ======== MLP 分类器：接收编码器输出，输出分类预测值 ========
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)

# ======== 构造器函数：从配置构造编码器与分类器 ========
# 参数 config 应包含：
#   - input_shape: list[int, int]，输入特征维度，如 [12, 32]
#   - d_model: int，Transformer 编码维度
#   - num_classes: int，分类类别数量
#   - device: str，使用设备（'cuda' 或 'cpu'）
def build_models(config):
    encoder = FTEncoder(
        input_shape=config['input_shape'],
        d_model=config['d_model']
    ).to(config['device'])

    classifier = MLPClassifier(
        input_dim=config['d_model'],
        num_classes=config['num_classes']
    ).to(config['device'])

    return encoder, classifier
