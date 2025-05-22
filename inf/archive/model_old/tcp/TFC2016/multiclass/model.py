# model_multiclass.py
# 作者：msy
# 日期：2025
# 说明：FT-Encoder++ 多分类版本，支持多类别频谱输入的编码与识别，基于频谱注意力+Transformer+MLP分类器构建

import torch
import torch.nn as nn
import torch.nn.functional as F

# ======== 频谱注意力模块 ========
class SpectralAttention(nn.Module):
    def __init__(self, input_shape, d_model):
        super().__init__()
        C, F = input_shape
        self.query = nn.Linear(F, d_model)
        self.key = nn.Linear(F, d_model)
        self.value = nn.Linear(F, d_model)

    def forward(self, x):
        Q = self.query(x)  # [B, C, d_model]
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.shape[-1] ** 0.5)
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)  # [B, C, d_model]
        return output

# ======== 位置编码模块 ========
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
        return x + self.pe[:, :x.size(1)]  # x: [B, C, d_model]

# ======== Transformer 编码器模块 ========
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=256, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.pos_enc = PositionalEncoding(d_model)

    def forward(self, x):
        x = self.pos_enc(x)       # [B, C, d_model]
        x = x.transpose(0, 1)     # [C, B, d_model]
        for layer in self.layers:
            x = layer(x)
        return x.transpose(0, 1)  # [B, C, d_model]

# ======== FT-Encoder 主体模块 ========
class FTEncoder(nn.Module):
    def __init__(self, input_shape=(12, 32), d_model=128):
        super().__init__()
        self.attn = SpectralAttention(input_shape, d_model)
        self.encoder = TransformerEncoder(d_model=d_model)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.attn(x)              # [B, C, d_model]
        x = self.encoder(x)           # [B, C, d_model]
        x = x.transpose(1, 2)         # [B, d_model, C]
        x = self.pooling(x).squeeze(-1)  # [B, d_model]
        return x

# ======== 多分类 MLP 分类器模块 ========
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=18):  # 根据实际类别数修改
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)
