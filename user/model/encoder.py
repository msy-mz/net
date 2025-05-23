# Filename: encoder.py
# Path: user/model/encoder.py
# Description: 将连接级别的协议字段特征编码为统一维度的向量，供时间序列模型使用。
# Author: msy
# Date: 2025

import torch
import torch.nn as nn

# 多层感知机编码器
class FingerprintEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        :param input_dim: 输入特征维度
        :param hidden_dim: MLP 隐层维度
        :param output_dim: 输出编码维度（用于 TCN 输入或直接分类）
        """
        super(FingerprintEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [batch_size, input_dim]
        return self.encoder(x)  # 返回: [batch_size, output_dim]
