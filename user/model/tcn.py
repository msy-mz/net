# Filename: tcn.py
# Path: user/model/tcn.py
# Description: 基于时间卷积网络（TCN）的身份建模模块，输入连接特征序列，输出身份嵌入向量。
# Author: msy
# Date: 2025

import torch
import torch.nn as nn

# 单个残差卷积块
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) * dilation,
                               dilation=dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) * dilation,
                               dilation=dilation)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

# 多层时间卷积网络（TCN）
class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_levels, kernel_size):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(num_levels):
            dilation = 2 ** i
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(TemporalBlock(in_dim, hidden_dim, kernel_size, dilation))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 身份建模模块：ID-TCN
class IDTCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, levels=3, kernel_size=3):
        """
        :param input_dim: TCN 输入维度
        :param hidden_dim: TCN 隐层维度
        :param embed_dim: 输出身份嵌入维度
        :param levels: TCN 层数
        :param kernel_size: 卷积核大小
        """
        super(IDTCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_dim, hidden_dim, levels, kernel_size)
        self.fc_identity = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)            # 转为 [B, C, T]
        x = self.tcn(x)                   # TCN 提取特征
        x = x.mean(dim=2)                 # 时间维平均池化
        h_identity = self.fc_identity(x)  # 投影为身份嵌入
        return h_identity
