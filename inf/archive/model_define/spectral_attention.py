# src/spectral_attention.py
# author: msy

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralAttention(nn.Module):
    def __init__(self, input_shape, d_model):
        super().__init__()
        C, F = input_shape
        self.query = nn.Linear(F, d_model)
        self.key = nn.Linear(F, d_model)
        self.value = nn.Linear(F, d_model)

    def forward(self, x):
        # x: [B, C, F]
        Q = self.query(x)  # [B, C, d_model]
        K = self.key(x)    # [B, C, d_model]
        V = self.value(x)  # [B, C, d_model]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.shape[-1] ** 0.5)
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)  # [B, C, d_model]
        return output
