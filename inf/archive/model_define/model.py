# src/model.py
# author: msy

import torch.nn as nn
from spectral_attention import SpectralAttention
from transformer_encoder import TransformerEncoder

class FTEncoder(nn.Module):
    def __init__(self, input_shape=(12, 32), d_model=128):
        super().__init__()
        self.attn = SpectralAttention(input_shape, d_model)
        self.encoder = TransformerEncoder(d_model=d_model)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [B, C, F]
        x = self.attn(x)              # [B, C, d_model]
        x = self.encoder(x)          # [B, C, d_model]
        x = x.transpose(1, 2)        # [B, d_model, C]
        x = self.pooling(x).squeeze(-1)  # [B, d_model]
        return x
