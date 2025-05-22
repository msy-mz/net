# src/transformer_encoder.py
# author: msy

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=256, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.pos_enc = PositionalEncoding(d_model)

    def forward(self, x):
        # x: [B, C, d_model]
        x = self.pos_enc(x)
        x = x.transpose(0, 1)  # Transformer expects [seq_len, B, d_model]
        for layer in self.layers:
            x = layer(x)
        return x.transpose(0, 1)
