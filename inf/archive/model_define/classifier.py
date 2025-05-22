# src/classifier.py
# author: msy

import torch.nn as nn

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
