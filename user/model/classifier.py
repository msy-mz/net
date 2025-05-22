# Filename: classifier.py
# Path: user/model/classifier.py
# Description: 身份分类器模块，定义结构并支持从模型状态加载。
# Author: msy
# Date: 2025

import torch
import torch.nn as nn

# 分类器结构定义
class IdentityClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# 加载分类器参数（外部传入结构参数 + 权重）
def load_classifier(model_state_dict, embed_dim, num_classes):
    classifier = IdentityClassifier(embed_dim=embed_dim, num_classes=num_classes)
    classifier.load_state_dict(model_state_dict)
    classifier.eval()
    return classifier
