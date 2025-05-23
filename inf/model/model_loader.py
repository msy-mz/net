# Filename: model_loader.py
# Path: inf/model/model_loader.py
# Description: FT 编码器与分类器模型加载模块，支持外部模型路径自动加载
# Author: msy
# Date: 2025

import torch

# 加载 encoder 与 classifier 的权重
def load_model_weights(encoder, classifier, model_path_pair, device):
    encoder_path, classifier_path = model_path_pair
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    print(f"[加载最优模型] encoder: {encoder_path}")
    print(f"[加载最优模型] classifier: {classifier_path}")
