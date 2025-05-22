# Filename: model_loader.py
# Path: user/infer/model_loader.py
# Description: 加载 FingerprintEncoder、IDTCNModel 和分类器组件及其权重。
# Author: msy
# Date: 2025

import torch
from user.model.encoder import FingerprintEncoder
from user.model.tcn import IDTCNModel
from user.infer.infer import IdentityClassifier

# 加载训练好的模型结构与参数
def load_model_components(model_path, config):
    """
    :param model_path: 模型参数文件路径
    :param config: 配置字典
    :return: encoder, model, classifier 三个组件（已加载参数）
    """
    state = torch.load(model_path, map_location='cpu')

    encoder = FingerprintEncoder(config["input_dim"],
                                 config["encoder_hidden_dim"],
                                 config["encoded_dim"])

    model = IDTCNModel(config["encoded_dim"],
                       config["tcn_hidden_dim"],
                       config["embed_dim"],
                       levels=config["tcn_levels"],
                       kernel_size=config["kernel_size"])

    classifier = IdentityClassifier(config["embed_dim"],
                                    config["num_classes"])

    encoder.load_state_dict(state["encoder"])
    model.load_state_dict(state["model"])
    classifier.load_state_dict(state["classifier"])

    encoder.eval()
    model.eval()
    classifier.eval()

    return encoder, model, classifier
