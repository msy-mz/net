# Filename: run_eval.py
# Path: scripts/run_eval.py
# Description: 单独运行身份模型评估流程，加载模型和验证集进行测试。
# Author: msy
# Date: 2025

import os
import sys
import yaml
import torch

sys.path.append(os.path.abspath("."))

from user.model.loader import load_model_components
from user.model.classifier import load_classifier
from user.eval.evaluate_identity import evaluate_all
from user.data.identity_dataset import build_eval_loader

# 加载 YAML 配置
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_yaml("config/eval.yaml")
    state = torch.load(config["model_path"], map_location="cpu")

    encoder, model, _ = load_model_components(config["model_path"], config)
    classifier = load_classifier(state["classifier"],
                                 embed_dim=config["embed_dim"],
                                 num_classes=config["num_classes"])

    dataloader = build_eval_loader(config["eval_csv"], config["batch_size"])

    evaluate_all(dataloader, encoder, model, classifier,
                 label_names=[str(i) for i in range(config["num_classes"])])
