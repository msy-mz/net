# Filename: train_runner.py
# Module: inf.runner.train_runner
# Description: 使用配置文件执行 FT-Encoder++ 多分类训练与评估（模块化版本）
# Author: msy
# Date: 2025

import argparse
import os
from datetime import datetime
from inf.utils.config import load_config
from inf.data.loader import load_data, build_dataloaders
from inf.model.train import train_model
from inf.model.evaluate import evaluate_model

DEFAULT_CONFIG_PATH = "inf/runner/config/ustc2016/tcp/multiclass/train.yaml"

def run_train(config):
    device = config.get("device", "cuda")

    # 时间戳与保存路径构造
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if config.get("use_timestamp", True) else ""
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints/default/")
    os.makedirs(checkpoint_dir, exist_ok=True)

    encoder_name = f"ft_{timestamp}.pt" if timestamp else "ft.pt"
    classifier_name = f"classifier_{timestamp}.pt" if timestamp else "classifier.pt"
    fig_dir_name = f"figs_{timestamp}" if timestamp else "figs"

    config["model_save_path"] = os.path.join(checkpoint_dir, encoder_name)
    config["classifier_save_path"] = os.path.join(checkpoint_dir, classifier_name)
    config["fig_save_dir"] = os.path.join(checkpoint_dir, fig_dir_name)
    os.makedirs(config["fig_save_dir"], exist_ok=True)

    # 加载数据
    print("加载数据...")
    X, y, class_names = load_data(config)
    num_classes = len(class_names)

    print("构建数据加载器...")
    train_loader, val_loader = build_dataloaders(X, y, config)

    print("启动模型训练...")
    encoder, classifier = train_model(train_loader, val_loader, num_classes, config, device)

    if not os.path.exists(config["model_save_path"]) or not os.path.exists(config["classifier_save_path"]):
        print("[跳过评估] 未保存新模型，验证集性能未提升")
        return

    print("执行模型评估...")
    evaluate_model(
        encoder,
        classifier,
        val_loader,
        class_names,
        device,
        save_dir=config["fig_save_dir"],
        model_path_pair=(config["model_save_path"], config["classifier_save_path"])
    )

    print("训练完成。模型与评估结果已保存。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    config = load_config(args.config)
    run_train(config)
