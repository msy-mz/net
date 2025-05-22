# Filename: train.py
# Path: inf/model/train.py
# Description: FT-Encoder++ 多分类训练模块，支持 early stopping 与 macro F1 验证评估
# Author: msy
# Date: 2025

import torch
import os
import torch.nn.functional as F
from sklearn.metrics import f1_score
from inf.model.model import FTEncoder, MLPClassifier

# 模型训练函数
# 训练 FT-Encoder++ 多分类模型，支持 early stopping 和模型保存功能
def train_model(train_loader, val_loader, num_classes, config, device):
    # 解包配置参数
    input_channels = config["input_channels"]
    freq_dim = config["freq_dim"]
    hidden_dim = config["hidden_dim"]
    lr = config["lr"]
    epochs = config["epochs"]
    early_stop_patience = config["early_stop_patience"]
    grad_clip_value = config.get("grad_clip_value", 1.0)
    no_improve_epochs_init = config.get("no_improve_epochs_init", 0)
    pretrained_encoder_path = config.get("pretrained_encoder_path")
    pretrained_classifier_path = config.get("pretrained_classifier_path")
    model_save_path = config.get("model_save_path")
    classifier_save_path = config.get("classifier_save_path")

    no_improve_epochs = no_improve_epochs_init

    # 初始化模型
    encoder = FTEncoder(input_shape=(input_channels, freq_dim), d_model=hidden_dim).to(device)
    classifier = MLPClassifier(input_dim=hidden_dim, num_classes=num_classes).to(device)

    # 加载预训练模型参数（如路径存在）
    if pretrained_encoder_path and os.path.exists(pretrained_encoder_path):
        encoder.load_state_dict(torch.load(pretrained_encoder_path, map_location=device, weights_only=True))
    if pretrained_classifier_path and os.path.exists(pretrained_classifier_path):
        classifier.load_state_dict(torch.load(pretrained_classifier_path, map_location=device, weights_only=True))

    # 优化器与损失函数
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # 初始评估
    init_f1 = evaluate_val(encoder, classifier, val_loader, device)
    print(f"[Init] 验证 Macro F1（加载{'预训练' if pretrained_encoder_path else '随机'}参数后）: {init_f1:.4f}")

    best_f1 = init_f1
    best_encoder = encoder.state_dict()
    best_classifier = classifier.state_dict()

    # 训练主循环
    for epoch in range(1, epochs + 1):
        encoder.train()
        classifier.train()
        total_loss = 0

        # 批次训练
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            embeddings = encoder(X)
            logits = classifier(embeddings)
            loss = criterion(logits, y)

            # 忽略 NaN
            if torch.isnan(loss):
                print("出现 NaN，跳过该 batch")
                continue

            # 反向传播与梯度裁剪
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=grad_clip_value)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=grad_clip_value)
            optimizer.step()

            total_loss += loss.item()

        # 验证集评估
        val_f1 = evaluate_val(encoder, classifier, val_loader, device)
        print(f"[Epoch {epoch}/{epochs}] Loss: {total_loss:.4f} | Val Macro F1: {val_f1:.4f}")

        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_encoder = encoder.state_dict()
            best_classifier = classifier.state_dict()
            print(f"   新最优模型，保存中... Val Macro F1: {val_f1:.4f}")
            if model_save_path:
                torch.save(best_encoder, model_save_path)
            if classifier_save_path:
                torch.save(best_classifier, classifier_save_path)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"   未提升，EarlyStop计数：{no_improve_epochs}/{early_stop_patience}")
            if no_improve_epochs >= early_stop_patience:
                print(" 触发EarlyStopping，提前终止训练。")
                break

    # 加载最优模型参数
    encoder.load_state_dict(best_encoder)
    classifier.load_state_dict(best_classifier)

    return encoder, classifier

# 验证集评估函数
# 返回：macro F1-score
def evaluate_val(encoder, classifier, loader, device):
    encoder.eval()
    classifier.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = classifier(encoder(X))
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return macro_f1
