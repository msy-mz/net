# Filename: train.py
# Module: 
# Description: FT-Encoder++ 模型多分类训练模块，支持训练与验证 F1 评估与 early stopping。
# Author: msy
# Date: 2025

import torch
import os
import torch.nn.functional as F
from inf.model.tcp.TFC2016.multiclass.model import FTEncoder,MLPClassifier
from sklearn.metrics import f1_score

# ======== 模型训练函数 ========
def train_model(train_loader, val_loader,
                input_channels, freq_dim,
                num_classes, hidden_dim,
                device, lr=1e-4, epochs=20,
                pretrained_encoder_path=None,
                pretrained_classifier_path=None,
                model_save_path=None,
                classifier_save_path=None,
                early_stop_patience=10,
                no_improve_epochs_init=0):

    """
    输入：
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        input_channels: 输入通道数（如 12）
        freq_dim: 每通道频谱长度（如 32）
        num_classes: 多分类类别数
        hidden_dim: 编码器输出维度
        device: 运算设备
        lr: 学习率
        epochs: 训练轮数
    输出：
        encoder, classifier：训练完成的模型
    """
    no_improve_epochs = no_improve_epochs_init
    encoder = FTEncoder(input_shape=(input_channels, freq_dim), d_model=hidden_dim).to(device)
    classifier = MLPClassifier(input_dim=hidden_dim, num_classes=num_classes).to(device)

    # 加载已有参数（如果提供）
    if pretrained_encoder_path and os.path.exists(pretrained_encoder_path):
        encoder.load_state_dict(torch.load(pretrained_encoder_path, map_location=device, weights_only=True))
    if pretrained_classifier_path and os.path.exists(pretrained_classifier_path):
        classifier.load_state_dict(torch.load(pretrained_classifier_path, map_location=device, weights_only=True))


    # 定义优化器（模型加载之后）
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # 初始准确率评估
    # init_acc = evaluate_val(encoder, classifier, val_loader, device)
    # 初始macro F1评估
    init_f1 = evaluate_val(encoder, classifier, val_loader, device)
    print(f"[Init] 验证 Macro F1（加载参数后）: {init_f1:.4f}")

    best_f1 = init_f1
    best_encoder = encoder.state_dict()
    best_classifier = classifier.state_dict()



    for epoch in range(1, epochs + 1):
        encoder.train()
        classifier.train()
        total_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            embeddings = encoder(X)
            logits = classifier(embeddings)
            loss = criterion(logits, y)

            if torch.isnan(loss):
                print("出现 NaN，跳过该 batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        val_f1 = evaluate_val(encoder, classifier, val_loader, device)
        print(f"[Epoch {epoch}/{epochs}] Loss: {total_loss:.4f} | Val Macro F1: {val_f1:.4f}")


        if val_f1 > best_f1:
            best_f1 = val_f1
            best_encoder = encoder.state_dict()
            best_classifier = classifier.state_dict()

            print(f"   新最优模型，保存中... Val Macro F1: {val_f1:.4f}")
            if model_save_path:
                torch.save(best_encoder, model_save_path)
            if classifier_save_path:
                torch.save(best_classifier, classifier_save_path)
            
            no_improve_epochs = 0  # 重置无提升轮数
        else:
            no_improve_epochs += 1
            print(f"   未提升，EarlyStop计数：{no_improve_epochs}/{early_stop_patience}")
            if no_improve_epochs >= early_stop_patience:
                print(" 触发EarlyStopping，提前终止训练。")
                break



    if best_encoder is not None:
        encoder.load_state_dict(torch.load(pretrained_encoder_path, map_location=device, weights_only=True))
        classifier.load_state_dict(torch.load(pretrained_classifier_path, map_location=device, weights_only=True))


    return encoder, classifier

# ======== 验证评估函数 ========
def evaluate_val(encoder, classifier, loader, device):
    """
    验证模型在验证集上的 macro F1 分数
    """
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

    # macro F1-score：每类等权重平均
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return macro_f1
