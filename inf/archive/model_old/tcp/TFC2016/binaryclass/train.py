# train.py
# author: msy
# description: FT-Encoder++ 模型训练模块，联合训练编码器与分类器，用于加密流量多分类任务

import torch
from model import FTEncoder
from model import MLPClassifier
import torch.nn.functional as F

def train_model(train_loader, val_loader,
                input_channels, freq_dim,
                num_classes, hidden_dim,
                device, lr=1e-4, epochs=20):   # ← 添加 epochs 参数

    """
    构建 FT-Encoder + 分类器模型，联合训练
    参数：
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        input_channels: 输入通道数（通常为 12）
        freq_dim: 频谱长度（通常为 32）
        num_classes: 分类数（任务目标）
        hidden_dim: 编码器输出维度
        device: 'cuda' 或 'cpu'
        lr: 学习率（默认 1e-4）
    返回：
        encoder, classifier：训练后的两个模型
    """
    encoder = FTEncoder(input_shape=(input_channels, freq_dim), d_model=hidden_dim).to(device)
    classifier = MLPClassifier(input_dim=hidden_dim, num_classes=num_classes).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0
    best_encoder = None
    best_classifier = None

    for epoch in range(1, epochs + 1):
        encoder.train()
        classifier.train()
        total_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            # 前向传播
            embeddings = encoder(X)
            logits = classifier(embeddings)
            loss = criterion(logits, y)

            # 避免 NaN 扰乱训练
            if torch.isnan(loss):
                print("出现 NaN，跳过该 batch")
                continue

            # 反向传播 + 梯度裁剪
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        # 验证阶段
        acc = evaluate_val(encoder, classifier, val_loader, device)
        print(f"[Epoch {epoch}] Loss: {total_loss:.4f} | Val Acc: {acc:.4f}")

        # 记录最优模型
        if acc > best_acc:
            best_acc = acc
            best_encoder = encoder.state_dict()
            best_classifier = classifier.state_dict()

    # 如果没有更好模型，仍需返回当前模型参数
    if best_encoder is not None:
        encoder.load_state_dict(best_encoder)
        classifier.load_state_dict(best_classifier)

    return encoder, classifier

def evaluate_val(encoder, classifier, loader, device):
    """
    验证模型在验证集上的准确率
    """
    encoder.eval()
    classifier.eval()
    correct = total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = classifier(encoder(X)).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total if total > 0 else 0.0
