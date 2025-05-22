# main.py
# author: msy
# description: 训练 FT-Encoder++ + 分类器，执行加密流量二分类任务

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train import train_model
from inf.evaluate import evaluate_model
from model import FTEncoder, MLPClassifier  

# === 参数配置 ===
DATA_PATH = "inf/data/npz/tcp/USTC-TFC2016_all_npz_normalized.npz"
MODEL_SAVE_PATH = "inf/model/tcp/two_class/model_ft_tcp.pt"
CLASSIFIER_SAVE_PATH = "inf/model/tcp/two_class/model_classifier_tcp.pt"

INPUT_CHANNELS = 12
FREQ_DIM = 32
HIDDEN_DIM = 128
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASS_NAMES = ["Benign", "Malware"]

# === Step 1: 加载数据 ===
def load_data(path):
    data = np.load(path)
    X = torch.tensor(data["X"], dtype=torch.float)
    y = torch.tensor(data["y"], dtype=torch.long)
    return X, y

# === Step 2: 构建数据加载器 ===
def build_dataloaders(X, y, train_ratio=0.8):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=1 - train_ratio, stratify=y, random_state=42
    )

    train_set = TensorDataset(X_train.clone().detach().float(),
                              y_train.clone().detach().long())
    val_set = TensorDataset(X_val.clone().detach().float(),
                            y_val.clone().detach().long())

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    return train_loader, val_loader

# === 主流程 ===
def main():
    print("加载数据...")
    X, y = load_data(DATA_PATH)
    train_loader, val_loader = build_dataloaders(X, y)

    print("启动模型训练...")
    encoder, classifier = train_model(
        train_loader, val_loader,
        input_channels=INPUT_CHANNELS,
        freq_dim=FREQ_DIM,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        device=DEVICE,
        epochs=EPOCHS  # ✅ 明确传入训练轮次
    )

    print("执行模型评估...")
    evaluate_model(encoder, classifier, val_loader, class_names=CLASS_NAMES, device=DEVICE)

    print("模型保存中...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(encoder.state_dict(), MODEL_SAVE_PATH)
    torch.save(classifier.state_dict(), CLASSIFIER_SAVE_PATH)
    print("模型已保存。")

if __name__ == "__main__":
    main()
