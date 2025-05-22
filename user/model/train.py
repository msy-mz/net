# Filename: train.py
# Path: user/train/train.py
# Description: 从 YAML 加载参数的训练主控模块
# Author: msy
# Date: 2025

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from user.model.encoder import FingerprintEncoder
from user.model.tcn import IDTCNModel
from user.infer.infer import IdentityClassifier

# 加载 YAML 配置
def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 示例数据集（需替换）
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n=1000, T=10, D=64, num_classes=10):
        self.x = torch.randn(n, T, D)
        self.y = torch.randint(0, num_classes, (n,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 主训练逻辑
def train(cfg):
    dataset = DummyDataset(D=cfg["input_dim"], num_classes=cfg["num_classes"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    encoder = FingerprintEncoder(cfg["input_dim"], cfg["encoder_hidden_dim"], cfg["encoded_dim"])
    model = IDTCNModel(cfg["encoded_dim"], cfg["tcn_hidden_dim"], cfg["embed_dim"],
                       levels=cfg["tcn_levels"], kernel_size=cfg["kernel_size"])
    classifier = IdentityClassifier(cfg["embed_dim"], cfg["num_classes"])

    optimizer = optim.Adam(list(encoder.parameters()) + list(model.parameters()) + list(classifier.parameters()),
                           lr=cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg["epochs"]):
        total_loss = 0
        for x, y in dataloader:
            x_encoded = encoder(x)
            h = model(x_encoded)
            logits = classifier(h)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1:02d}, Loss: {total_loss:.4f}")

    torch.save({
        'encoder': encoder.state_dict(),
        'model': model.state_dict(),
        'classifier': classifier.state_dict()
    }, cfg["save_path"])
    print(f"模型已保存到 {cfg['save_path']}")

if __name__ == "__main__":
    config = load_config("config/train.yaml")
    train(config)
