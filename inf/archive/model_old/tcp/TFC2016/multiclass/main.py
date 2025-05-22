# main.py
# 作者：msy
# 日期：2025
# 说明：训练 FT-Encoder++ + 多分类器，执行加密流量多分类任务

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 加入项目根路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inf.model.tcp.TFC2016.multiclass.train import train_model
from inf.model.tcp.TFC2016.multiclass.evaluate import evaluate_model
from inf.model.tcp.TFC2016.multiclass.model import FTEncoder, MLPClassifier

# ======== 参数配置 ========
DATA_PATH = "data/npz/payload_feature/tcp/USTC-TFC2016/all/multiclass/USTC-TFC2016_all_mulClass_normalized.npz"
LABEL_MAP_PATH = "data/npz/payload_feature/tcp/USTC-TFC2016/all/multiclass/USTC-TFC2016_all_mulClass_label.json"
MODEL_SAVE_PATH = "inf/model/tcp/multiclass/model_ft_multiclass.pt"
CLASSIFIER_SAVE_PATH = "inf/model/tcp/multiclass/model_classifier_multiclass.pt"
FIG_SAVE_DIR = "inf/report/evaluate_multiclass_tcp"


INPUT_CHANNELS = 12
FREQ_DIM = 32
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 20
EARLY_STOP_PATIENCE = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ======== Step 1: 加载数据 ========
def load_data(path):
    data = np.load(path)
    X = torch.tensor(data["X"], dtype=torch.float)
    y = torch.tensor(data["y"], dtype=torch.long)
    return X, y

# ======== Step 2: 加载类别名 ========
def load_class_names(json_path):
    import json
    with open(json_path, "r") as f:
        label_map = json.load(f)
    class_names = [""] * len(label_map)
    for name, idx in label_map.items():
        class_names[idx] = name
    return class_names

# ======== Step 3: 构建数据加载器 ========
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

# ======== 主函数入口 ========
def main():
    print("加载数据...")
    X, y = load_data(DATA_PATH)
    class_names = load_class_names(LABEL_MAP_PATH)
    num_classes = len(class_names)

    train_loader, val_loader = build_dataloaders(X, y)

    print("启动模型训练...")
    encoder, classifier = train_model(
    train_loader, val_loader,
    input_channels=INPUT_CHANNELS,
    freq_dim=FREQ_DIM,
    num_classes=num_classes,
    hidden_dim=HIDDEN_DIM,
    device=DEVICE,
    epochs=EPOCHS,
    early_stop_patience=EARLY_STOP_PATIENCE,
    pretrained_encoder_path=MODEL_SAVE_PATH,
    pretrained_classifier_path=CLASSIFIER_SAVE_PATH,
    model_save_path=MODEL_SAVE_PATH,
    classifier_save_path=CLASSIFIER_SAVE_PATH
)
    
    print("执行模型评估...")
    evaluate_model(encoder, classifier, val_loader, class_names=class_names, device=DEVICE, save_dir=FIG_SAVE_DIR)


    print("模型保存中...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(encoder.state_dict(), MODEL_SAVE_PATH)
    torch.save(classifier.state_dict(), CLASSIFIER_SAVE_PATH)
    print("模型已保存。")

if __name__ == "__main__":
    main()
