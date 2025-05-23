# Filename: infer_live.py
# Path: inf/infer/infer_live.py
# Description: 单条 TCP Payload 实时推理模块，返回分类与频谱特征等结构化信息（配置化版本）
# Author: msy
# Date: 2025

import yaml
import json
import numpy as np

from inf.infer.infer import load_models, predict_class
from inf.payload.feature import extract_feature_from_bytes
from inf.utils.normalize import normalize_log1p
from inf.utils.vis import spectral_centroid

# === 配置路径 ===
MODEL_CONFIG_PATH = 'inf/runner/config/ustc2016/tcp/multiclass/train.yaml'

# === 加载配置 ===
with open(MODEL_CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

# === 加载标签映射 ===
with open(cfg['label_map_path'], 'r') as f:
    label_map = json.load(f)

# === 构建标签映射关系 ===
class_names = {v: k for k, v in label_map.items()}
name_to_idx = {v: k for k, v in class_names.items()}
num_classes = len(class_names)

# === 加载模型 ===
encoder, classifier = load_models(
    input_channels=cfg['input_channels'],
    freq_dim=cfg['freq_dim'],
    hidden_dim=cfg['hidden_dim'],
    num_classes=num_classes,
    encoder_path=cfg['pretrained_encoder_path'],
    classifier_path=cfg['pretrained_classifier_path'],
    device=cfg.get('device', 'cuda')
)

# === 正常类别编号集合 ===
normal_class_ids = {name_to_idx[name] for name in cfg['normal_labels']}

def infer_payload(payload_bytes):
    """
    对单条 TCP Payload 执行特征提取与推理
    返回结构化推理结果字典
    """
    # 特征提取与归一化
    feature = extract_feature_from_bytes(payload_bytes)
    feature = normalize_log1p(feature)

    # 分类预测
    label = predict_class(
        feature,
        encoder=encoder,
        classifier=classifier,
        class_names=class_names,
        device=cfg.get('device', 'cuda')
    )

    # 提取结构化特征指标
    mean_energy = float(np.sum(feature))
    mean_centroid, centroids = spectral_centroid(feature)
    mean_per_channel = feature.mean(axis=1).tolist()
    std_per_channel = feature.std(axis=1).tolist()

    # 构建返回结果
    return {
        "label": label,
        "is_abnormal": name_to_idx[label] not in normal_class_ids,
        "mean_energy": mean_energy,
        "mean_centroid": mean_centroid,
        "centroids": centroids,
        "mean_per_channel": mean_per_channel,
        "std_per_channel": std_per_channel,
        "spectrum": feature.tolist()
    }
