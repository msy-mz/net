# Filename: predict_dict.py
# Path: inf/scripts/inference/predict_dict.py
# Description: 从 features_dict.npz + meta.json 加载特征，执行批量推理，输出每条流的预测结果及五元组信息。
# Author: msy
# Date: 2025

import os
import json
import argparse
import numpy as np
import torch
from inf.inference.inference import load_models

# ======== 模型配置路径 ========
CONFIG_PATH = "inf/config/ustc2016/tcp/multiclass/train.yaml"

# ======== 加载模型配置和权重 ========
def load_predictor():
    import yaml
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    with open(cfg["label_map_path"], "r") as f:
        label_map = json.load(f)
    class_names = {v: k for k, v in label_map.items()}

    encoder, classifier = load_models(
        input_channels=cfg["input_channels"],
        freq_dim=cfg["freq_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_classes=len(class_names),
        encoder_path=cfg["pretrained_encoder_path"],
        classifier_path=cfg["pretrained_classifier_path"],
        device=cfg.get("device", "cpu")
    )

    return encoder, classifier, class_names, cfg.get("device", "cpu")

# ======== 主推理函数 ========
def predict_batch(feature_dict, meta_dict, encoder, classifier, class_names, device):
    results = []

    for fname, feature in feature_dict.items():
        x = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            z = encoder(x)
            logits = classifier(z)
            probs = torch.softmax(logits, dim=1).squeeze()
            pred_idx = torch.argmax(probs).item()

        meta = meta_dict.get(fname, {})
        results.append({
            "file": fname,
            "label": class_names[pred_idx],
            "confidence": round(probs[pred_idx].item(), 4),
            "src_ip": meta.get("src_ip", "N/A"),
            "dst_ip": meta.get("dst_ip", "N/A"),
            "src_port": meta.get("src_port", "N/A"),
            "dst_port": meta.get("dst_port", "N/A"),
            "timestamp": meta.get("timestamp", "N/A")
        })

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量推理：从 dict.npz 和 meta.json 中加载并执行模型分类")
    parser.add_argument("--input-dir", required=True, help="输入目录，包含 features_dict.npz 和 meta.json")
    args = parser.parse_args()

    feature_path = os.path.join(args.input_dir, "features_dict.npz")
    meta_path = os.path.join(args.input_dir, "meta.json")

    if not os.path.exists(feature_path) or not os.path.exists(meta_path):
        print("[错误] 缺少必要文件：features_dict.npz 或 meta.json")
        exit(1)

    data = np.load(feature_path)
    with open(meta_path, "r") as f:
        meta_dict = json.load(f)

    encoder, classifier, class_names, device = load_predictor()

    results = predict_batch(data, meta_dict, encoder, classifier, class_names, device)

    print(f"\n[预测结果] 共 {len(results)} 条：\n")
    for r in results:
        print(f"[{r['label']}] {r['confidence']}  ← {r['src_ip']}:{r['src_port']} → {r['dst_ip']}:{r['dst_port']} @ {r['timestamp']}")
