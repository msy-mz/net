# Filename: infer_single.py
# Path: inf/infer/infer_single.py
# Description: 单个 PCAP 推理主流程，返回结构化结果字典
# Author: msy
# Date: 2025

import os
import json
import yaml
import numpy as np
from collections import Counter

from inf.infer.infer import load_models, predict_class
from inf.payload.extract.tcp import extract_payloads_from_pcap
from inf.payload.feature import extract_feature_from_bytes
from inf.utils.normalize import normalize_log1p
from inf.utils.vis import spectral_centroid

def run_single_infer(config_path, pcap_path):
    """
    对单个 PCAP 文件执行推理，返回结构化结果列表与统计信息。

    参数:
        config_path: 推理配置文件路径
        pcap_path: 单个 PCAP 文件路径

    返回:
        result_dict: {
            "summary": {...},
            "detailed": [...],
            "features": [feature1, feature2, ...]
        }
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    OUTPUT_DIR = cfg['output_dir']
    MAX_PAYLOAD_LEN = cfg['max_payload_len']
    MAX_FLOWS = cfg.get('max_flows', None)
    TRAIN_CFG_PATH = cfg['config_path']
    NORMAL_RANGE = range(cfg['normal_class_range'][0], cfg['normal_class_range'][1] + 1)

    # 覆盖当前 PCAP 路径
    cfg['pcap_path'] = pcap_path

    # 加载训练配置
    with open(TRAIN_CFG_PATH, 'r') as f:
        full_cfg = yaml.safe_load(f)

    # 只提取推理所需字段
    train_cfg = {
        'input_channels': full_cfg['input_channels'],
        'freq_dim': full_cfg['freq_dim'],
        'hidden_dim': full_cfg['hidden_dim'],
        'device': full_cfg.get('device', 'cuda'),
        'pretrained_encoder_path': full_cfg['pretrained_encoder_path'],
        'pretrained_classifier_path': full_cfg['pretrained_classifier_path'],
        'label_map_path': full_cfg['label_map_path']
    }


    with open(train_cfg['label_map_path'], 'r') as f:
        label_map = json.load(f)
        num_classes = len(label_map)

    encoder, classifier = load_models(
        input_channels=train_cfg['input_channels'],
        freq_dim=train_cfg['freq_dim'],
        hidden_dim=train_cfg['hidden_dim'],
        num_classes=num_classes,
        encoder_path=train_cfg['pretrained_encoder_path'],
        classifier_path=train_cfg['pretrained_classifier_path'],
        device=train_cfg.get('device', 'cuda')
    )


    # 提取 payload
    bin_files = extract_payloads_from_pcap(
        pcap_path=pcap_path,
        output_dir=OUTPUT_DIR,
        max_len=MAX_PAYLOAD_LEN,
        max_flows=MAX_FLOWS
    )

    if not bin_files:
        raise RuntimeError(f"未提取到任何流：{pcap_path}，请检查 PCAP 文件格式或内容")

    # 加载 meta 信息
    meta_path = os.path.join(OUTPUT_DIR, 'meta.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    with open(train_cfg['label_map_path'], 'r') as f:
        name_to_idx = json.load(f)
    class_names = {v: k for k, v in name_to_idx.items()}
    name_to_idx = {v: k for k, v in class_names.items()}
    normal_class_ids = set(NORMAL_RANGE)

    detailed_results = []
    label_counter = Counter()
    abnormal_count = 0

    for bin_path in bin_files:
        fname = os.path.basename(bin_path)
        with open(bin_path, 'rb') as f:
            payload_bytes = f.read()

        feature = extract_feature_from_bytes(payload_bytes)
        feature = normalize_log1p(feature)

        pred = predict_class(feature, encoder, classifier, class_names, device=train_cfg.get('device', 'cuda'))
        mean_energy = float(np.sum(feature))
        mean_centroid, centroids = spectral_centroid(feature)
        mean_per_channel = feature.mean(axis=1).tolist()
        std_per_channel = feature.std(axis=1).tolist()

        record = meta.get(fname, {})
        is_abnormal = name_to_idx[pred] not in normal_class_ids
        if is_abnormal:
            abnormal_count += 1

        record.update({
            "filename": fname,
            "label": pred,
            "is_abnormal": is_abnormal,
            "mean_energy": mean_energy,
            "mean_centroid": mean_centroid,
            "centroids": centroids,
            "mean_per_channel": mean_per_channel,
            "std_per_channel": std_per_channel,
            "spectrum": feature.tolist()
        })

        detailed_results.append(record)
        label_counter[pred] += 1

        os.remove(bin_path)

    return {
        "summary": {
            "total_flows": len(detailed_results),
            "abnormal_flows": abnormal_count,
            "label_distribution": dict(label_counter)
        },
        "detailed": detailed_results
    }
