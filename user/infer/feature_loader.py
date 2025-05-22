# Filename: feature_loader.py
# Path: user/infer/feature_loader.py
# Description: 加载特征 CSV 与提取连接五元组元数据（含时间戳）。
# Author: msy
# Date: 2025

import csv
from datetime import datetime
from scapy.all import PcapReader, IP, TCP

# 加载 CSV 文件中的连接特征与连接标识符
def load_features(csv_path):
    features = []
    conn_ids = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            feature_vector = [
                float(row['ttl']),
                float(row['tcp_window']),
                float(row['mss']) if row['mss'] else 0.0,
                float(row['ttl_variance']),
                float(row['src_port_entropy']),
                float(row['has_tls']),
            ]
            features.append(feature_vector)
            conn_ids.append(row['conn_id'])
    return features, conn_ids

# 提取 PCAP 中每个连接的五元组与时间戳信息
def extract_conn_metadata(pcap_path):
    meta = {}
    with PcapReader(pcap_path) as pcap:
        for pkt in pcap:
            if IP in pkt and TCP in pkt:
                ip = pkt[IP]
                tcp = pkt[TCP]
                conn_id = f"{ip.src}-{ip.dst}-{tcp.sport}-{tcp.dport}-6"
                ts = datetime.fromtimestamp(pkt.time).isoformat()
                meta[conn_id] = {
                    "src_ip": ip.src,
                    "dst_ip": ip.dst,
                    "src_port": tcp.sport,
                    "dst_port": tcp.dport,
                    "proto": 6,
                    "timestamp": ts
                }
    return meta
