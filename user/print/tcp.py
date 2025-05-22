# Filename: tcp.py
# Path: user/print/tcp.py
# Description: 提取 TCP 报文的连接标识符和协议元信息，如 TTL、MSS、窗口大小等。
# Author: msy
# Date: 2025

import hashlib
import numpy as np
from collections import Counter
from scapy.all import IP, TCP

# 构建连接标识符（五元组）
def build_conn_id(src_ip, dst_ip, src_port, dst_port, proto=6):
    return f"{src_ip}-{dst_ip}-{src_port}-{dst_port}-{proto}"

# 计算字符串的 MD5 哈希
def hash_str(s):
    return hashlib.md5(s.encode()).hexdigest() if s else ""

# 计算熵，用于衡量端口等特征的离散性
def calc_entropy(values):
    counter = Counter(values)
    probs = np.array(list(counter.values())) / len(values)
    return float(-np.sum(probs * np.log2(probs))) if probs.size > 0 else 0.0

# 从单个 TCP 报文中提取协议字段特征
def extract_tcp_metadata(pkt):
    ip, tcp = pkt[IP], pkt[TCP]
    options = tcp.options if tcp.options else []
    opt_names = ",".join([str(opt[0]) for opt in options])
    mss = next((opt[1] for opt in options if opt[0] == "MSS"), None)

    return {
        "src_ip": ip.src,
        "dst_ip": ip.dst,
        "src_port": tcp.sport,
        "dst_port": tcp.dport,
        "ttl": ip.ttl,
        "tcp_window": tcp.window,
        "mss": mss or "",
        "tcp_options": opt_names,
        "tcp_options_hash": hash_str(opt_names)
    }
