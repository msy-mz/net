# Filename: extract.py
# Path: user/print/extract.py
# Description: 网络流量指纹主控模块，调度特征提取与输出逻辑。
# Author: msy
# Date: 2025

from scapy.all import PcapReader, IP, TCP, Raw
from collections import defaultdict
import numpy as np

from user.print.tcp import build_conn_id, extract_tcp_metadata, calc_entropy
from user.print.tls import parse_tls_client_hello, parse_ja3
from user.print.writer import write_csv

# 提取 TLS 和 TCP 指纹信息，输出结构化数据
def extract_fingerprint(pcap_path, output_csv, label=None, max_packets=None):
    rows = []
    tcp_meta = {}
    tls_info = {}
    conn_packets = defaultdict(list)

    with PcapReader(pcap_path) as pcap:
        for i, pkt in enumerate(pcap):
            if max_packets and i >= max_packets:
                break
            if IP in pkt and TCP in pkt:
                conn_id = build_conn_id(pkt[IP].src, pkt[IP].dst, pkt[TCP].sport, pkt[TCP].dport)
                conn_packets[conn_id].append(pkt)

                # 提取 TCP 元数据
                if conn_id not in tcp_meta:
                    tcp_meta[conn_id] = extract_tcp_metadata(pkt)
                    tcp_meta[conn_id]["conn_id"] = conn_id
                    tcp_meta[conn_id]["has_tls"] = 0
                    tcp_meta[conn_id]["ja3_fingerprint"] = ""

                # TLS Client Hello 检测与解析
                if Raw in pkt:
                    payload = bytes(pkt[Raw])
                    if len(payload) >= 5 and payload[0] == 0x16 and payload[5] == 0x01:
                        ciphers = parse_tls_client_hello(payload)
                        if ciphers:
                            tcp_meta[conn_id]["has_tls"] = 1
                            tls_info[conn_id] = {
                                "cipher_suites": ciphers,
                                "cipher_suite_top1": ciphers[0],
                                "cipher_hash": tcp_meta[conn_id]["tcp_options_hash"]
                            }
                        ja3_fingerprint = parse_ja3(payload)
                        if ja3_fingerprint:
                            tcp_meta[conn_id]["ja3_fingerprint"] = ja3_fingerprint

    # 分组计算统计量
    src_groups = defaultdict(list)
    for conn_id in tcp_meta:
        src_groups[tcp_meta[conn_id]["src_ip"]].append(conn_id)

    # 构造每一条连接的输出记录
    for conn_id, meta in tcp_meta.items():
        group = src_groups[meta["src_ip"]]
        ttl_vals = [tcp_meta[c]["ttl"] for c in group]
        src_ports = [tcp_meta[c]["src_port"] for c in group]
        tls = tls_info.get(conn_id, {})

        row = {
            **meta,
            "cipher_suites": ",".join([f"0x{b:02x}" for b in tls.get("cipher_suites", b"")]),
            "cipher_suite_top1": tls.get("cipher_suite_top1", ""),
            "cipher_hash": tls.get("cipher_hash", ""),
            "ttl_variance": float(np.std(ttl_vals)) if ttl_vals else 0.0,
            "src_port_entropy": calc_entropy(src_ports),
            "label": label
        }
        rows.append(row)

    # 定义字段顺序并写出结果
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(rows, output_csv, fieldnames)
    print(f"完成指纹提取：连接数={len(rows)}，TLS数={sum(1 for r in rows if r['has_tls'])}")
    return len(rows)
