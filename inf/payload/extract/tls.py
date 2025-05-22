# Filename: tls.py
# Path: inf/payload/extract/tls.py
# Description: 
#   TLS 协议流量提取模块，解析 PCAP 中的 TCP 流，并筛选为有效 TLS 流。
#   保存为 bin 文件并生成 meta.json，由 base 工具模块处理保存与统计。
# Author: msy
# Date: 2025

import os
from scapy.all import RawPcapReader, PcapReader, Ether, IP, TCP, Raw
from scapy.layers.l2 import CookedLinux, Dot3
from collections import defaultdict

from inf.payload.extract.base import save_bin_payloads

# 判断流是否为 TLS 流（基于典型 TLS 握手首字节）
def is_tls_payload(payload: bytes) -> bool:
    return (
        len(payload) >= 5 and
        payload[0] in [0x16, 0x17] and
        payload[1] == 0x03 and
        payload[2] in [0x01, 0x02, 0x03, 0x04]
    )

# 提取 TLS 相关流（仅聚合，不保存）
def extract_flows_from_pcap(pcap_path, 
                            split_gap_threshold=30.0, 
                            max_flow_duration=300.0):
    flows = defaultdict(list)
    seq_seen = defaultdict(set)

    try:
        with PcapReader(pcap_path) as pr:
            _ = pr.read_packet()
            link_type = pr.linktype
    except Exception as e:
        print(f"[ERROR] 读取 PCAP 头失败：{e}")
        return flows

    use_sll = (link_type == 113)
    use_dot3 = (link_type == 1)

    last_seen = defaultdict(lambda: 0.0)
    flow_start_time = defaultdict(lambda: 0.0)
    flow_count = defaultdict(int)

    for pkt_data, pkt_metadata in RawPcapReader(pcap_path):
        try:
            if use_sll:
                pkt = CookedLinux(pkt_data)
            else:
                try:
                    pkt = Ether(pkt_data)
                except Exception:
                    try:
                        pkt = Dot3(pkt_data)
                    except Exception:
                        from scapy.layers.inet import IP as IP_Only
                        pkt = IP_Only(pkt_data)

            if not pkt.haslayer(IP) or not pkt.haslayer(TCP) or not pkt.haslayer(Raw):
                continue

            ip = pkt[IP]
            tcp = ip[TCP]
            payload = bytes(tcp[Raw])
            if len(payload) == 0:
                continue

            ts = float(pkt_metadata.sec) + pkt_metadata.usec / 1e6
            fid_base = f"{ip.dst}-{ip.src}-{tcp.dport}-{tcp.sport}-6"

            gap = ts - last_seen[fid_base]
            if gap > split_gap_threshold or (ts - flow_start_time[fid_base] > max_flow_duration):
                flow_count[fid_base] += 1
                flow_start_time[fid_base] = ts

            last_seen[fid_base] = ts
            fid = f"{fid_base}__{flow_count[fid_base]}"

            seq = tcp.seq
            if seq in seq_seen[fid]:
                continue
            seq_seen[fid].add(seq)

            flows[fid].append((seq, payload, ts))

        except Exception:
            continue

    return flows

# TLS 提取主函数（返回 bin 文件路径 + stats）
def extract_with_stats(pcap_path, output_dir, max_len=1024):
    flows = extract_flows_from_pcap(pcap_path)
    return save_bin_payloads(flows, output_dir, is_valid_fn=is_tls_payload, max_payload_len=max_len)

# 简化版，仅返回 bin 文件路径
def extract_payloads_from_pcap(pcap_path, output_dir, max_len=1024):
    bin_files, _ = extract_with_stats(pcap_path, output_dir, max_len=max_len)
    return bin_files
