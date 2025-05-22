# Filename: tcp.py
# Path: inf/payload/extract/tcp.py
# Description: 
#   TCP 协议流量提取模块，支持多流聚合、截断保存，并生成 meta.json。
#   保留全部 TCP 流（不做协议内容筛选），依赖 base 工具模块完成保存与统计。
# Author: msy
# Date: 2025

import os
from scapy.all import RawPcapReader, PcapReader, Ether, IP, TCP, Raw
from scapy.layers.l2 import CookedLinux, Dot3
from collections import defaultdict

from inf.payload.extract.base import save_bin_payloads

# TCP 默认认为所有流都有效
def accept_all(_payload: bytes) -> bool:
    return True

# 提取 TCP 流（聚合、去重、方向无关）
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

# TCP 提取主函数（返回 bin 文件路径 + stats）
def extract_with_stats(pcap_path, output_dir, max_len=1024):
    flows = extract_flows_from_pcap(pcap_path)
    return save_bin_payloads(flows, output_dir, is_valid_fn=accept_all, max_payload_len=max_len)

# 简化版，仅返回 bin 文件路径
def extract_payloads_from_pcap(pcap_path, output_dir, max_len=1024):
    bin_files, _ = extract_with_stats(pcap_path, output_dir, max_len=max_len)
    return bin_files
