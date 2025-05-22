# Filename: encrypted_payload_extract.py
# Description: 核心逻辑模块，提取加密协议流量载荷（TLS/QUIC/SSH/IPSec 等），TCP流按seq拼接，支持可选长度限制
# Author: msy
# Date: 2025

import os
from scapy.all import RawPcapReader, Ether, IP, TCP, UDP, Raw
from collections import defaultdict

# 判断一个报文是否属于加密协议流量
def is_encrypted_packet(ip, l4, encrypted_ports):
    proto = 'tcp' if isinstance(l4, TCP) else 'udp' if isinstance(l4, UDP) else None
    if not proto:
        return False

    ports = encrypted_ports.get(proto, [])
    if l4.dport in ports or l4.sport in ports:
        return True

    # TLS 报文特征：0x16 0x03 开头（仅 TCP）
    if isinstance(l4, TCP) and Raw in l4:
        payload = bytes(l4[Raw])
        if len(payload) >= 3 and payload[0] == 0x16 and payload[1] == 0x03:
            return True

    return False

# 提取加密协议流量，按TCP或UDP组织流，TCP流保留seq用于重组
def extract_encrypted_flows(pcap_path, encrypted_ports, min_len=None):
    flows = defaultdict(list)
    try:
        for pkt_data, _ in RawPcapReader(pcap_path):
            try:
                pkt = Ether(pkt_data)
                if not pkt.haslayer(IP):
                    continue

                ip = pkt[IP]
                if ip.haslayer(TCP):
                    l4 = ip[TCP]
                    proto = 'tcp'
                elif ip.haslayer(UDP):
                    l4 = ip[UDP]
                    proto = 'udp'
                else:
                    continue

                if not l4.haslayer(Raw):
                    continue

                if not is_encrypted_packet(ip, l4, encrypted_ports):
                    continue

                payload = bytes(l4[Raw])
                if min_len is not None and len(payload) < min_len:
                    continue

                fid = f"{ip.src}-{ip.dst}-{l4.sport}-{l4.dport}-{ip.proto}"
                if proto == 'tcp':
                    flows[fid].append((l4.seq, payload))  # TCP使用seq排序
                else:
                    flows[fid].append(payload)  # UDP直接拼接

            except Exception:
                continue

    except Exception:
        pass

    return flows

# 保存流载荷，TCP使用seq排序拼接，UDP直接拼接，支持不截断
def save_payloads(flows, out_folder, max_payload_len=None):
    os.makedirs(out_folder, exist_ok=True)
    file_count = 0
    total_bytes = 0

    for i, (fid, chunks) in enumerate(flows.items()):
        if chunks and isinstance(chunks[0], tuple):  # TCP流
            sorted_chunks = sorted(chunks, key=lambda x: x[0])
            payload = b''.join([c[1] for c in sorted_chunks])
        else:
            payload = b''.join(chunks)

        if len(payload) == 0:
            continue
        if max_payload_len is not None:
            payload = payload[:max_payload_len]

        safe_name = f"{fid.replace(':', '_').replace('/', '_')}_{i}.bin"
        path = os.path.join(out_folder, safe_name)
        with open(path, 'wb') as f:
            f.write(payload)
        file_count += 1
        total_bytes += len(payload)

    return file_count, total_bytes
