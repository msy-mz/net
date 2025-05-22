# Filename: encrypted.py
# Path: inf/payload/extract/encrypted.py
# Description: 提取加密协议流量（TLS/QUIC/SSH/IPSec 等），按统一结构组织
# Author: msy
# Date: 2025

from scapy.all import RawPcapReader, Ether, IP, TCP, UDP, Raw
from collections import defaultdict
from inf.payload.extract.base import save_bin_payloads

# 判断一个报文是否为加密协议（加密端口或 TLS 特征）
def is_encrypted_packet(ip, l4, encrypted_ports):
    proto = 'tcp' if isinstance(l4, TCP) else 'udp' if isinstance(l4, UDP) else None
    if not proto:
        return False

    ports = encrypted_ports.get(proto, [])
    if l4.dport in ports or l4.sport in ports:
        return True

    if isinstance(l4, TCP) and Raw in l4:
        payload = bytes(l4[Raw])
        if len(payload) >= 3 and payload[0] == 0x16 and payload[1] == 0x03:
            return True

    return False

# 提取加密协议流，返回标准流格式 {fid: [(seq, payload, timestamp)]}
def extract_encrypted_flows(pcap_path, encrypted_ports, min_len=None):
    flows = defaultdict(list)
    try:
        for pkt_data, pkt_meta in RawPcapReader(pcap_path):
            try:
                pkt = Ether(pkt_data)
                if not pkt.haslayer(IP):
                    continue
                ip = pkt[IP]
                ts = float(pkt_meta.sec) + pkt_meta.usec / 1e6

                if ip.haslayer(TCP):
                    l4 = ip[TCP]
                    proto = 'tcp'
                    seq = l4.seq
                elif ip.haslayer(UDP):
                    l4 = ip[UDP]
                    proto = 'udp'
                    seq = len(flows)  # UDP 无序，用编号替代
                else:
                    continue

                if not l4.haslayer(Raw):
                    continue
                if not is_encrypted_packet(ip, l4, encrypted_ports):
                    continue

                payload = bytes(l4[Raw])
                if min_len and len(payload) < min_len:
                    continue

                fid = f"{ip.src}-{ip.dst}-{l4.sport}-{l4.dport}-{proto}"
                flows[fid].append((seq, payload, ts))

            except Exception:
                continue
    except Exception:
        pass

    return flows

# 恒为 True（所有加密流都认为有效）
def accept_all(payload: bytes) -> bool:
    return True

# 提取 + 保存（生成 bin 文件与 meta）
def extract_with_stats(pcap_path, output_dir, encrypted_ports, min_len=None, max_len=1024):
    flows = extract_encrypted_flows(pcap_path, encrypted_ports, min_len=min_len)
    return save_bin_payloads(flows, output_dir, is_valid_fn=accept_all, max_payload_len=max_len)

# 简化接口
def extract_payloads_from_pcap(pcap_path, output_dir, encrypted_ports, max_len=1024):
    bin_files, _ = extract_with_stats(pcap_path, output_dir, encrypted_ports, max_len=max_len)
    return bin_files
