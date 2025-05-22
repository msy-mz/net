# tls.py

import os
from scapy.all import RawPcapReader, PcapReader, Ether, IP, TCP, Raw
from scapy.layers.l2 import CookedLinux, Dot3
from collections import defaultdict, Counter

def extract_flows_from_pcap(pcap_path,
                             split_gap_threshold=30.0,
                             max_flow_duration=300.0):
    flows = defaultdict(list)
    seq_seen = defaultdict(set)
    total_pkt, parsed_pkt = 0, 0
    skipped_non_ipv4, skipped_no_payload, failed_pkt = 0, 0, 0

    try:
        with PcapReader(pcap_path) as pr:
            _ = pr.read_packet()
            link_type = pr.linktype
    except Exception as e:
        print(f"无法读取文件头：{e}")
        return flows

    use_sll = (link_type == 113)
    use_dot3 = (link_type == 1)

    last_seen = defaultdict(lambda: 0.0)
    flow_start_time = defaultdict(lambda: 0.0)
    flow_count = defaultdict(int)

    try:
        for pkt_data, pkt_metadata in RawPcapReader(pcap_path):
            total_pkt += 1
            try:
                pkt = None
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

                if not pkt.haslayer(IP):
                    skipped_non_ipv4 += 1
                    continue

                ip = pkt[IP]
                if not ip.haslayer(TCP) or not ip.haslayer(Raw):
                    skipped_no_payload += 1
                    continue

                tcp = ip[TCP]
                payload = bytes(tcp[Raw])
                if len(payload) == 0:
                    skipped_no_payload += 1
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

                flows[fid].append((seq, payload))
                parsed_pkt += 1

            except Exception as e:
                failed_pkt += 1
                if failed_pkt <= 5:
                    print(f"解析失败：{e}")
                continue

    except Exception as e:
        print(f"无法读取 {pcap_path}，错误：{e}")

    print("\n报文统计：")
    print(f"  总包数       ：{total_pkt}")
    print(f"  成功解析     ：{parsed_pkt}")
    print(f"  跳过非IPv4   ：{skipped_non_ipv4}")
    print(f"  跳过无载荷   ：{skipped_no_payload}")
    print(f"  解析失败     ：{failed_pkt}")
    print(f"  有效流数量   ：{len(flows)}")

    return flows


def is_tls_payload(payload: bytes) -> bool:
    """
    拼接后的流载荷判断是否为 TLS（典型握手或应用数据）
    """
    return (
        len(payload) >= 5 and
        payload[0] in [0x16, 0x17] and
        payload[1] == 0x03 and
        payload[2] in [0x01, 0x02, 0x03, 0x04]  # 支持 TLS 1.0 - 1.3+
    )


def save_payloads(flows, output_dir, max_payload_len=1024):
    """
    保存拼接后的 payload 为 bin 文件，且只保留 TLS 加密流（基于拼接后首字节判断）
    """
    os.makedirs(output_dir, exist_ok=True)
    file_count = 0
    total_bytes = 0
    filtered_count = 0

    for fid, pkt_list in flows.items():
        pkt_list.sort(key=lambda x: x[0])
        payload = b"".join([p for _, p in pkt_list])
        if len(payload) == 0:
            continue

        #  判断是否为 TLS 加密流
        if not is_tls_payload(payload):
            filtered_count += 1
            continue

        payload = payload[:max_payload_len]
        fname = os.path.join(output_dir, f"{fid}.bin")
        with open(fname, "wb") as f:
            f.write(payload)

        file_count += 1
        total_bytes += len(payload)

    print(f"\nTLS 流筛选完成：总流数 = {len(flows)}，TLS流 = {file_count}，非TLS流 = {filtered_count}")
    return file_count, total_bytes
