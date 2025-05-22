# Filename: tcp.py
# Module: inf.payload.extract.tcp
# Description: 提取 TCP 报文流并保存其有效载荷
# Author: msy
# Date: 2025

import os
from scapy.all import RawPcapReader, PcapReader, Ether, IP, TCP, Raw, conf
from scapy.layers.l2 import CookedLinux, Dot3
from collections import defaultdict

# 封装 extract_flows_from_pcap 与 save_payloads，简化上层调用
def extract_payloads_from_pcap(pcap_path, save_dir):
    flows = extract_flows_from_pcap(pcap_path)
    return save_payloads(flows, save_dir)

# 从 PCAP 文件中提取所有 TCP 流，按五元组标识流并聚合 TCP 负载
def extract_flows_from_pcap(pcap_path):
    flows = defaultdict(list)  # 使用字典保存流的 payload 列表，按 fid（五元组）分组
    total_pkt = 0              # 总包数统计
    parsed_pkt = 0             # 成功解析的包数
    failed_pkt = 0             # 解析失败的包数
    skipped_non_ipv4 = 0       # 跳过的非 IPv4 报文数量
    skipped_no_payload = 0     # 跳过的无 TCP 负载报文数量

    # 自动判断链路层类型（Ethernet 或 Linux cooked capture）
    try:
        with PcapReader(pcap_path) as pr:
            _ = pr.read_packet()
            link_type = pr.linktype
    except Exception as e:
        print(f" 无法读取文件头：{e}")
        return flows

    use_sll = (link_type == 113)  # Linux cooked capture
    use_dot3 = (link_type == 1)   # Ethernet II

    try:
        for pkt_data, _ in RawPcapReader(pcap_path):
            total_pkt += 1
            try:
                # 根据链路类型解析数据包
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

                # 跳过非 IPv4 报文
                if not pkt.haslayer(IP):
                    skipped_non_ipv4 += 1
                    continue

                ip = pkt[IP]
                # 跳过无 TCP 或无有效载荷的报文
                if not ip.haslayer(TCP) or not ip.haslayer(Raw):
                    skipped_no_payload += 1
                    continue

                tcp = ip[TCP]
                payload = bytes(tcp[Raw])
                if len(payload) == 0:
                    skipped_no_payload += 1
                    continue

                # 构造唯一的流标识符（fid）：目标IP-源IP-目标端口-源端口-协议
                fid = f"{ip.dst}-{ip.src}-{tcp.dport}-{tcp.sport}-6"
                flows[fid].append((tcp.seq, payload))  # 按序号记录 TCP 片段
                parsed_pkt += 1

            except Exception as e:
                failed_pkt += 1
                if failed_pkt <= 5:
                    print(f" 解析失败：{e}")
                continue

    except Exception as e:
        print(f" 无法读取 {pcap_path}，错误：{e}")

    # 打印解析统计信息
    print(f"\n 报文统计：")
    print(f"   总包数       ：{total_pkt}")
    print(f"   成功解析     ：{parsed_pkt}")
    print(f"    跳过非IPv4   ：{skipped_non_ipv4}")
    print(f"    跳过无载荷   ：{skipped_no_payload}")
    print(f"   解析失败     ：{failed_pkt}")

    return flows

# 将每个 TCP 流的有效载荷按序重组，并写入独立文件保存
def save_payloads(flows, out_folder, max_payload_len=1024):
    os.makedirs(out_folder, exist_ok=True)  # 创建输出目录
    file_count = 0     # 成功写入的文件数量
    total_bytes = 0    # 所有文件总字节数

    for i, (fid, chunks) in enumerate(flows.items()):
        # 按 TCP 序号对 payload 分段排序
        sorted_chunks = sorted(chunks, key=lambda x: x[0])
        payload = b''.join([c[1] for c in sorted_chunks])
        if len(payload) == 0:
            continue
        payload = payload[:max_payload_len]  # 截断最大长度

        # 构造文件名，替换不安全字符
        safe_name = f"{fid.replace(':', '_').replace('/', '_')}_{i}.bin"
        path = os.path.join(out_folder, safe_name)

        # 写入二进制文件
        with open(path, 'wb') as f:
            f.write(payload)

        file_count += 1
        total_bytes += len(payload)

    return file_count, total_bytes


