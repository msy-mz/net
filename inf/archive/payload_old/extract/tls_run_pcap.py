import os
from pathlib import Path
from inf.payload.extract.tls import extract_flows_from_pcap, is_tls_payload

# === 配置参数 ===
pcap_path = "data/pcap/converted/CIC-IDS2017/Split_10000_Monday_WorkingHours_00000.pcap"
output_folder = "inf/data/bin/tls_payload/CIC-IDS2017/Split_10000_Monday"
max_payload_len = 1024

os.makedirs(output_folder, exist_ok=True)

# === 提取流 ===
flows = extract_flows_from_pcap(pcap_path)

# === 保存为 TLS bin 文件 ===
file_count = 0
filtered_count = 0
total_bytes = 0

for fid, pkt_list in flows.items():
    pkt_list.sort(key=lambda x: x[0])
    merged_payload = b"".join([p for _, p in pkt_list])
    if len(merged_payload) == 0:
        continue

    if not is_tls_payload(merged_payload):
        filtered_count += 1
        continue

    payload = merged_payload[:max_payload_len]
    out_path = Path(output_folder) / f"{fid}.bin"
    with open(out_path, "wb") as f:
        f.write(payload)

    file_count += 1
    total_bytes += len(payload)

print(f"\n TLS 流筛选完成：总流数 = {len(flows)}，TLS流 = {file_count}，非TLS流 = {filtered_count}")
