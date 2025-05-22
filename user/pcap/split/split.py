from scapy.utils import PcapReader, PcapWriter
from scapy.all import Ether
import os

PCAP_PATH = r"data\pcap\CIC-IDS2017_converted\Monday-WorkingHours_converted.pcap"
OUTPUT_PATH = r"data\pcap\CIC-IDS2017_splited\Monday_splited"
MAX_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2GB

def split_by_size(pcap_path, output_path, max_size):
    total_size = 0
    with PcapReader(pcap_path) as reader, PcapWriter(output_path, sync=True, linktype=1) as writer:
        for pkt in reader:
            try:
                # 强制按 Ethernet 层解包
                ether_pkt = Ether(bytes(pkt))
                pkt_len = len(bytes(ether_pkt)) + 16  # pcap header 16 bytes
                if total_size + pkt_len > max_size:
                    break
                writer.write(ether_pkt)
                total_size += pkt_len
            except Exception as e:
                print(f"❌ 跳过错误包: {e}")
    print(f"✅ 已写入 {total_size / (1024**2):.2f} MB 到 {output_path}")

if __name__ == "__main__":
    split_by_size(PCAP_PATH, OUTPUT_PATH, MAX_SIZE_BYTES)