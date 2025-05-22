from scapy.all import PcapReader, Raw, TCP, IP

pcap_path = "data/pcap/aes-128-gcm/adblockplus.org/traffic_2024-01-18_adblockplus.org_aes-128_chromium_1.pcap.TCP_10-0-2-15_48032_148-251-232-132_443.pcap"
tls_count = 0

with PcapReader(pcap_path) as pcap:
    for pkt in pcap:
        if pkt.haslayer(IP) and pkt.haslayer(TCP) and pkt.haslayer(Raw):
            payload = bytes(pkt[Raw])
            if len(payload) >= 5:
                head = list(payload[:5])
                if payload[0] in [0x16, 0x17] and payload[1] == 0x03 and payload[2] in [0x01, 0x02, 0x03, 0x04]:
                    tls_count += 1
                    print(f"TLS: {head}")
print(f"\n共找到 {tls_count} 个 TLS 报文")
