import os
from inf.payload.extract.tls import extract_flows_from_pcap, save_payloads

# 配置：要处理的 PCAP 文件路径列表（更新为 USTC-TFC2016）
pcap_list = [
    "data/pcap/USTC-TFC2016/Benign/BitTorrent.pcap",
    "data/pcap/USTC-TFC2016/Benign/Facetime.pcap",
    "data/pcap/USTC-TFC2016/Benign/FTP.pcap",
    "data/pcap/USTC-TFC2016/Benign/Gmail.pcap",
    "data/pcap/USTC-TFC2016/Benign/MySQL.pcap",
    "data/pcap/USTC-TFC2016/Benign/Outlook.pcap",
    "data/pcap/USTC-TFC2016/Benign/Skype.pcap",
    "data/pcap/USTC-TFC2016/Benign/SMB-1.pcap",
    "data/pcap/USTC-TFC2016/Benign/SMB-2.pcap",
    "data/pcap/USTC-TFC2016/Benign/Weibo-1.pcap",
    "data/pcap/USTC-TFC2016/Benign/Weibo-2.pcap",
    "data/pcap/USTC-TFC2016/Benign/Weibo-3.pcap",
    "data/pcap/USTC-TFC2016/Benign/Weibo-4.pcap",
    "data/pcap/USTC-TFC2016/Benign/WorldOfWarcraft.pcap",
    "data/pcap/USTC-TFC2016/Malware/Cridex.pcap",
    "data/pcap/USTC-TFC2016/Malware/Geodo.pcap",
    "data/pcap/USTC-TFC2016/Malware/Htbot.pcap",
    "data/pcap/USTC-TFC2016/Malware/Miuref.pcap",
    "data/pcap/USTC-TFC2016/Malware/Neris.pcap",
    "data/pcap/USTC-TFC2016/Malware/Nsis-ay.pcap",
    "data/pcap/USTC-TFC2016/Malware/Shifu.pcap",
    "data/pcap/USTC-TFC2016/Malware/Tinba.pcap",
    "data/pcap/USTC-TFC2016/Malware/Virut.pcap",
    "data/pcap/USTC-TFC2016/Malware/Zeus.pcap"
]

# 输出根目录
OUTPUT_ROOT = "inf/data/bin/tls_payload"
MAX_PAYLOAD_LEN = 1024

def process_one(pcap_path):
    print(f"\n处理文件: {pcap_path}")

    rel_path = os.path.relpath(pcap_path, "data/pcap")
    output_dir = os.path.splitext(os.path.join(OUTPUT_ROOT, rel_path))[0]

    flows = extract_flows_from_pcap(pcap_path)
    if not isinstance(flows, dict):  # 或 isinstance(flows, defaultdict)
        print("提取失败，未返回有效流对象。")
        return

    print(f"提取到 {len(flows)} 条符合条件的单向 TCP 流")

    os.makedirs(output_dir, exist_ok=True)
    file_count, total_bytes = save_payloads(flows, output_dir, max_payload_len=MAX_PAYLOAD_LEN)

    print(f"生成 bin 文件数：{file_count}")
    print(f"总提取字节数：{total_bytes}")
    print(f"输出路径：{output_dir}")


def interactive_mode():
    print("可处理的 PCAP 文件：")
    for i, path in enumerate(pcap_list):
        print(f"[{i}] {path}")

    try:
        index = int(input("\n请输入要处理的编号（或 -1 表示全部处理）："))
        if index == -1:
            return "all"
        return pcap_list[index]
    except (ValueError, IndexError):
        print("输入无效，已取消。")
        return None

def main():
    choice = interactive_mode()
    if choice == "all":
        print("\n开始批量处理所有 PCAP 文件...\n")
        for pcap_path in pcap_list:
            process_one(pcap_path)
        print("\n所有文件处理完成。")
    elif isinstance(choice, str):
        process_one(choice)

if __name__ == "__main__":
    main()
