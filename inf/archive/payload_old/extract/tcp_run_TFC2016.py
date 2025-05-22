import os
from inf.payload.extract.tcp import extract_flows_from_pcap, save_payloads

#  配置：要处理的 PCAP 文件路径列表
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

#  输出主目录（按分类动态创建子目录）
OUTPUT_ROOT = r"inf/data/bin/payload/tcp/USTC-TFC2016"
MAX_PAYLOAD_LEN = 1024

def auto_get_name(pcap_path):
    """
    从路径中提取 pcap 文件名（不含扩展名）
    """
    return os.path.splitext(os.path.basename(pcap_path))[0]

def auto_get_category(pcap_path):
    """
    自动识别所属类别（Benign 或 Malware）
    """
    return "Malware" if "Malware" in pcap_path else "Benign"

def process_one(pcap_path):
    """
    处理单个 PCAP 文件并保存为多个 .bin 文件
    """
    name = auto_get_name(pcap_path)
    category = auto_get_category(pcap_path)
    output_dir = os.path.join(OUTPUT_ROOT, category, name)

    print(f"\n 正在处理：{pcap_path}")
    flows = extract_flows_from_pcap(pcap_path)
    print(f" 提取到 {len(flows)} 条 TCP 单向流")

    file_count, total_bytes = save_payloads(flows, output_dir, max_payload_len=MAX_PAYLOAD_LEN)

    print(f" 生成 bin 文件数：{file_count}")
    print(f" 总字节数：{total_bytes}")
    print(f" 输出目录：{output_dir}")

def interactive_mode():
    """
    手动选择模式：单个处理
    """
    print(" 可选的 PCAP 文件：")
    for i, path in enumerate(pcap_list):
        print(f"  [{i}] {path}")

    try:
        index = int(input("\n请输入要处理的编号（或 -1 处理所有）："))
        if index == -1:
            return "all"
        return pcap_list[index]
    except (ValueError, IndexError):
        print(" 输入无效，取消处理。")
        return None

def main():
    choice = interactive_mode()
    if choice == "all":
        print("\n 正在批量处理所有 PCAP 文件...\n")
        for pcap_path in pcap_list:
            process_one(pcap_path)
        print("\n 所有文件处理完成。")
    elif isinstance(choice, str):
        process_one(choice)

if __name__ == "__main__":
    main()
