import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from inf.payload.extract.tcp import extract_flows_from_pcap, save_payloads  # 自定义模块

# 根目录配置
ROOT_PCAP_DIR = r"data\pcap\CIC-IDS2017"
OUTPUT_ROOT = r"data\bin\payload\tcp"
MAX_PAYLOAD_LEN = 1024
MAX_WORKERS = 8

def find_all_pcap_files(root_dir):
    pcap_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".pcap"):
                pcap_paths.append(os.path.join(dirpath, fname))
    return sorted(pcap_paths)

def auto_get_name(pcap_path):
    return os.path.splitext(os.path.basename(pcap_path))[0]

def auto_get_category(pcap_path):
    return "Malware" if "Malware" in pcap_path else "Benign"

def process_one(pcap_path):
    name = auto_get_name(pcap_path)
    
    # 生成新的输出目录路径，并确保目录结构为 'CIC-IDS2017\文件名'
    output_dir = os.path.join(OUTPUT_ROOT, "CIC-IDS2017", name)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n正在处理：{pcap_path}")
    flows = extract_flows_from_pcap(pcap_path)
    print(f"提取到 {len(flows)} 条 TCP 单向流")

    # 将生成的 bin 文件保存到输出路径，并确保文件后缀是 .bin
    file_count, total_bytes = save_payloads(flows, output_dir, max_payload_len=MAX_PAYLOAD_LEN)

    print(f"生成 bin 文件数：{file_count}")
    print(f"总字节数：{total_bytes}")
    print(f"输出目录：{output_dir}")

def interactive_mode(pcap_paths):
    print("\n可处理的 PCAP 文件：")
    for i, path in enumerate(pcap_paths):
        print(f"[{i}] {path}")
    try:
        index = int(input("\n请输入要处理的编号（或 -1 处理所有）："))
        if index == -1:
            return "all"
        return pcap_paths[index]
    except (ValueError, IndexError):
        print("输入无效，取消处理。")
        return None

def main():
    pcap_list = find_all_pcap_files(ROOT_PCAP_DIR)
    if not pcap_list:
        print(f"未找到 PCAP 文件：{ROOT_PCAP_DIR}")
        return

    choice = interactive_mode(pcap_list)
    if choice == "all":
        print(f"\n正在并行处理 {len(pcap_list)} 个 PCAP 文件...\n")
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(pcap_list))) as executor:
            futures = [executor.submit(process_one, path) for path in pcap_list]
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    future.result()
                except Exception as e:
                    print(f"错误：{e}")
                print(f"[进度] [{i}/{len(futures)}]")
        print("\n所有文件处理完成。")
    elif isinstance(choice, str):
        process_one(choice)

if __name__ == "__main__":
    main()
