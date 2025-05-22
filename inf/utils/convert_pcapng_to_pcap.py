# convert_single_pcapng.py
# 将单个 .pcapng 文件转换为 .pcap 格式

import os
import subprocess
from pathlib import Path

# === 配置输入输出文件 ===
INPUT_FILE = Path("data/pcap/CIC-IDS2017/Split_10000_Monday_WorkingHours_00000_20170703195558.pcap")  # 输入文件（必须是 .pcapng）
OUTPUT_FILE = Path("data/pcap/converted/CIC-IDS2017/Split_10000_Monday_WorkingHours_00000.pcap")  # 输出文件路径

def convert_pcapng_file(input_file: Path, output_file: Path):
    if not input_file.exists():
        print(f"❌ 输入文件不存在: {input_file}")
        return
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["editcap", "-F", "pcap", str(input_file), str(output_file)]
        subprocess.run(cmd, check=True)
        print(f"✅ 转换成功: {input_file} → {output_file}")
    except Exception as e:
        print(f"❌ 转换失败: {input_file}，错误信息: {e}")

if __name__ == "__main__":
    convert_pcapng_file(INPUT_FILE, OUTPUT_FILE)
