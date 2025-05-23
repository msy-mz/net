# Filename: realtime_sniff.py
# Path: inf/sniffer/realtime_sniff.py
# Description: 实时抓取真实网卡 TCP 流量，调用推理逻辑并记录结果、频谱特征与图像（自动清理、打印增强）
# Author: msy
# Date: 2025

import os
import json
import threading
from collections import deque
from datetime import datetime
import pyshark

from inf.infer.infer_live import infer_payload

# === 配置参数（统一使用 realtime 根目录）===
REALTIME_DIR = '/tmp/realtime'
LOG_PATH = os.path.join(REALTIME_DIR, 'log.json')
SPECTRUM_DIR = os.path.join(REALTIME_DIR, 'spectrum')
PLOT_DIR = os.path.join(REALTIME_DIR, 'plot')
MAX_ENTRIES = 30

INTERFACE = 'enp4s0'
BPF_FILTER = 'tcp port 80'

# === 初始化路径与缓存 ===
os.makedirs(SPECTRUM_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
log_buffer = deque(maxlen=MAX_ENTRIES)
log_lock = threading.Lock()

def log_result(entry):
    with log_lock:
        log_buffer.append(entry)
        with open(LOG_PATH, 'w') as f:
            json.dump(list(log_buffer), f, ensure_ascii=False, indent=2)

def clean_old_files(folder, suffix):
    files = sorted(
        [f for f in os.listdir(folder) if f.endswith(suffix)],
        key=lambda f: os.path.getmtime(os.path.join(folder, f))
    )
    for f in files[:-MAX_ENTRIES]:
        os.remove(os.path.join(folder, f))

def save_spectrum(spectrum_data, timestamp):
    fname = f"{timestamp}.json"
    fpath = os.path.join(SPECTRUM_DIR, fname)
    with open(fpath, 'w') as f:
        json.dump(spectrum_data, f)
    clean_old_files(SPECTRUM_DIR, '.json')

def handle_packet(pkt):
    try:
        if 'TCP' in pkt and hasattr(pkt.tcp, 'payload'):
            payload = bytes.fromhex(pkt.tcp.payload.replace(':', ''))
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

            result = infer_payload(payload)

            record = {
                "timestamp": timestamp,
                "src_ip": pkt.ip.src,
                "dst_ip": pkt.ip.dst,
                "src_port": pkt.tcp.srcport,
                "dst_port": pkt.tcp.dstport,
                "length": len(payload),
                "label": result['label'],
                "is_abnormal": result['is_abnormal']
            }

            log_result(record)
            save_spectrum(result['spectrum'], timestamp)

            print(f"[✓] {timestamp} | {record['src_ip']}:{record['src_port']} → {record['dst_ip']}:{record['dst_port']} | {record['label']} | {'异常' if record['is_abnormal'] else '正常'}")

    except Exception as e:
        print("[异常]", e)

def start_sniff():
    print(f"[抓包] 接口: {INTERFACE} | 过滤: {BPF_FILTER}")
    print(f"[输出] 日志路径: {LOG_PATH}")
    print(f"[输出] 频谱目录: {SPECTRUM_DIR}")
    capture = pyshark.LiveCapture(interface=INTERFACE, bpf_filter=BPF_FILTER)
    for pkt in capture.sniff_continuously():
        handle_packet(pkt)

if __name__ == '__main__':
    start_sniff()
