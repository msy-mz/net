# Filename: run_realtime.py
# Path: inf/runner/run_realtime.py
# Description: 一键启动抓包+推理+频谱渲染系统（增强打印，统一输出目录）
# Author: msy
# Date: 2025

import os
import time
import threading
import subprocess
from inf.utils.spectral_plot import plot_all_recent

SNIFFER_SCRIPT = 'inf/sniffer/realtime_sniff.py'
PYTHON_PATH = '/home/msy/.conda/envs/net/bin/python'
SUDO_PASSWORD = 'msy'
PROJECT_ROOT = '/home/msy/net'

REALTIME_DIR = '/tmp/realtime'
PLOT_DIR = os.path.join(REALTIME_DIR, 'plot')
PLOT_INTERVAL = 10

def run_plot_loop():
    print(f"[频谱图] 自动渲染每 {PLOT_INTERVAL}s")
    while True:
        try:
            plot_all_recent()
        except Exception as e:
            print("[频谱渲染异常]", e)
        time.sleep(PLOT_INTERVAL)

def run_sniffer():
    print("[启动] 抓包推理系统 ...")
    cmd = f"echo {SUDO_PASSWORD} | sudo -S env PYTHONPATH={PROJECT_ROOT} {PYTHON_PATH} {SNIFFER_SCRIPT}"
    process = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    # 启动频谱渲染线程
    time.sleep(2)
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_thread = threading.Thread(target=run_plot_loop, daemon=True)
    plot_thread.start()

    print(f"[日志路径] {os.path.join(REALTIME_DIR, 'log.json')}")
    print(f"[频谱目录] {os.path.join(REALTIME_DIR, 'plot')}")

    for line in process.stdout:
        print(line, end='')

if __name__ == '__main__':
    run_sniffer()
