import os
import subprocess
import time
import signal

# === 配置参数 ===
MODE = "firefox"               # 可选: firefox, curl, python
INDEX = 1                      # 样本编号
IFACE = "enp4s0"               # 网卡接口
DURATION = 6                   # 抓包时长（秒）
SUDO_PASSWORD = "qsw"          # ⚠️ 仅限本机调试使用

# === 路径配置 ===
PROJECT_ROOT = "netprint"
CLIENT_SCRIPT_DIR = os.path.join(PROJECT_ROOT, "src/capture/clients")
PCAP_OUTPUT_PATH = os.path.join(PROJECT_ROOT, f"data/pcap/capture/{MODE}/{MODE}_capture_{INDEX}.pcap")

# === 启动客户端脚本 ===
def run_client(mode):
    script_map = {
        "firefox": "firefox_client.py",
        "curl": "curl_client.py",
        "python": "python_client.py"
    }
    script_file = script_map.get(mode)
    script_path = os.path.join(CLIENT_SCRIPT_DIR, script_file)

    if not os.path.isfile(script_path):
        print(f"[错误] 脚本文件不存在: {script_path}")
        return False

    try:
        subprocess.run(["python3", script_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[错误] 客户端脚本执行失败: {e}")
        return False

# === 启动 tcpdump 抓包进程 ===
def start_tcpdump(pcap_path, iface, sudo_password):
    os.makedirs(os.path.dirname(pcap_path), exist_ok=True)
    cmd = f"echo {sudo_password} | sudo -S tcpdump -i {iface} -nn -s 0 port 443 -w {pcap_path}"
    process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    return process

# === 主控流程 ===
def main():
    print(f"准备抓取样本：{MODE}_{INDEX}")
    print(f"保存路径：{PCAP_OUTPUT_PATH}")

    # 启动抓包
    tcpdump_proc = start_tcpdump(PCAP_OUTPUT_PATH, IFACE, SUDO_PASSWORD)
    time.sleep(1)  # 确保 tcpdump 启动

    # 执行客户端请求
    print(f"启动客户端连接: {MODE}")
    client_success = run_client(MODE)

    # 等待抓包时长
    print(f"抓包中，持续 {DURATION} 秒...")
    time.sleep(DURATION)

    # 停止抓包
    if tcpdump_proc.poll() is None:
        os.killpg(os.getpgid(tcpdump_proc.pid), signal.SIGTERM)
        tcpdump_proc.wait()

    if client_success:
        print(f"[完成] 抓包结束，保存至：{PCAP_OUTPUT_PATH}")
    else:
        print(f"[失败] 客户端执行失败，抓包文件可能无效")

if __name__ == "__main__":
    main()
