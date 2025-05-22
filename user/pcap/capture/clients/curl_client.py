import subprocess
import time

url = "https://httpbin.org/get"
print("使用 curl 发起请求...")

# 使用 curl 发起一次 HTTPS 请求（默认使用 TLS）
try:
    subprocess.run(["curl", "-k", "-s", url], check=True)
    time.sleep(3)  # 等待以抓包
    print("curl 请求完成")
except subprocess.CalledProcessError as e:
    print(f"curl 请求失败: {e}")
