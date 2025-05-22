import requests
import time

url = "https://httpbin.org/get"
print("使用 requests 发起请求...")

try:
    response = requests.get(url, verify=False)
    print(f"响应状态码: {response.status_code}")
    time.sleep(3)  # 等待以抓包
    print("requests 请求完成")
except Exception as e:
    print(f"请求失败: {e}")
