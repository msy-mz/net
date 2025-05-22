from pyvirtualdisplay import Display
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
import time
import sys
import traceback

url = "https://httpbin.org/get"
wait_time = 5

# ✅ 启动虚拟显示（Xvfb）
display = Display(visible=0, size=(1024, 768))
display.start()

options = Options()
options.headless = True

service = Service(executable_path="netprint/bin/geckodriver")

driver = None
try:
    print("启动 headless Firefox...")
    driver = webdriver.Firefox(service=service, options=options)
    driver.get(url)
    print(f"等待 {wait_time} 秒...")
    time.sleep(wait_time)
except Exception as e:
    print("Firefox 启动失败:")
    traceback.print_exc()
    sys.exit(1)
finally:
    if driver:
        driver.quit()
        print("Firefox 会话已关闭")
    display.stop()
