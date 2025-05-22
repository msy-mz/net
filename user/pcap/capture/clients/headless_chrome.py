from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import WebDriverException, TimeoutException
import time
import sys
import traceback

url = "https://httpbin.org/get"
wait_time = 5
timeout = 10

options = Options()
options.binary_location = "/opt/google/chrome/chrome"
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")

service = Service(executable_path="netprint/bin/chromedriver")

driver = None
try:
    print("启动 headless Chrome...")
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(timeout)

    print(f"访问目标网址: {url}")
    driver.get(url)

    print(f"等待 {wait_time} 秒...")
    time.sleep(wait_time)

except (WebDriverException, TimeoutException) as e:
    print("Chrome 启动失败:")
    traceback.print_exc()
    sys.exit(1)
finally:
    if driver:
        driver.quit()
        print("Chrome 会话已关闭")
