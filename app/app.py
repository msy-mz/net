# Filename: app.py
# Path: app/app.py
# Description: Flask 主服务入口，增加 result 目录暴露
# Author: msy
# Date: 2025

import os
import subprocess
from flask import Flask, send_from_directory, request, jsonify
from flask import send_from_directory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, 'frontend')
PCAP_DIR = os.path.join(BASE_DIR, 'pcap_inputs')

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')

# 路由：前端页面
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# 路由：上传 PCAP 文件
@app.route('/upload_pcap', methods=['POST'])
def upload_pcap():
    file = request.files['file']
    os.makedirs(PCAP_DIR, exist_ok=True)
    save_path = os.path.join(PCAP_DIR, file.filename)
    file.save(save_path)
    return jsonify({"status": "success", "pcap_path": save_path})

# 路由：触发推理
@app.route('/run_infer', methods=['POST'])
def run_infer():
    pcap_path = request.json.get('pcap_path')
    if not os.path.exists(pcap_path):
        return jsonify({"status": "error", "message": "PCAP 文件不存在"}), 400

    # 更新 config 中的 pcap_path（建议使用模板复制并替换）
    config_path = os.path.join(BASE_DIR, 'inf/runner/config/ustc2016/tcp/multiclass/infer.yaml')
    with open(config_path, 'r') as f:
        lines = f.readlines()
    with open(config_path, 'w') as f:
        for line in lines:
            if line.strip().startswith('pcap_path:'):
                f.write(f'pcap_path: {pcap_path}\n')
            else:
                f.write(line)

    # 运行推理脚本
    result = subprocess.run(['python', 'inf/runner/infer_runner.py'], capture_output=True, text=True)

    if result.returncode == 0:
        return jsonify({"status": "success", "message": "推理完成"})
    else:
        return jsonify({"status": "error", "message": result.stderr}), 500
    


#  正确位置：放在 app.run() 之前
@app.route('/result/<path:filename>')
def serve_result_file(filename):
    result_dir = os.path.join(BASE_DIR, 'result')
    return send_from_directory(result_dir, filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8888, debug=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8888, debug=True)
