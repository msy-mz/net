# Filename: app.py
# Path: app/app.py
# Description: Flask 主服务入口，支持多个 PCAP 上传、模块化推理与可视化结果展示
# Author: msy
# Date: 2025

import os
import json
import sys
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


from inf.runner.run_infer import run_batch_infer
from inf.utils.visualizer import draw_visualizations

# ====== 全局路径配置 ======
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, 'frontend')
PCAP_DIR = os.path.join(BASE_DIR, 'pcap_inputs')
INFER_CONFIG = os.path.join(BASE_DIR, 'inf/runner/config/ustc2016/tcp/multiclass/infer.yaml')
RESULT_DIR = os.path.join(BASE_DIR, 'result/inf/infer')

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')


# ====== 页面路由：前端页面 ======
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


# ====== 接口：上传多个 PCAP 文件 ======
@app.route('/upload_pcap', methods=['POST'])
def upload_pcap():
    files = request.files.getlist('file')
    if not files:
        return jsonify({"status": "error", "message": "未选择文件"}), 400

    os.makedirs(PCAP_DIR, exist_ok=True)
    saved_paths = []

    for f in files:
        save_path = os.path.join(PCAP_DIR, f.filename)
        f.save(save_path)
        saved_paths.append(save_path)

    return jsonify({"status": "success", "pcap_paths": saved_paths})


# ====== 实时日志文件接口（log.json） ======
@app.route('/realtime/log.json')
def serve_realtime_log():
    log_path = '/tmp/realtime/log.json'
    if not os.path.exists(log_path):
        return jsonify([])  # 若文件不存在，返回空数组
    return send_from_directory('/tmp/realtime', 'log.json')


# ====== 实时频谱图像访问（PNG） ======
@app.route('/realtime/plot/<path:filename>')
def serve_realtime_plot(filename):
    plot_dir = '/tmp/realtime/plot'
    return send_from_directory(plot_dir, filename)



@app.route('/run_infer', methods=['POST'])
def run_infer():
    try:
        data = request.get_json(force=True)
        pcap_paths = data.get('pcap_paths')
        model_key = data.get('model', 'ft')  # 默认使用 ft 模型

        if not pcap_paths or not isinstance(pcap_paths, list) or len(pcap_paths) == 0:
            return jsonify({"status": "error", "message": "未提供有效的 PCAP 路径"}), 400

        # 模型配置路径映射（你可在此添加更多模型）
        config_map = {
            "ft": os.path.join(BASE_DIR, "inf/runner/config/ustc2016/tcp/multiclass/infer.yaml"),
            "id": os.path.join(BASE_DIR, "inf/runner/config/ustc2016/tcp/id_tcn/infer.yaml"),
            "fusion": os.path.join(BASE_DIR, "inf/runner/config/ustc2016/tcp/fusion/infer.yaml")
        }

        infer_config = config_map.get(model_key)
        if not infer_config or not os.path.exists(infer_config):
            return jsonify({"status": "error", "message": f"模型配置文件不存在：{model_key}"}), 400

        # 清理旧结果（推荐）
        os.makedirs(RESULT_DIR, exist_ok=True)
        for fname in ['summary.json', 'detailed_results.json']:
            fpath = os.path.join(RESULT_DIR, fname)
            if os.path.exists(fpath):
                os.remove(fpath)

        # 执行推理与聚合
        result = run_batch_infer(infer_config, pcap_paths)

        # 保存结构化结果
        with open(os.path.join(RESULT_DIR, 'summary.json'), 'w') as f:
            json.dump(result['summary'], f, indent=2)
        with open(os.path.join(RESULT_DIR, 'detailed_results.json'), 'w') as f:
            json.dump(result['detailed'], f, indent=2)

        # 绘图
        vis_output_dir = os.path.join(RESULT_DIR, 'feature_vis')
        draw_visualizations(result['detailed'], result['summary'], vis_output_dir)

        return jsonify({"status": "success", "message": f"{model_key} 模型推理完成"})

    except Exception as e:
        print("[系统异常]", str(e))
        return jsonify({"status": "error", "message": "系统异常，推理中断"}), 500

# ====== 静态资源暴露（图像可视化） ======
@app.route('/result/<path:filename>')
def serve_result_file(filename):
    result_root = os.path.join(BASE_DIR, 'result')
    return send_from_directory(result_root, filename)

# ====== 实时监听进程控制 ======
import subprocess
import signal

REALTIME_SCRIPT = os.path.join(BASE_DIR, 'inf/runner/run_realtime.py')
PID_FILE = '/tmp/realtime/pid'

@app.route('/realtime/start', methods=['POST'])
def start_realtime():
    if os.path.exists(PID_FILE):
        return jsonify({"status": "running", "message": "监听已在运行"})

    os.makedirs('/tmp/realtime', exist_ok=True)
    process = subprocess.Popen(
        ['python3', REALTIME_SCRIPT],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    with open(PID_FILE, 'w') as f:
        f.write(str(process.pid))
    return jsonify({"status": "started", "message": "已启动监听"})


import platform
import psutil  #  需要安装 pip install psutil

@app.route('/realtime/stop', methods=['POST'])
def stop_realtime():
    if not os.path.exists(PID_FILE):
        return jsonify({"status": "stopped", "message": "监听未运行"})

    try:
        with open(PID_FILE, 'r') as f:
            pid = int(f.read())

        import psutil
        if psutil.pid_exists(pid):
            proc = psutil.Process(pid)
            proc.terminate()
            proc.wait(timeout=5)  # 可选：等待其完全退出
            print(f"[监听已终止] PID={pid}")
        else:
            print(f"[监听进程已消失] PID={pid}")

        os.remove(PID_FILE)
        return jsonify({"status": "stopped", "message": "监听终止成功（或进程已不存在）"})

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[监听停止异常] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ====== 启动服务 ======
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8888, debug=True)
