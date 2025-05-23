# Filename: realtime.py
# Path: backend/api/realtime.py
# Description: 实时监听启动、停止控制模块
# Author: msy
# Date: 2025

import os
import subprocess
import psutil
from flask import Blueprint, jsonify

realtime_api = Blueprint('realtime_api', __name__)

REALTIME_SCRIPT = os.path.abspath('inf/runner/run_realtime.py')
PID_FILE = '/tmp/realtime/pid'

@realtime_api.route('/start', methods=['POST'])
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

@realtime_api.route('/stop', methods=['POST'])
def stop_realtime():
    if not os.path.exists(PID_FILE):
        return jsonify({"status": "stopped", "message": "监听未运行"})

    try:
        with open(PID_FILE, 'r') as f:
            pid = int(f.read())

        if psutil.pid_exists(pid):
            proc = psutil.Process(pid)
            proc.terminate()
            proc.wait(timeout=5)
        os.remove(PID_FILE)
        return jsonify({"status": "stopped", "message": "监听终止成功"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
