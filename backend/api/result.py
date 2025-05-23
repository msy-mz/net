# Filename: result.py
# Path: backend/api/result.py
# Description: 提供静态图像结果、日志 JSON、特征图等资源的访问接口
# Author: msy
# Date: 2025

import os
from flask import Blueprint, send_from_directory, jsonify

result_api = Blueprint('result_api', __name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

@result_api.route('/log.json')
def serve_log():
    log_path = '/tmp/realtime/log.json'
    if not os.path.exists(log_path):
        return jsonify([])
    return send_from_directory('/tmp/realtime', 'log.json')

@result_api.route('/plot/<path:filename>')
def serve_plot(filename):
    return send_from_directory('/tmp/realtime/plot', filename)

@result_api.route('/file/<path:filename>')
def serve_result_file(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'result'), filename)
