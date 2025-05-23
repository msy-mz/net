# Filename: app.py
# Path: backend/app.py
# Description: Flask 后端服务，挂载各模块与前端页面，支持 NIST 测试
# Author: msy
# Date: 2025

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from api.upload import upload_api
from api.infer import infer_api
from api.monitor import realtime_api
from api.result import result_api
from backend.utils.nist_run import run_nist_test
from api.nist_api import nist_api

app = Flask(__name__, static_folder='../frontend/dist', static_url_path='/')

CORS(app)

# 注册蓝图模块
app.register_blueprint(upload_api, url_prefix='/upload')
app.register_blueprint(infer_api, url_prefix='/infer')
app.register_blueprint(realtime_api, url_prefix='/realtime')
app.register_blueprint(result_api, url_prefix='/result')
app.register_blueprint(nist_api, url_prefix='/api/nist')

# Vue 前端页面兼容（支持刷新路径 /nist-test 等）
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_vue(path):
    full_path = os.path.join(app.static_folder, path)
    if path != "" and os.path.exists(full_path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# NIST 测试接口
@app.route('/api/nist_test', methods=['POST'])
def nist_test():
    if 'file' not in request.files:
        return jsonify({'error': '未提供文件'}), 400
    file = request.files['file']
    temp_path = os.path.join('/tmp', file.filename)
    file.save(temp_path)
    try:
        result = run_nist_test(temp_path)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(temp_path)

# 启动服务
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=False)
