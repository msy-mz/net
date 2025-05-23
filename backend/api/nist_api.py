# Filename: nist_api.py
# Path: backend/api/nist_api.py
# Description: Flask 接口：接收上传文件，执行 NIST 测试，返回报告
# Author: msy
# Date: 2025

from flask import Blueprint, request, jsonify
import os
from utils.nist_run import run_nist_test

nist_api = Blueprint('nist_api', __name__)

@nist_api.route('/test', methods=['POST'])
def handle_nist_test():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "未提供文件"}), 400

    temp_path = os.path.join('/tmp', file.filename)
    file.save(temp_path)

    try:
        result = run_nist_test(temp_path)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_path)
