# Filename: upload.py
# Path: backend/api/upload.py
# Description: 上传接口（支持多文件）
# Author: msy
# Date: 2025

import os
from flask import Blueprint, request, jsonify

upload_api = Blueprint('upload_api', __name__)
PCAP_DIR = os.path.abspath('pcap_inputs')

@upload_api.route('/pcap', methods=['POST'])
def upload_pcap():
    files = request.files.getlist('file')
    if not files:
        return jsonify({"status": "error", "message": "未选择文件"}), 400

    os.makedirs(PCAP_DIR, exist_ok=True)
    saved_paths = []

    for f in files:
        path = os.path.join(PCAP_DIR, f.filename)
        f.save(path)
        saved_paths.append(path)

    return jsonify({"status": "success", "pcap_paths": saved_paths})
