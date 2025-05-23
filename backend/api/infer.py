# Filename: infer.py
# Path: backend/api/infer.py
# Description: 模型推理相关接口，支持多模型、多文件推理及结构化可视化
# Author: msy
# Date: 2025

import os
import json
from flask import Blueprint, request, jsonify
from inf.runner.run_infer import run_batch_infer
from inf.utils.visualizer import draw_visualizations

infer_api = Blueprint('infer_api', __name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RESULT_DIR = os.path.join(BASE_DIR, 'result/inf/infer')

@infer_api.route('/run', methods=['POST'])
def run_infer():
    try:
        data = request.get_json(force=True)
        pcap_paths = data.get('pcap_paths')
        model_key = data.get('model', 'ft')

        if not pcap_paths or not isinstance(pcap_paths, list):
            return jsonify({"status": "error", "message": "未提供有效的 PCAP 路径"}), 400

        config_map = {
            "ft": os.path.join(BASE_DIR, "inf/runner/config/ustc2016/tcp/multiclass/infer.yaml"),
            "id": os.path.join(BASE_DIR, "inf/runner/config/ustc2016/tcp/id_tcn/infer.yaml"),
            "fusion": os.path.join(BASE_DIR, "inf/runner/config/ustc2016/tcp/fusion/infer.yaml")
        }

        infer_config = config_map.get(model_key)
        if not infer_config or not os.path.exists(infer_config):
            return jsonify({"status": "error", "message": f"模型配置文件不存在：{model_key}"}), 400

        os.makedirs(RESULT_DIR, exist_ok=True)
        for fname in ['summary.json', 'detailed_results.json']:
            fpath = os.path.join(RESULT_DIR, fname)
            if os.path.exists(fpath):
                os.remove(fpath)

        result = run_batch_infer(infer_config, pcap_paths)

        with open(os.path.join(RESULT_DIR, 'summary.json'), 'w') as f:
            json.dump(result['summary'], f, indent=2)
        with open(os.path.join(RESULT_DIR, 'detailed_results.json'), 'w') as f:
            json.dump(result['detailed'], f, indent=2)

        vis_output_dir = os.path.join(RESULT_DIR, 'feature_vis')
        draw_visualizations(result['detailed'], result['summary'], vis_output_dir)

        return jsonify({"status": "success", "message": f"{model_key} 模型推理完成"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
