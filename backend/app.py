# Filename: app.py
# Path: backend/app.py
# Description: 提供 /api/nist-test 接口，执行 NIST SP 800-22 随机性测试
# Author: msy
# Date: 2025

from flask import Flask, request, jsonify
from flask_cors import CORS
from nistrng import nist_test_suite

app = Flask(__name__)
CORS(app)  # 允许跨域请求

@app.route('/api/nist-test', methods=['POST'])
def nist_test():
    data = request.get_json()
    binary_data = data.get('binary_data', '')

    # 将二进制字符串转换为字节数组
    byte_data = bytes(int(binary_data[i:i+8], 2) for i in range(0, len(binary_data), 8))

    # 执行 NIST 测试
    results = nist_test_suite.run_all(byte_data)

    # 格式化结果
    formatted_results = []
    for test_name, result in results.items():
        formatted_results.append({
            'test_name': test_name,
            'p_value': result['p_value'],
            'success': result['success']
        })

    return jsonify({'results': formatted_results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
