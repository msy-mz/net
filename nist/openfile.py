import json
import os
from flask import Flask, jsonify, request, Response
from flask_cors import *

# 创建一个服务，赋值给APP
app = Flask(__name__)

@app.route('/SendRequest', methods=['post'])
@cross_origin(supports_credentials=True)
def send_command_parameter():

    # DataSegmentLength = request.form.get('SLength')
    # Option = request.form.get('operation')
    # FileName = request.form.get('filename')
    # TestMode = request.form.get('testmode')
    # TestParameter = request.form.get('testpara')
    # StreamNumber = request.form.get('streamnumber')
    # FileFormat = request.form.get('fileformat')

    File = request.files['file']
    File.save(File.filename)

    DataSegmentLength = "1000000"
    # Option = "0"
    FileName = File.filename
    # TestMode = "1"
    # TestParameter = "0"
    StreamNumber = "1"
    FileFormat = "0"

    # 根据当前文件所在位置组装模板文件路径
    current_dir = os.path.abspath(os.path.dirname(__file__))
    template_path = current_dir + '/' + FileName

    # 如果路径存在则运行命令
    if os.path.isfile(template_path):
        out = os.popen("./assess " + DataSegmentLength + " 0 " + FileName + " 1 0 " + StreamNumber + " " + FileFormat)
        string = out.read()
        if string.find("Statistical Testing Complete") >= 0:
            with open("resultsum.txt", "r", encoding="utf-8")as file:
                str = file.read()
                data = json.loads(str)
                return data
        else:
            return "run failed"
    else:
        return FileName + " is not existed"

@app.route('/result', methods=['get'])
@cross_origin(supports_credentials=True)
def get_result():
    with open("resultsum.txt", "r", encoding="utf-8")as file:
        str = file.read()
        data = json.loads(str)
        return data
    return {}

if __name__ == '__main__':
    ip = '0.0.0.0'
    port = 19996
    app.run(ip, port, debug=True)
