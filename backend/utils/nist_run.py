# Filename: nist_runner.py
# Path: backend/utils/nist_runner.py
# Description: 封装 NIST assess 工具调用，兼容 Web 接入
# Author: msy
# Date: 2025

import os
import subprocess

# NIST assess 所在目录（请修改为你机器上的真实路径）
NIST_TOOL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../nist'))
ASSESS_EXEC = os.path.join(NIST_TOOL_PATH, 'assess')

def run_nist_test(file_path):
    """
    调用 NIST assess 工具检测指定文件
    :param file_path: 二进制文件路径
    :return: 包含摘要和报告内容的 dict
    """
    if not os.path.exists(file_path):
        raise RuntimeError(f"输入文件不存在: {file_path}")
    if not os.path.exists(ASSESS_EXEC):
        raise RuntimeError(f"NIST assess 未找到: {ASSESS_EXEC}")

    command = [
        ASSESS_EXEC, "100000", "0", file_path, "1", "0", "1", "1"
    ]

    try:
        subprocess.run(command, cwd=NIST_TOOL_PATH, check=True, timeout=20)
        report_file = os.path.join(NIST_TOOL_PATH, "experiments", "AlgorithmTesting", "finalAnalysisReport.txt")
        if not os.path.exists(report_file):
            raise RuntimeError("NIST 未生成报告文件")

        with open(report_file, 'r', encoding='utf-8') as f:
            report = f.read()

        return {
            "summary": "NIST 测试完成",
            "report": report[:1000]  # 摘要截取
        }

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"NIST assess 执行失败: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"NIST 测试异常: {str(e)}")
