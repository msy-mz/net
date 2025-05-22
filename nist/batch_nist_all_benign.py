# nist测试代码必须放在 /sts-2.1.2/ 目录下
# 运行nist检测代码必须在 /sts-2.1.2/ 下运行

# nist检测代码中使用绝对路径（需要使用inf_autoencoder/下的数据）
# 运行inf_autoencoder/下的代码，需要将cwd切换为inf_autoencoder/（inf_autoencoder/下路径写的全部是相对路径）

import os
import json
import csv

# 输入目录、输出CSV文件路径和标签
input_directory = "/home/msy/net/inf/data/bin/payload/tcp/USTC-TFC2016/Benign/BitTorrent"  # 输入文件夹路径
output_csv_file = "/home/msy/net/nist/nist_result/tcp/USTC-TFC2016/Benign/BitTorrent.csv"  # 输出CSV文件路径
traffic_label = "BitTorrent"  # 固定标签为 "BENIGN"

# 随机性检测工具的参数（根据需要调整）
DataSegmentLength = "3000"  
Option = "0"                   
TestMode = "1"                
StreamNumber = "1"            
FileFormat = "1"  # 假设FileFormat=1表示二进制文件格式（根据实际情况调整）

os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)

def run_randomness_test(input_file):
    """
    运行随机性检测工具并返回结果。
    
    参数:
        input_file (str): 输入文件路径。
    
    返回:
        dict: 包含15项随机性测试指标的结果。
    """
    # 获取当前脚本所在目录
    current_dir = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(current_dir, input_file)

    # 检查输入文件是否存在
    if not os.path.isfile(input_file):
        print(f"Error: File '{input_file}' does not exist.")
        return None


    # 构造命令并调用随机性检测工具
    command = f"./assess {DataSegmentLength} {Option} {file_path} {TestMode} 0 {StreamNumber} {FileFormat}"
    print(f"Running command: {command}")
    out = os.popen(command)
    result = out.read()

    # 检查检测是否完成
    if "Statistical Testing Complete" in result:
        # 检查结果文件是否存在
        result_file_path = os.path.join(current_dir, "resultsum.txt")
        if os.path.isfile(result_file_path):
            try:
                # 读取结果文件并解析为 JSON
                with open(result_file_path, "r", encoding="utf-8") as file:
                    result_data = file.read()
                    json_data = json.loads(result_data)
                    return json_data
            except json.JSONDecodeError:
                print(f"Error: Failed to parse resultsum.txt for file '{input_file}'.")
        else:
            print(f"Error: Result file 'resultsum.txt' not found for file '{input_file}'.")
    else:
        print(f"Error: Randomness test failed or did not complete for file '{input_file}'.")
    
    return None

def batch_randomness_test(input_dir, output_csv, label="BENIGN"):
    """
    批量运行随机性检测，并将结果保存为CSV文件。
    
    参数:
        input_dir (str): 输入文件夹路径，包含待检测的 .txt 或 .bin 文件。
        output_csv (str): 输出CSV文件路径。
        label (str): 流量标签，默认为 "BENIGN"。
    """
    # 检查输入目录是否存在
    if not os.path.isdir(input_dir):
        print(f"Error: Directory '{input_dir}' does not exist.")
        return

    # 获取目录中的所有 .txt 和 .bin 文件
    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt") or f.endswith(".bin")]
    if not txt_files:
        print(f"Error: No .txt or .bin files found in directory '{input_dir}'.")
        return

    # 初始化CSV文件头，将标签列放在最后一列
    csv_header = [
        "File Name",
        "Frequency",
        "Block Frequency",
        "Cumulative Sums",
        "Runs",
        "Longest Run",
        "Rank",
        "FFT",
        "Non-overlapping Template",
        "Overlapping Template",
        "Universal",
        "Approximate Entropy",
        "Random Excursions",
        "Random Excursions Variant",
        "Serial",
        "Linear Complexity",
        "Label"  # 标签列放在最后一列 [[5]]
    ]
    
    # 打开CSV文件并写入表头
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_header)

        # 遍历每个文件并运行随机性检测
        for txt_file in txt_files:
            print(f"Processing file: {txt_file}")
            input_file_path = os.path.join(input_dir, txt_file)
            result = run_randomness_test(input_file_path)
            if result:
                # 构造CSV行数据，将标签放在最后一列
                row = [
                    txt_file,
                    result.get("Frequency", ""),
                    result.get("BlockFrequency", ""),
                    result.get("CumulativeSums", ""),
                    result.get("Runs", ""),
                    result.get("LongestRun", ""),
                    result.get("Rank", ""),
                    result.get("FFT", ""),
                    result.get("NonOverlappingTemplate", ""),
                    result.get("OverlappingTemplate", ""),
                    result.get("Universal", ""),
                    result.get("ApproximateEntropy", ""),
                    result.get("RandomExcursions", ""),
                    result.get("RandomExcursionsVariant", ""),
                    result.get("Serial", ""),
                    result.get("LinearComplexity", ""),
                    label  # 标签放在最后一列 [[5]]
                ]
                writer.writerow(row)

    print(f"Batch processing completed. Results saved to '{output_csv}'.")

if __name__ == '__main__':
    # 执行批量随机性检测
    batch_randomness_test(input_directory, output_csv_file, traffic_label)
