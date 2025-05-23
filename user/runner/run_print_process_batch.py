# Filename: run_batch_preprocess.py
# Description: 启动批处理，将文件夹内所有 CSV → .npz
# Author: msy
# Date: 2025

import os
import yaml
from tqdm import tqdm
from user.data_process.preprocess_print import preprocess_csv_to_npz

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def batch_preprocess(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    for file in tqdm(files, desc="批量预处理"):
        input_path = os.path.join(input_dir, file)
        label = os.path.splitext(file)[0]
        output_path = os.path.join(output_dir, f"{label}.npz")
        preprocess_csv_to_npz(input_path, output_path)

if __name__ == "__main__":
    config = load_config("user/runner/config/batch_preprocess_print.yaml")
    batch_preprocess(config["input_dir"], config["output_dir"])
