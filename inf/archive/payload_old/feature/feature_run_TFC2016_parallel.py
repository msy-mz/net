# Filename: payload_feature_extract_run_TFC2016_parallel.py
# Description: 批量处理 bin 文件，提取频域特征并保存为 npz 文件（支持多线程并发）
# Author: msy
# Date: 2025

import os
import sys
import numpy as np
import time
import threading
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加模块路径，导入特征提取函数
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from inf.payload.feature.payload_feature_extract import extract_multiscale_features, compute_fft_spectrogram  # type: ignore

# === 可配置参数区域 ===
ROOT_BIN_DIR = "inf/data/bin/tcp_payload/USTC-TFC2016"   # bin 文件根目录（按类划分）
OUTPUT_DIR = "inf/data/npz/tcp/USTC-TFC2016"             # 输出 npz 目录
MAX_WORKERS = 8                                           # 最大并发线程数

# === 类别自动转为标签：Benign → 0，Malware → 1 ===
def auto_get_label(class_dir):
    return 0 if class_dir.upper() == "BENIGN" else 1

# === 扫描所有任务（每个子目录为一个任务） ===
def find_all_bin_tasks(root_bin_dir, output_root_dir):
    task_list = []
    for class_dir in ["Benign", "Malware"]:
        class_path = os.path.join(root_bin_dir, class_dir)
        if not os.path.exists(class_path):
            continue
        label = auto_get_label(class_dir)

        for subdir in sorted(os.listdir(class_path)):
            bin_folder = os.path.join(class_path, subdir)
            if not os.path.isdir(bin_folder):
                continue

            output_path = os.path.join(output_root_dir, class_dir, f"{subdir}.npz")
            task_list.append((class_dir, subdir, bin_folder, output_path, label))
    return task_list

# === 处理一个子目录：读取多个 bin 文件，提取频谱特征，统一保存为 npz 文件 ===
def process_bin_folder(bin_folder_path, output_path, label):
    spectrograms, labels, filenames = [], [], []

    for fname in os.listdir(bin_folder_path):
        if not fname.endswith(".bin"):
            continue
        fpath = os.path.join(bin_folder_path, fname)
        try:
            with open(fpath, "rb") as f:
                payload = f.read()

            feats = extract_multiscale_features(payload)     # 提取多尺度结构特征
            S = compute_fft_spectrogram(feats)               # 做 FFT 转为频谱图

            if S.shape != (12, 32):                           # 仅保留有效特征矩阵
                continue

            spectrograms.append(S)
            labels.append(label)
            filenames.append(fname)

        except Exception as e:
            print(f" 跳过 {fpath}: {e}")

    if spectrograms:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(output_path,
                 spectrograms=np.array(spectrograms),
                 labels=np.array(labels),
                 filenames=np.array(filenames))
        print(f" 保存: {output_path}，样本数: {len(labels)}")
    else:
        print(f" 无有效数据: {bin_folder_path}")

# === 控制台交互选择处理范围 ===
def interactive_mode(task_list):
    print("\n可处理的 bin 子目录：")
    for i, (cls, name, _, _, _) in enumerate(task_list):
        print(f"[{i}] {cls}/{name}")
    try:
        index = int(input("\n请输入要处理的编号（或 -1 表示全部处理）："))
        if index == -1:
            return "all"
        return task_list[index]
    except (ValueError, IndexError):
        print("输入无效，已取消。")
        return None

# === 主函数：控制流程 ===
def main():
    task_list = find_all_bin_tasks(ROOT_BIN_DIR, OUTPUT_DIR)
    total_tasks = len(task_list)
    print(f"\n输出目录: {OUTPUT_DIR}")
    print(f"待处理 bin 子目录总数: {total_tasks}")

    choice = interactive_mode(task_list)

    if choice == "all":
        print(f"\n准备构建并发任务队列...")
        valid_tasks = [
            (cls, name, bin_path, out_path, label)
            for cls, name, bin_path, out_path, label in task_list
            if not os.path.exists(out_path)
        ]
        print(f"有效任务数（尚未生成 .npz）: {len(valid_tasks)}")
        if not valid_tasks:
            print("所有任务均已处理完成，无需重复执行。")
            return

        print("\n将提交以下任务：")
        for cls, name, *_ in valid_tasks:
            print(f" - {cls}/{name}")

        print(f"\n开始并行处理 {len(valid_tasks)} 个 bin 子目录...\n")
        start_time = time.time()
        completed = 0
        task_times = []

        # === 子线程处理包装函数（打印进度） ===
        def wrapped_process(bin_path, out_path, label, cls, name):
            thread_id = threading.current_thread().name
            print(f"[{thread_id}] 开始处理: {cls}/{name}")
            t0 = time.time()
            process_bin_folder(bin_path, out_path, label)
            duration = time.time() - t0
            print(f"[{thread_id}] 完成: {cls}/{name}，耗时: {duration:.2f}s")
            return duration

        # === 使用线程池并发执行任务 ===
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(wrapped_process, bin_path, out_path, label, cls, name)
                for cls, name, bin_path, out_path, label in valid_tasks
            ]

            task_times = []
            for future in as_completed(futures):
                try:
                    duration = future.result()
                    task_times.append(duration)
                    completed += 1
                    avg_time = sum(task_times) / len(task_times)
                    remaining = avg_time * (len(futures) - completed)
                    print(f"[进度] [{completed}/{len(futures)}] 平均耗时: {avg_time:.2f}s 预计剩余: {str(timedelta(seconds=int(remaining)))}")
                except Exception as e:
                    print(f"任务失败：{e}")

        total_time = time.time() - start_time
        print(f"\n所有任务处理完成，总耗时: {str(timedelta(seconds=int(total_time)))}")

    elif isinstance(choice, tuple):
        cls, name, bin_path, out_path, label = choice
        print(f"\n处理单个目录: {cls}/{name}")
        print(f"输出目录: {out_path}")
        start_time = time.time()
        process_bin_folder(bin_path, out_path, label)
        duration = time.time() - start_time
        print(f"单个任务完成，耗时: {duration:.2f}秒")

# === 入口点 ===
if __name__ == "__main__":
    main()
