# build_dataset_from_npz.py
# author: msy
# description: 合并 spectrogram_split 文件夹下的 .npz 特征数据，生成总数据集

import os
import numpy as np

# ======== 参数配置（集中在顶部） ========
INPUT_FOLDER = "inf/data/npz/tcp/USTC-TFC2016/"
OUTPUT_FILE = "inf/data/npz/tcp/USTC-TFC2016_all.npz"
CLASS_LABELS = [("Benign", 0), ("Malware", 1)]
# =======================================

def load_dataset(folder_path):
    X, y = [], []
    for label_name, label in CLASS_LABELS:
        subfolder = os.path.join(folder_path, label_name)
        if not os.path.exists(subfolder):
            print(f"跳过不存在的文件夹: {subfolder}")
            continue

        for fname in os.listdir(subfolder):
            if fname.endswith(".npz"):
                path = os.path.join(subfolder, fname)
                data = np.load(path)
                if "spectrograms" not in data:
                    raise KeyError(f"{fname} 中不包含 'spectrograms'")
                spectrogram = data["spectrograms"]
                X.append(spectrogram)
                y.extend([label] * spectrogram.shape[0])

    return np.concatenate(X, axis=0), np.array(y)

if __name__ == "__main__":
    X, y = load_dataset(INPUT_FOLDER)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    np.savez(OUTPUT_FILE, X=X, y=y)
    print(f"保存合并数据集至: {OUTPUT_FILE}")
    print(f"特征维度: {X.shape}，标签数: {y.shape}")
