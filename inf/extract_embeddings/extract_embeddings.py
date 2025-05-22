"""
extract_embeddings.py

批量提取加密流载荷的内容行为嵌入向量（FT-Encoder++ 输出）。
适用于下游聚类分析、异常检测、客户端行为对比等任务。

作者：msy
时间：2025
"""

import os
import torch
import numpy as np
import sys
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from inf.src.payload_feature_extract.payload_feature_extract import extract_multiscale_features, compute_fft_spectrogram
from model import FTEncoder

def extract_embeddings_from_folder(input_folder, encoder, output_path,
                                   input_channels=12, freq_dim=32, device='cuda'):
    """
    批量从 .bin 文件中提取嵌入向量并保存
    参数：
        input_folder: 存放 bin 文件的目录
        encoder: 已加载的 FT-Encoder 模型
        output_path: 保存嵌入向量和标签的 .npz 文件
    """
    encoder.eval()
    embeddings = []
    filenames = []

    for fname in tqdm(os.listdir(input_folder)):
        if not fname.endswith(".bin"):
            continue
        fpath = os.path.join(input_folder, fname)
        try:
            with open(fpath, "rb") as f:
                payload = f.read()
            feats = extract_multiscale_features(payload)
            spectrogram = compute_fft_spectrogram(feats)
            if spectrogram.shape != (input_channels, freq_dim):
                continue  # 跳过异常输入

            spectrogram = torch.tensor(spectrogram, dtype=torch.float).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = encoder(spectrogram)  # shape: [1, hidden_dim]
            embeddings.append(emb.cpu().numpy()[0])
            filenames.append(fname)
        except Exception as e:
            print(f" 跳过文件 {fname}，原因: {e}")
            continue

    # 保存结果
    np.savez(output_path, embeddings=np.array(embeddings), filenames=np.array(filenames))
    print(f" 已保存嵌入向量: {output_path}")
