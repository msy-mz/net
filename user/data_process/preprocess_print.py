# Filename: preprocess_single_csv.py
# Description: 提供 preprocess_csv_to_npz(input_path, output_path) 函数
#              自动生成标签、保留全部字段、标准化并输出 .npz
# Author: msy
# Date: 2025

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_csv_to_npz(input_csv: str, output_npz: str):
    """
    处理单个 CSV 文件，保留全部特征，自动以文件名作为标签，输出为 .npz（label 为字符串）
    """
    df = pd.read_csv(input_csv)

    # 标签来源 = 文件名（不含扩展名）
    label_name = os.path.splitext(os.path.basename(input_csv))[0]

    # 移除 label 字段（如果存在）
    if 'label' in df.columns:
        df.drop(columns=['label'], inplace=True)

    # 填补缺失值
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(0)

    # 所有样本标签均为文件名
    labels = np.array([label_name] * len(df))

    # 编码所有对象字段
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col])

    # 标准化特征
    scaler = StandardScaler()
    features = scaler.fit_transform(df.values)

    # 保存字符串标签（不要编码！）
    np.savez(output_npz, features=features, labels=labels)
    print(f"[保存] {output_npz} | 标签: '{label_name}' | 特征维度: {features.shape[1]}")
