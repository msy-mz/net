# Filename: build_user_profile.py
# Description: 从标注好的连接级 CSV 中提取用户画像（以 src_ip 为单位），输出用户级 CSV
# Author: msy
# Date: 2025

import os
import pandas as pd
from collections import Counter
from tqdm import tqdm

def build_user_profile_from_folder(input_dir, output_csv):
    """
    将目录下所有 CSV 中的连接按 src_ip 聚合，提取用户画像统计特征

    :param input_dir: 原始标注连接 CSV 所在文件夹
    :param output_csv: 输出用户画像表路径
    """
    all_data = []

    for file in tqdm(os.listdir(input_dir), desc="加载连接数据"):
        if file.endswith(".csv"):
            path = os.path.join(input_dir, file)
            df = pd.read_csv(path)
            all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)

    # 预处理：填空值
    df_all.fillna({
        "ja3_fingerprint": "unknown",
        "cipher_suites": "unknown",
        "cipher_suite_top1": "unknown",
        "tcp_options": "unknown"
    }, inplace=True)

    df_all.fillna(0, inplace=True)

    # 按 src_ip 聚合
    group = df_all.groupby("src_ip")

    user_rows = []

    for src_ip, gdf in tqdm(group, desc="构建用户画像"):
        row = {"src_ip": src_ip}

        row["conn_count"] = len(gdf)
        row["ttl_mean"] = gdf["ttl"].mean()
        row["ttl_std"] = gdf["ttl"].std()
        row["tcp_window_mean"] = gdf["tcp_window"].mean()
        row["mss_mean"] = gdf["mss"].mean()
        row["has_tls_ratio"] = gdf["has_tls"].mean()
        row["ja3_count"] = gdf["ja3_fingerprint"].nunique()
        row["cipher_count"] = gdf["cipher_suites"].nunique()
        row["port_entropy_mean"] = gdf["src_port_entropy"].mean()

        # 主标签：统计出现最多的 label
        labels = gdf["label"].tolist()
        most_common_label = Counter(labels).most_common(1)[0][0]
        row["label"] = most_common_label

        user_rows.append(row)

    df_user = pd.DataFrame(user_rows)
    df_user.to_csv(output_csv, index=False)
    print(f"[完成] 共提取用户画像 {len(df_user)} 条，已保存至: {output_csv}")
