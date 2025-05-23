# Filename: run_build_user_profile.py
# Description: 启动用户画像构建流程，将连接级 CSV 聚合为用户级特征表
# Author: msy
# Date: 2025

from user.model.build_user_profile import build_user_profile_from_folder

if __name__ == "__main__":
    # === 参数配置 ===
    input_dir = "data/cic2017/print/train"               # 已标注连接级 CSV 的文件夹
    output_csv = "data/cic2017/print/user_profiles.csv"        # 用户级画像输出路径

    print(f"[启动] 构建用户画像: {input_dir}")
    build_user_profile_from_folder(input_dir, output_csv)
