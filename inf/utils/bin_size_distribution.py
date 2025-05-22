# Filename: bin_size_distribution.py
# Description: 按固定字节间隔统计文件大小分布并输出频率图
# Author: msy
# Date: 2025

# ================= 可修改参数 =================
# 目标文件夹路径
target_folder = 'inf/data/bin/payload/tcp/USTC-TFC2016/Benign/BitTorrent'

# 文件扩展名过滤（只统计此类文件）
file_extension = '.bin'

# 分布区间大小（单位：字节）
bin_size_step = 100  # 每100B为一个区间

# 是否显示柱状图
show_plot = True

# 输出文件路径（保存统计文本结果）
output_file = 'inf/report/bin_size_distribution/tcp/USTC-TFC2016/Benign/BitTorrent.txt'

# 分布图输出路径（保存图像）
output_image = 'inf/report/bin_size_distribution/tcp/USTC-TFC2016/Benign/BitTorrent.png'
# ============================================

# 导入所需模块
import os
from collections import Counter
import matplotlib.pyplot as plt

# 确保输出目录存在
for path in [output_file, output_image]:
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

# 用于收集文件大小的列表
file_sizes = []

# 遍历文件夹，收集bin文件的大小
for root, _, files in os.walk(target_folder):
    for file in files:
        if file.lower().endswith(file_extension):
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path)
            file_sizes.append(size)

# 将文件大小归类到不同的区间（按100字节分段）
size_bins = [size // bin_size_step for size in file_sizes]
size_distribution = Counter(size_bins)

# 计算频率分布
total_files = len(file_sizes)
frequency_distribution = {k: v / total_files for k, v in size_distribution.items()}

# 输出统计结果到文件
with open(output_file, 'w') as f:
    f.write('file size frequency distribution (step = {} bytes):\n'.format(bin_size_step))
    for bin_index in sorted(frequency_distribution):
        lower = bin_index * bin_size_step
        upper = (bin_index + 1) * bin_size_step
        freq = frequency_distribution[bin_index]
        f.write('[{} - {}): {:.4f}\n'.format(lower, upper, freq))

# 输出控制台提示
print('Frequency distribution saved to {}'.format(output_file))

# 可视化：绘制柱状图（频率分布）
bin_labels = ['[{}-{})'.format(i * bin_size_step, (i + 1) * bin_size_step) for i in sorted(frequency_distribution)]
frequencies = [frequency_distribution[i] for i in sorted(frequency_distribution)]

plt.figure(figsize=(12, 6))
plt.bar(bin_labels, frequencies, color='skyblue')
plt.xlabel('File Size Interval (Bytes)')
plt.ylabel('Frequency')
plt.title('File Size Frequency Distribution')
plt.xticks(rotation=90)
plt.tight_layout()

# 保存图像
plt.savefig(output_image)
print('Plot saved to {}'.format(output_image))

# 根据需要显示图像
if show_plot:
    plt.show()
