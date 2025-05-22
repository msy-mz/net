import pandas as pd

# 输入和输出文件路径
input_file = 'data/csv/CIC-IDS2017/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'  # 输入CSV文件路径
output_file = 'user/data/csv/nonBENIGN_from_CIC-IDS2017/nonBENIGN_FridayDDos.csv'  # 输出CSV文件路径

# 读取输入CSV文件
try:
    # 尝试使用默认编码读取文件
    df = pd.read_csv(input_file, encoding='utf-8')
except UnicodeDecodeError:
    # 如果出现UnicodeDecodeError，尝试使用ISO-8859-1编码读取文件
    print("UnicodeDecodeError encountered, trying ISO-8859-1 encoding.")
    df = pd.read_csv(input_file, encoding='ISO-8859-1')

# 筛选出标签不是 'BENIGN' 的行
attack_df = df[df[' Label'] != 'BENIGN']

# 将结果保存到新的CSV文件
attack_df.to_csv(output_file, index=False)

print(f"攻击流量已保存到 {output_file}")
