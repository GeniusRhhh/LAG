import pandas as pd
import numpy as np
import os
import glob
import random
from sklearn.utils import shuffle

# 输入文件夹路径
input_folder = 'chuli1'
# 输出文件夹路径
output_folder = 'chuli2'
os.makedirs(output_folder, exist_ok=True)

# 滑动窗口参数
window_size = 50  # 窗口长度为50（对应10秒的时间序列数据）
def create_sliding_windows(df, window_size):
    """
    使用滑动窗口法从数据框中提取时间序列窗口
    
    参数:
    df: 输入数据框
    window_size: 窗口大小
    
    返回:
    windows: 提取的所有窗口列表
    """
    windows = []
    
    # 确保数据按时间排序
    if 'Time_s' in df.columns:
        df = df.sort_values('Time_s')
    
    # 提取所有数值列（排除不需要的列）
    # 保留原始特征和新添加的意图及独热编码
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 创建滑动窗口
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i+window_size][numeric_cols].values
        windows.append(window)
    
    return windows, numeric_cols

def process_files():
    """
    处理所有文件夹中的CSV文件，提取滑动窗口并保存
    """
    all_windows = []
    all_column_names = None
    processed_file_count = 0
    
    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
    
    for folder in subfolders:
        folder_path = os.path.join(input_folder, folder)
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        
        for csv_file in csv_files:
            try:
                # 读取CSV文件
                print(f"处理文件: {csv_file}")
                df = pd.read_csv(csv_file)
                
                # 创建滑动窗口
                windows, column_names = create_sliding_windows(df, window_size)
                
                # 保存列名信息（只保存一次）
                if all_column_names is None:
                    all_column_names = column_names
                
                # 添加到总窗口列表
                all_windows.extend(windows)
                processed_file_count += 1
                print(f"  提取了 {len(windows)} 个窗口")
                
            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {str(e)}")
    
    print(f"\n所有文件处理完成，共处理 {processed_file_count} 个文件")
    print(f"总共提取了 {len(all_windows)} 个时间序列窗口")
    
    return all_windows, all_column_names

def save_processed_data(windows, column_names):
    """
    打乱并保存处理后的数据
    """
    # 打乱数据
    print("\n打乱数据...")
    random.shuffle(windows)
    
    # 将窗口数据转换为numpy数组
    windows_array = np.array(windows)
    
    # 保存数据
    output_file = os.path.join(output_folder, 'processed_time_series_data.npz')
    print(f"保存处理后的数据到: {output_file}")
    np.savez(output_file, 
             data=windows_array,
             window_size=window_size,
             column_names=np.array(column_names))
    
    # 同时保存数据集信息文件
    info_file = os.path.join(output_folder, 'dataset_info.txt')
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"数据集信息\n")
        f.write(f"- 总窗口数: {len(windows)}\n")
        f.write(f"- 窗口大小: {window_size}\n")
        f.write(f"- 每个窗口的特征数: {windows_array.shape[2]}\n")
        f.write(f"- 数据形状: {windows_array.shape}\n\n")
        f.write("特征列名:\n")
        for i, col in enumerate(column_names):
            f.write(f"{i}: {col}\n")
    
    print(f"数据集信息已保存到: {info_file}")

def main():
    """
    主函数
    """
    print("开始使用滑动窗口法处理数据...")
    print(f"输入文件夹: {os.path.abspath(input_folder)}")
    print(f"输出文件夹: {os.path.abspath(output_folder)}")
    print(f"滑动窗口大小: {window_size} (对应10秒的时间序列数据)")
    
    # 处理文件并提取滑动窗口
    windows, column_names = process_files()
    
    # 保存处理后的数据
    if windows:
        save_processed_data(windows, column_names)
        print("\n数据处理完成！")
    else:
        print("\n未提取到任何窗口数据，请检查输入文件。")

if __name__ == "__main__":
    main()