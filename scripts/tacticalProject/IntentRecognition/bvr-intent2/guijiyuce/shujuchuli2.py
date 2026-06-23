import os
import pandas as pd
import numpy as np

# 设置文件路径
folder_path = r"c:\Users\86188\Desktop\try"
output_folder = os.path.join(folder_path, "windowed_data")

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有CSV文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 存储所有处理后的时间序列数据
all_time_series = []
series_labels = []

# 处理每个CSV文件
for file_idx, csv_file in enumerate(csv_files):
    print(f"Processing file {file_idx + 1}/{len(csv_files)}: {csv_file}")

    # 读取CSV文件
    file_path = os.path.join(folder_path, csv_file)
    try:
        df = pd.read_csv(file_path)

        # 检查是否包含Agent_ID列
        if 'Agent_ID' not in df.columns:
            print(f"  Warning: {csv_file} does not contain 'Agent_ID' column, skipping.")
            continue

        # 获取唯一的Agent_ID
        unique_agent_ids = df['Agent_ID'].unique()

        # 按照Agent_ID分组处理数据
        for agent_id in unique_agent_ids:
            # 提取当前Agent_ID的数据
            agent_data = df[df['Agent_ID'] == agent_id].copy()

            # 确保数据按时间排序
            if 'Time_s' in agent_data.columns:
                agent_data.sort_values('Time_s', inplace=True)

            # 为时间序列创建唯一标号
            # 格式: 文件名索引_Agent_ID
            series_label = f"{file_idx}_{agent_id}"

            # 存储时间序列数据（不包含时间列）
            # 选择需要的列作为特征
            feature_columns = []
            if all(col in agent_data.columns for col in
                   ['X_m', 'Y_m', 'Z_m', 'Heading_deg', 'Pitch_deg', 'Velocity_m_s']):
                feature_columns = ['X_m', 'Y_m', 'Z_m', 'Heading_deg', 'Pitch_deg', 'Velocity_m_s']
            else:
                # 如果找不到预期的列，使用所有数值列
                numeric_cols = agent_data.select_dtypes(include=[np.number]).columns.tolist()
                if 'Time_s' in numeric_cols:
                    numeric_cols.remove('Time_s')
                feature_columns = numeric_cols
                print(f"  Using available numeric columns: {feature_columns}")

            time_series_data = agent_data[feature_columns].values

            # 存储时间信息（如果有）
            time_info = agent_data['Time_s'].values if 'Time_s' in agent_data.columns else np.arange(len(agent_data))

            # 将时间序列、标号和时间信息添加到列表中
            all_time_series.append({
                'label': series_label,
                'data': time_series_data,
                'time_info': time_info,
                'original_file': csv_file,
                'agent_id': agent_id,
                'feature_columns': feature_columns
            })

            series_labels.append(series_label)
    except Exception as e:
        print(f"  Error processing {csv_file}: {str(e)}")
        continue

print(f"\n共处理了 {len(all_time_series)} 个时间序列")


# 定义滑动窗口函数
def sliding_window_with_history_and_label(data, window_size=75, history_size=50, label_size=25, step_size=1):
    """
    将时间序列分割成多个窗口，每个窗口包含历史数据和标签数据

    参数:
    data: 输入时间序列数据，形状为 (时间点数量, 特征维度)
    window_size: 窗口总长度
    history_size: 历史数据长度
    label_size: 标签数据长度
    step_size: 滑动步长

    返回:
    包含历史数据和标签数据的字典列表
    """
    windows = []

    # 检查窗口参数是否合法
    if history_size + label_size != window_size:
        raise ValueError("history_size + label_size must equal window_size")

    # 获取数据长度
    data_len = len(data)

    # 如果数据长度小于窗口大小，无法生成窗口
    if data_len < window_size:
        return windows

    # 生成滑动窗口
    i = 0
    while i <= data_len - window_size:
        # 添加当前窗口
        window_data = data[i:i + window_size]
        history_data = window_data[:history_size]
        label_data = window_data[history_size:]

        windows.append({
            'history': history_data,
            'label': label_data,
            'start_idx': i,
            'end_idx': i + window_size
        })

        # 如果剩余数据不足以按照步长滑动，但至少还能生成一个窗口，则不再继续滑动
        if i + step_size > data_len - window_size:
            break

        # 否则按照步长继续滑动
        i += step_size

    return windows


# 处理每个时间序列，应用滑动窗口
all_windows = []
window_metadata = []

for idx, time_series in enumerate(all_time_series):
    print(f"Applying sliding window to time series {idx + 1}/{len(all_time_series)}: {time_series['label']}")

    try:
        # 应用滑动窗口
        windows = sliding_window_with_history_and_label(
            time_series['data'],
            window_size=75,
            history_size=50,
            label_size=25,
            step_size=1
        )

        # 存储窗口数据和元数据
        for window_idx, window in enumerate(windows):
            # 为每个窗口创建唯一ID
            window_id = f"{time_series['label']}_win{window_idx}"

            all_windows.append({
                'window_id': window_id,
                'history': window['history'],
                'label': window['label']
            })

            window_metadata.append({
                'window_id': window_id,
                'original_series_label': time_series['label'],
                'original_file': time_series['original_file'],
                'agent_id': time_series['agent_id'],
                'window_index': window_idx,
                'start_idx': window['start_idx'],
                'end_idx': window['end_idx'],
                'feature_columns': time_series['feature_columns']
            })
    except Exception as e:
        print(f"  Error processing window for {time_series['label']}: {str(e)}")
        continue

print(f"\n共生成了 {len(all_windows)} 个滑动窗口")

# 准备保存的数据结构
if all_windows:
    # 创建数组存储所有历史数据和标签数据
    # 找到最大的时间步长，确保所有数据维度一致
    max_history_len = max(len(w['history']) for w in all_windows)
    max_label_len = max(len(w['label']) for w in all_windows)
    feature_dim = all_windows[0]['history'].shape[1] if all_windows else 0

    # 初始化存储数组
    history_data = np.zeros((len(all_windows), max_history_len, feature_dim))
    label_data = np.zeros((len(all_windows), max_label_len, feature_dim))
    window_ids = []

    # 填充数据
    for i, window in enumerate(all_windows):
        history_len = len(window['history'])
        label_len = len(window['label'])

        history_data[i, :history_len, :] = window['history']
        label_data[i, :label_len, :] = window['label']
        window_ids.append(window['window_id'])

    # 保存处理后的数据（使用新的文件名以区分窗口设置）
    save_path = os.path.join(output_folder, "windowed_trajectory_data_75_50_25.npz")
    np.savez_compressed(
        save_path,
        history_data=history_data,
        label_data=label_data,
        window_ids=window_ids,
        metadata=window_metadata
    )

    print(f"\n滑动窗口数据处理完成！")
    print(f"数据保存在: {save_path}")
    print(f"历史数据形状: {history_data.shape}")
    print(f"标签数据形状: {label_data.shape}")
    print("\n窗口设置: 总长度75，前50步为输入，后25步为标签")
    print("\n后续可以使用以下代码读取数据:")
    print("------------------------------------")
    print("import numpy as np\n")
    print("# 读取处理后的数据")
    print(f"data = np.load('{save_path}', allow_pickle=True)")
    print("history_data = data['history_data']")
    print("label_data = data['label_data']")
    print("window_ids = data['window_ids']")
    print("metadata = data['metadata']\n")
    print("# 查看数据信息")
    print(f"print(f'历史数据形状: {{{{history_data.shape}}}}')")
    print(f"print(f'标签数据形状: {{{{label_data.shape}}}}')")
    print(f"print(f'窗口数量: {{{{len(window_ids)}}}}')")
else:
    print("\n没有生成任何滑动窗口数据！")