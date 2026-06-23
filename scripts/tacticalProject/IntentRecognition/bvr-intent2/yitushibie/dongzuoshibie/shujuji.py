import os
import sys
import numpy as np
import pandas as pd
import json
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 定义11种动作类型及其对应的独热编码 - 窗口长度为50
action_types = {
    'level_flight': 0,
    'accelerate': 1,
    'decelerate': 2,
    'turn_left': 3,
    'turn_right': 4,
    'climb': 5,
    'climb_left': 6,
    'climb_right': 7,
    'dive': 8,
    'dive_left': 9,
    'dive_right': 10
}


# 将动作类型转换为独热编码
def action_to_onehot(action):
    onehot = np.zeros(len(action_types))
    if action in action_types:
        onehot[action_types[action]] = 1
    return onehot

# 从索引获取动作名称
def idx_to_action(idx):
    for action, index in action_types.items():
        if index == idx:
            return action
    return None

# 简单的LSTM模型定义，用于加载预训练模型
class LSTMActionClassifier(torch.nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, num_classes=11, dropout=0.2):
        super(LSTMActionClassifier, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        out = self.dropout(last_time_step)
        out = self.fc(out)
        return out, self.softmax(out)


# 数据处理主函数
def process_data(base_dir, window_size=50, output_file='processed_data.npz'):
    # 存储所有处理后的样本和标签
    all_samples = []
    all_labels = []

    # 统计每个动作类型的样本数量
    label_counts = {action: 0 for action in action_types}

    print(f"开始处理数据，基础目录: {base_dir}")
    print(f"窗口大小: {window_size}")

    # 验证窗口大小是否合理
    if window_size <= 0:
        raise ValueError(f"窗口大小必须为正数，但实际为 {window_size}")

    # 遍历所有动作类型文件夹
    action_folders_processed = 0
    for action_folder in os.listdir(base_dir):
        action_folder_path = os.path.join(base_dir, action_folder)

        # 确保是文件夹
        if not os.path.isdir(action_folder_path):
            print(f"跳过非文件夹: {action_folder}")
            continue

        # 规范化动作类型名称
        action_name = action_folder.lower()

        # 映射到标准动作类型
        # 首先尝试完全匹配
        if action_name in action_types:
            mapped_action = action_name
        else:
            # 如果没有完全匹配，尝试子字符串匹配
            mapped_action = None
            # 先检查较长的动作名称（包含其他动作名称的动作）
            longer_actions = ['climb_left', 'climb_right', 'dive_left', 'dive_right']
            for action in longer_actions:
                if action in action_name:
                    mapped_action = action
                    break
            # 如果没有匹配到较长的动作名称，再检查较短的动作名称
            if mapped_action is None:
                for standard_action in action_types:
                    if standard_action in action_name and standard_action not in longer_actions:
                        mapped_action = standard_action
                        break

        if mapped_action is None:
            print(f"警告: 文件夹 '{action_folder}' 未映射到任何已知动作类型，跳过。")
            continue

        action_folders_processed += 1
        print(f"处理动作类型: {mapped_action} ({action_folder})")

        # 为该动作生成独热编码标签
        action_label = action_to_onehot(mapped_action)

        # 按ID分组处理CSV文件
        files_by_id = {}
        csv_files_processed = 0

        # 遍历文件夹中的所有CSV文件
        for file in os.listdir(action_folder_path):
            if file.endswith('.csv'):
                csv_files_processed += 1
                file_path = os.path.join(action_folder_path, file)

                try:
                    # 读取CSV文件
                    df = pd.read_csv(file_path)

                    # 检查必需的列是否存在
                    required_columns = ['Time_s', 'Agent_ID', 'X_m', 'Y_m', 'Z_m', 'Velocity_m_s']
                    if not all(col in df.columns for col in required_columns):
                        print(f"警告: 文件 {file} 缺少必需的列，跳过。")
                        continue

                    # 按Agent_ID分组
                    for agent_id, group in df.groupby('Agent_ID'):
                        # 按时间排序
                        group_sorted = group.sort_values('Time_s')

                        # 提取所需特征：X, Y, Z, Velocity
                        features = group_sorted[['X_m', 'Y_m', 'Z_m', 'Velocity_m_s']].values

                        if len(features) > 0:
                            if agent_id not in files_by_id:
                                files_by_id[agent_id] = []
                            files_by_id[agent_id].append((file_path, features))
                except Exception as e:
                    print(f"处理文件 {file} 时出错: {e}")

        print(f"  处理了 {csv_files_processed} 个CSV文件")
        print(f"  识别到 {len(files_by_id)} 个不同的Agent_ID")

        # 对每个Agent_ID的所有数据进行处理
        samples_generated = 0
        for agent_id, file_features_list in files_by_id.items():
            # 将同一ID的所有文件数据合并
            all_features = []
            for file_path, features in file_features_list:
                all_features.extend(features)

            # 转换为numpy数组
            all_features = np.array(all_features)

            # 检查是否有足够的数据生成样本
            if len(all_features) >= window_size:
                # 使用滑动窗口生成样本
                num_samples = len(all_features) - window_size + 1
                for i in range(num_samples):
                    window = all_features[i:i + window_size].copy()  # 创建副本以避免引用问题
                    all_samples.append(window)
                    all_labels.append(action_label.copy())  # 创建副本以避免引用问题

                samples_generated += num_samples
            else:
                print(
                    f"  警告: Agent {agent_id} 的数据长度 ({len(all_features)}) 小于窗口大小 ({window_size})，无法生成样本。")

        print(f"  为动作类型 {mapped_action} 生成了 {samples_generated} 个样本")

        # 更新标签计数
        label_counts[mapped_action] = samples_generated

    print(f"总共处理了 {action_folders_processed} 个动作类型文件夹")

    # 转换为numpy数组
    all_samples = np.array(all_samples)
    all_labels = np.array(all_labels)

    print(f"总共生成样本数: {len(all_samples)}")
    print(f"样本形状: {all_samples.shape}")
    print(f"标签形状: {all_labels.shape}")

    # 显示每个标签的样本数量
    print("\n不同标签对应的样本数量:")
    total_samples = 0
    for action, count in label_counts.items():
        if count > 0:
            print(f"  {action}: {count} 个样本")
            total_samples += count

    # 检查是否有标签没有样本
    zero_count_actions = [action for action, count in label_counts.items() if count == 0]
    if zero_count_actions:
        print("\n以下标签没有生成任何样本:")
        for action in zero_count_actions:
            print(f"  {action}")

    print(f"\n实际总样本数: {total_samples}")

    # 保存处理后的数据
    np.savez(output_file, samples=all_samples, labels=all_labels)
    print(f"数据已保存到 {output_file}")

    # 同时保存动作类型映射，以便后续使用
    with open('action_types.json', 'w') as f:
        json.dump(action_types, f)

    return all_samples, all_labels


# 划分训练集、验证集和测试集
def split_data(samples, labels, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    # 检查样本数量是否足够
    if len(samples) == 0:
        raise ValueError("没有生成任何样本，请检查数据处理过程。")

    # 验证比例是否合理
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"划分比例之和应为1，但实际为{train_ratio + val_ratio + test_ratio}")

    # 首先划分训练集和剩余部分
    # 确保test_size不会导致训练集为空
    if len(samples) > 1:
        X_train, X_temp, y_train, y_temp = train_test_split(
            samples, labels, test_size=min(val_ratio + test_ratio, 0.95), random_state=42, shuffle=True
        )
    else:
        # 如果只有一个样本，全部用于训练集
        X_train, y_train = samples, labels
        X_temp, y_temp = np.array([]), np.array([])

    # 然后从剩余部分划分验证集和测试集
    if len(X_temp) > 1:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_ratio / (val_ratio + test_ratio) if val_ratio + test_ratio > 0 else 0.5,
            random_state=42, shuffle=True
        )
    elif len(X_temp) == 1:
        # 如果只有一个样本用于验证和测试，全部用于验证集
        X_val, y_val = X_temp, y_temp
        X_test, y_test = np.array([]), np.array([])
    else:
        # 如果没有样本用于验证和测试，创建空数组
        X_val, X_test, y_val, y_test = np.array([]), np.array([]), np.array([]), np.array([])

    print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}, 测试集大小: {len(X_test)}")

    # 保存划分后的数据
    np.savez('training_data.npz', X=X_train, y=y_train)
    np.savez('validation_data.npz', X=X_val, y=y_val)
    np.savez('test_data.npz', X=X_test, y=y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


# 从chuli2加载数据并进行动作识别的主函数 - 窗口长度50
def process_chuli2_data_and_identify_actions():
    """利用best_lstm_model.pth对chuli2数据中的(x3,y3,z3,v3)和(x4,y4,z4,v4)特征进行动作识别，将结果保存到chuli3"""
    # 设置基础目录
    base_directory = os.path.dirname(os.path.abspath(__file__))
    chuli2_dir = os.path.join(os.path.dirname(base_directory), 'chuli2')
    chuli3_dir = os.path.join(os.path.dirname(base_directory), 'chuli3')
    
    # 创建chuli3目录如果不存在
    os.makedirs(chuli3_dir, exist_ok=True)
    
    # 定义窗口大小为50
    window_size = 50
    
    print(f"开始处理chuli2中的数据，数据目录: {chuli2_dir}")
    print(f"窗口大小: {window_size}")

    # 加载chuli2中的数据
    chuli2_data_path = os.path.join(chuli2_dir, 'processed_time_series_data.npz')
    if not os.path.exists(chuli2_data_path):
        raise FileNotFoundError(f"找不到chuli2数据文件: {chuli2_data_path}")
    
    # 加载数据
    try:
        data = np.load(chuli2_data_path)
        # 尝试不同的数据键名
        if 'time_series_data' in data:
            time_series_data = data['time_series_data']
        elif 'data' in data:
            time_series_data = data['data']
        else:
            # 获取第一个数组数据
            time_series_data = data[list(data.keys())[0]]
        
        # 获取特征名称
        if 'column_names' in data:
            feature_names = data['column_names']
        elif 'feature_names' in data:
            feature_names = data['feature_names']
        else:
            feature_names = None
        
        # 检查数据结构
        print(f"成功加载chuli2数据，数据形状: {time_series_data.shape}")
        print(f"数据字典中的键: {list(data.keys())}")
        
        # 显示特征信息
        if feature_names is not None:
            print(f"特征数量: {len(feature_names)}")
            print("特征名称列表:")
            for i, name in enumerate(feature_names):
                print(f"  {i}: {name}")
        else:
            print("警告: 未找到特征名称")
        
        # 定义飞机3和飞机4的特征名称
        plane3_features = ['x3_B0100_m', 'y3_B0100_m', 'z3_B0100_m', 'v3_B0100_ms']
        plane4_features = ['x4_B0200_m', 'y4_B0200_m', 'z4_B0200_m', 'v4_B0200_ms']
        
        # 尝试找到飞机3和飞机4的特征索引
        plane3_indices = []
        plane4_indices = []
        
        # 如果有特征名称，尝试匹配
        if feature_names is not None:
            # 先尝试精确匹配
            for i, name in enumerate(feature_names):
                if name in plane3_features:
                    plane3_indices.append(i)
                    print(f"找到飞机3特征: {name} 索引: {i}")
                elif name in plane4_features:
                    plane4_indices.append(i)
                    print(f"找到飞机4特征: {name} 索引: {i}")
            
            # 如果精确匹配失败，尝试模糊匹配
            if len(plane3_indices) < 4 or len(plane4_indices) < 4:
                print("精确匹配失败，尝试模糊匹配")
                # 定义模糊匹配模式
                plane3_patterns = ['x3', 'y3', 'z3', 'v3']
                plane4_patterns = ['x4', 'y4', 'z4', 'v4']
                
                # 模糊匹配飞机3特征
                for pattern in plane3_patterns:
                    if len(plane3_indices) >= 4:
                        break
                    for i, name in enumerate(feature_names):
                        if pattern in name.lower() and i not in plane3_indices:
                            plane3_indices.append(i)
                            print(f"模糊匹配到飞机3特征: {pattern} 在 {name} 索引: {i}")
                            break
                
                # 模糊匹配飞机4特征
                for pattern in plane4_patterns:
                    if len(plane4_indices) >= 4:
                        break
                    for i, name in enumerate(feature_names):
                        if pattern in name.lower() and i not in plane4_indices:
                            plane4_indices.append(i)
                            print(f"模糊匹配到飞机4特征: {pattern} 在 {name} 索引: {i}")
                            break
        
        # 如果仍然找不到足够的特征索引，使用默认索引
        if len(plane3_indices) < 4:
            print("无法完全匹配飞机3特征，使用默认索引0-3")
            plane3_indices = list(range(4))
        if len(plane4_indices) < 4:
            print("无法完全匹配飞机4特征，使用默认索引4-7")
            plane4_indices = list(range(4, 8))
        
        # 确保索引有效
        if plane3_indices and max(plane3_indices) >= time_series_data.shape[2]:
            print("飞机3的特征索引超出数据维度，将使用前4个特征")
            plane3_indices = list(range(4))
        if plane4_indices and max(plane4_indices) >= time_series_data.shape[2]:
            print("飞机4的特征索引超出数据维度，将使用第5-8个特征")
            plane4_indices = list(range(4, 8))
        
        print(f"最终使用的飞机3特征索引: {plane3_indices}")
        print(f"最终使用的飞机4特征索引: {plane4_indices}")
        
        # 加载dongzuoshibie.py生成的best_lstm_model.pth模型
        model_path = os.path.join(base_directory, 'best_lstm_model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到预训练模型 {model_path}")
        
        try:
            # 加载模型到GPU或CPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = LSTMActionClassifier(input_size=4, hidden_size=128, num_layers=2, num_classes=11)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print(f"成功加载best_lstm_model.pth到 {device}")
            
            # 尝试加载归一化参数
            normalization_path = os.path.join(base_directory, 'normalization_params.npz')
            normalization_loaded = False
            if os.path.exists(normalization_path):
                try:
                    norm_params = np.load(normalization_path)
                    print(f"成功加载归一化参数: 均值={norm_params['mean']}, 标准差={norm_params['scale']}")
                    normalization_loaded = True
                except Exception as e:
                    print(f"加载归一化参数时出错: {e}")
            
        except Exception as e:
            print(f"加载模型时出错: {e}")
            raise
        
        # 创建增强后的数据集，添加22个特征：两个飞机各自的11个动作独热编码
        enhanced_data = np.zeros((time_series_data.shape[0], time_series_data.shape[1], 
                                  time_series_data.shape[2] + 2 * len(action_types)))
        
        # 复制原始数据
        enhanced_data[:, :, :time_series_data.shape[2]] = time_series_data
        
        # 为每个窗口进行动作识别
        print(f"开始对每个窗口进行动作识别，总共 {time_series_data.shape[0]} 个窗口")
        
        # 获取实际窗口长度
        actual_window_size = time_series_data.shape[1]
        print(f"数据窗口长度: {actual_window_size}")
        
        for i in range(time_series_data.shape[0]):
            # 提取当前窗口
            window = time_series_data[i]
            
            try:
                # 提取飞机3的4个特征 (x3,y3,z3,v3)
                plane3_features_data = window[:, plane3_indices[:4]]
                
                # 提取飞机4的4个特征 (x4,y4,z4,v4)
                plane4_features_data = window[:, plane4_indices[:4]]
                
                # 预测飞机3的动作
                # 对窗口内的数据单独进行归一化
                plane3_scaler = StandardScaler()
                plane3_normalized = plane3_scaler.fit_transform(plane3_features_data)
                
                # 转换为PyTorch张量并进行预测
                plane3_tensor = torch.tensor(plane3_normalized, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, plane3_probs = model(plane3_tensor)
                    # 获取概率最大的动作类型
                    plane3_pred_idx = torch.argmax(plane3_probs, dim=1).item()
                    plane3_pred_action = idx_to_action(plane3_pred_idx)
                
                # 生成飞机3的动作独热编码
                plane3_onehot = np.zeros(len(action_types))
                plane3_onehot[plane3_pred_idx] = 1
                
                # 预测飞机4的动作
                # 对窗口内的数据单独进行归一化
                plane4_scaler = StandardScaler()
                plane4_normalized = plane4_scaler.fit_transform(plane4_features_data)
                
                # 转换为PyTorch张量并进行预测
                plane4_tensor = torch.tensor(plane4_normalized, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, plane4_probs = model(plane4_tensor)
                    # 获取概率最大的动作类型
                    plane4_pred_idx = torch.argmax(plane4_probs, dim=1).item()
                    plane4_pred_action = idx_to_action(plane4_pred_idx)
                
                # 生成飞机4的动作独热编码
                plane4_onehot = np.zeros(len(action_types))
                plane4_onehot[plane4_pred_idx] = 1
                
                # 将两个飞机的动作独热编码添加到数据集中
                # 为窗口中的每个时间步都添加相同的动作编码（因为窗口代表一个完整动作）
                for t in range(actual_window_size):
                    # 添加飞机3的11维动作独热编码
                    enhanced_data[i, t, time_series_data.shape[2]:time_series_data.shape[2] + len(action_types)] = plane3_onehot
                    # 添加飞机4的11维动作独热编码
                    enhanced_data[i, t, time_series_data.shape[2] + len(action_types):] = plane4_onehot
                    
            except Exception as e:
                print(f"处理窗口 {i} 时出错: {e}，使用默认动作编码")
                # 出错时使用默认动作（平飞）的独热编码
                default_action_idx = action_types['level_flight']
                default_onehot = np.zeros(len(action_types))
                default_onehot[default_action_idx] = 1
                
                for t in range(actual_window_size):
                    enhanced_data[i, t, time_series_data.shape[2]:time_series_data.shape[2] + len(action_types)] = default_onehot
                    enhanced_data[i, t, time_series_data.shape[2] + len(action_types):] = default_onehot
            
            # 显示进度
            if (i + 1) % 100 == 0 or i == time_series_data.shape[0] - 1:
                print(f"已处理 {i + 1}/{time_series_data.shape[0]} 个窗口")
        
        # 创建新的特征名称列表
        new_feature_names = []
        if feature_names is not None:
            new_feature_names.extend(feature_names)
        else:
            # 如果没有原始特征名称，创建默认名称
            for i in range(time_series_data.shape[2]):
                new_feature_names.append(f'original_feature_{i}')
        
        # 添加飞机3的动作特征名称
        for action in action_types:
            new_feature_names.append(f'plane3_action_{action}')
        
        # 添加飞机4的动作特征名称
        for action in action_types:
            new_feature_names.append(f'plane4_action_{action}')
        
        # 保存增强后的数据到chuli3目录
        enhanced_output_dir = os.path.join(chuli3_dir, '带动作独热编码的数据集')
        os.makedirs(enhanced_output_dir, exist_ok=True)
        enhanced_output_path = os.path.join(enhanced_output_dir, 'enhanced_time_series_data_with_actions.npz')
        
        try:
            # 使用savez_compressed以减小文件大小
            np.savez_compressed(
                enhanced_output_path,
                data=enhanced_data,
                feature_names=np.array(new_feature_names)
            )
            print(f"增强后的数据已保存到 {enhanced_output_path}")
            print(f"增强后的特征数量: {enhanced_data.shape[2]}")
            print(f"原始特征数: {time_series_data.shape[2]}")
            print(f"新增动作特征数: {enhanced_data.shape[2] - time_series_data.shape[2]} (飞机3和飞机4各11维独热编码)")
            print(f"数据形状: {enhanced_data.shape}")
            print(f"窗口长度: {enhanced_data.shape[1]}")
            print(f"样本数量: {enhanced_data.shape[0]}")
            
            # 生成数据集信息文件
            info_path = os.path.join(chuli3_dir, 'dataset_info.txt')
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"数据集信息 - 带动作识别结果的时间序列数据\n")
                f.write(f"- 数据形状: {enhanced_data.shape}\n")
                f.write(f"- 样本数量: {enhanced_data.shape[0]}\n")
                f.write(f"- 窗口长度: {enhanced_data.shape[1]}\n")
                f.write(f"- 原始特征数: {time_series_data.shape[2]}\n")
                f.write(f"- 新增动作特征数: {enhanced_data.shape[2] - time_series_data.shape[2]}\n")
                f.write(f"- 总特征数: {enhanced_data.shape[2]}\n")
                f.write(f"- 动作特征说明: 包含飞机3和飞机4各自的11维动作类型独热编码\n")
                f.write(f"- 使用模型: best_lstm_model.pth\n\n")
                f.write("特征列表 (前20个):\n")
                for i, name in enumerate(new_feature_names[:20]):
                    f.write(f"  {i+1}. {name}\n")
                if len(new_feature_names) > 20:
                    f.write(f"  ... 以及其他 {len(new_feature_names) - 20} 个特征\n")
            print(f"数据集信息已保存到: {info_path}")
            
        except Exception as e:
            print(f"保存数据时出错: {e}")
            raise
            
        except Exception as e:
            print(f"保存数据时出错: {e}")
            raise
        
        return enhanced_data, np.array(new_feature_names)
        
    except Exception as e:
        print(f"处理chuli2数据时出错: {e}")
        raise


def process_labeled_data():
    """处理带标签的动作数据并划分数据集"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    window_size = 50
    
    print("\n=== 开始处理带标签的动作数据 ===")
    print(f"窗口长度设置为: {window_size}")
    
    # 处理带标签的数据
    samples, labels = process_data(base_dir, window_size=window_size)
    
    # 划分数据集
    print("\n=== 开始划分数据集 ===")
    train_X, train_y, val_X, val_y, test_X, test_y = split_data(samples, labels, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
    print(f"\n数据集划分完成:")
    print(f"- 训练集样本数: {len(train_X)}")
    print(f"- 验证集样本数: {len(val_X)}")
    print(f"- 测试集样本数: {len(test_X)}")
    
    return True

def main():
    """主函数：处理chuli2数据并添加动作识别结果"""
    try:
        # 直接处理chuli2数据并添加动作识别结果（不处理带标签数据）
        print("=== 开始处理chuli2数据并添加动作识别结果 ===")
        enhanced_data, feature_names = process_chuli2_data_and_identify_actions()
        
        print("\n=== 处理完成！===")
        print(f"已将增强后的数据集保存到chuli3文件夹下的'带动作独热编码的数据集'子目录")
        print(f"增强后的数据包含原始特征 + 飞机3的11维动作独热编码 + 飞机4的11维动作独热编码")
        print(f"总共新增22维特征，用于强化数据集")
        print(f"动作识别基于dongzuoshibie.py生成的best_lstm_model.pth模型")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()