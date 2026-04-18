import numpy as np
import json
from collections import Counter
import os

# 数据集路径和保存路径（使用原始字符串避免Unicode转义问题）
input_dataset = r"C:\Users\SONG\Desktop\yitushibie\chuli3\带动作独热编码的数据集\enhanced_time_series_data_with_actions.npz"
output_dir = r"C:\Users\SONG\Desktop\yitushibie\chuli4"
output_dataset = os.path.join(output_dir, "data_with_unified_labels.npz")
output_distribution = os.path.join(output_dir, "label_distribution.json")

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"创建输出目录: {output_dir}")

# 计算众数的函数，当出现平局时返回最后出现的类别
def get_mode_with_last_occurrence(values):
    # 过滤掉无效值（如NaN）
    valid_values = [v for v in values if not np.isnan(v)]
    if not valid_values:
        return 0  # 默认返回0
    
    # 计算每个值出现的次数
    counter = Counter(valid_values)
    max_count = max(counter.values())
    
    # 找出所有出现次数最多的值
    modes = [value for value, count in counter.items() if count == max_count]
    
    # 如果只有一个众数，直接返回
    if len(modes) == 1:
        return modes[0]
    
    # 出现平局时，返回在原始列表中最后出现的值
    last_occurrence = -1
    selected_mode = modes[0]
    
    for mode in modes:
        # 从后往前找最后一次出现的位置
        for i in range(len(valid_values) - 1, -1, -1):
            if valid_values[i] == mode:
                if i > last_occurrence:
                    last_occurrence = i
                    selected_mode = mode
                break
    
    return selected_mode

# 将类别转换为独热编码
def class_to_one_hot(class_value, num_classes=6):
    one_hot = np.zeros(num_classes)
    class_idx = int(class_value)
    if 0 <= class_idx < num_classes:
        one_hot[class_idx] = 1
    return one_hot

# 处理数据集标签
def process_dataset_labels():
    print(f"正在处理数据集: {input_dataset}")
    
    # 标签分布统计
    b0100_counts = Counter()
    b0200_counts = Counter()
    
    try:
        # 加载数据集
        data = np.load(input_dataset)
        print(f"数据集加载成功")
        print(f"数据集中的所有键: {list(data.keys())}")
        
        # 获取数据和特征名称
        time_series_data = data['data']
        feature_names = data['feature_names'] if 'feature_names' in data else None
        
        print(f"数据形状: {time_series_data.shape}")
        total_windows = time_series_data.shape[0]
        print(f"总窗口数: {total_windows}")
        
        # 创建修改后的数据副本
        modified_data = time_series_data.copy()
        
        # 处理每个窗口
        for window_idx in range(total_windows):
            if window_idx % 1000 == 0:
                print(f"处理窗口 {window_idx}/{total_windows}")
            
            window = modified_data[window_idx]
            
            # 提取intent_B0100（索引30）和intent_B0200（索引37）的值
            b0100_values = window[:, 29]
            b0200_values = window[:, 36]
            
            # 计算众数
            b0100_mode = get_mode_with_last_occurrence(b0100_values)
            b0200_mode = get_mode_with_last_occurrence(b0200_values)
            
            # 更新统计
            b0100_counts[b0100_mode] += 1
            b0200_counts[b0200_mode] += 1
            
            # 生成独热编码
            b0100_one_hot = class_to_one_hot(b0100_mode)
            b0200_one_hot = class_to_one_hot(b0200_mode)
            
            # 统一标签值和独热编码
            for timestep in range(window.shape[0]):
                # 更新标签值
                modified_data[window_idx, timestep, 29] = b0100_mode
                modified_data[window_idx, timestep, 36] = b0200_mode
                
                # 更新独热编码
                modified_data[window_idx, timestep, 30:36] = b0100_one_hot
                modified_data[window_idx, timestep, 37:43] = b0200_one_hot
        
        # 保存修改后的数据集
        save_dict = {'data': modified_data}
        if feature_names is not None:
            save_dict['feature_names'] = feature_names
        
        np.savez_compressed(output_dataset, **save_dict)
        print(f"修改后的数据集已保存至: {output_dataset}")
        
        # 计算百分比并保存分布信息
        total_samples = total_windows
        b0100_percentages = {int(k): (v/total_samples)*100 for k, v in b0100_counts.items()}
        b0200_percentages = {int(k): (v/total_samples)*100 for k, v in b0200_counts.items()}
        
        distribution_info = {
            'intent_B0100': {
                'unique_labels': list(map(int, b0100_counts.keys())),
                'label_counts': {int(k): v for k, v in b0100_counts.items()},
                'label_percentages': b0100_percentages
            },
            'intent_B0200': {
                'unique_labels': list(map(int, b0200_counts.keys())),
                'label_counts': {int(k): v for k, v in b0200_counts.items()},
                'label_percentages': b0200_percentages
            },
            'total_samples': total_samples
        }
        
        with open(output_distribution, 'w', encoding='utf-8') as f:
            json.dump(distribution_info, f, ensure_ascii=False, indent=2)
        
        print(f"标签分布信息已保存至: {output_distribution}")
        print("\n标签分布统计:")
        print(f"intent_B0100 分布: {b0100_counts}")
        print(f"intent_B0200 分布: {b0200_counts}")
        
    except Exception as e:
        print(f"处理数据集时出错: {str(e)}")
        import traceback
        traceback.print_exc()

# 主函数
if __name__ == "__main__":
    process_dataset_labels()