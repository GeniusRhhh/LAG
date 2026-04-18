import numpy as np
import os
import json
from collections import Counter

def get_mode_with_last_occurrence(values):
    """
    计算数组的众数，如果有多个众数则返回最后出现的那个
    
    Args:
        values: 输入数组
        
    Returns:
        众数
    """
    # 如果数组为空，返回None
    if len(values) == 0:
        return None
    
    # 计算每个值的出现次数
    counter = Counter(values)
    
    # 找出最大出现次数
    max_count = max(counter.values())
    
    # 找出所有众数（出现次数等于最大次数的值）
    modes = [value for value, count in counter.items() if count == max_count]
    
    # 如果只有一个众数，直接返回
    if len(modes) == 1:
        return modes[0]
    
    # 如果有多个众数，返回最后出现的那个
    last_occurrence = -1
    last_mode = modes[0]
    
    for mode in modes:
        # 从后往前找，找到第一个出现的位置
        for i in range(len(values) - 1, -1, -1):
            if values[i] == mode:
                if i > last_occurrence:
                    last_occurrence = i
                    last_mode = mode
                break
    
    return last_mode

def class_to_one_hot(class_idx):
    """
    将类别索引转换为独热编码
    
    Args:
        class_idx: 类别索引（0-5）
        
    Returns:
        长度为6的独热编码数组
    """
    # 确保class_idx是整数
    class_idx = int(class_idx)
    # 创建独热编码
    one_hot = np.zeros(6, dtype=np.float64)
    if 0 <= class_idx <= 5:
        one_hot[class_idx] = 1.0
    return one_hot

def process_dataset_labels(input_path, output_path):
    """
    处理数据集中的标签，计算每个窗口的众数并统一标签
    
    Args:
        input_path: 输入数据集路径
        output_path: 输出数据集路径
        
    Returns:
        处理后的标签统计信息
    """
    print(f"正在加载数据集: {input_path}")
    
    # 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 找不到数据集文件 {input_path}")
        return None
    
    # 加载数据集
    try:
        data = np.load(input_path)
        samples = data['samples']
        labels = data['labels']
        print(f"数据集加载成功")
        print(f"样本形状: {samples.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"标签维度: {labels.ndim}")
    except Exception as e:
        print(f"加载数据集时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    # 创建新的标签数组
    new_labels = labels.copy()
    
    # 统计信息
    intent_b0100_counts = {}
    intent_b0200_counts = {}
    error_count = 0
    success_count = 0
    
    total_windows = samples.shape[0]
    print(f"总窗口数: {total_windows}")
    
    # 处理每个窗口
    for window_idx in range(total_windows):
        if window_idx % 1000 == 0:
            print(f"处理窗口 {window_idx}/{total_windows}, 成功: {success_count}, 错误: {error_count}")
        
        try:
            # 获取当前窗口的样本和标签
            window_samples = samples[window_idx]
            window_labels = labels[window_idx]
            
            # 检查标签维度
            if len(window_labels.shape) == 2:
                # 如果标签是二维的，说明每个时间步都有标签
                num_time_steps = window_labels.shape[0]
                
                # 提取intent_B0100的值（索引30）
                intent_b0100_values = window_labels[:, 30]
                # 提取intent_B0200的值（索引37）
                intent_b0200_values = window_labels[:, 37]
                
                # 计算众数
                mode_b0100 = get_mode_with_last_occurrence(intent_b0100_values)
                mode_b0200 = get_mode_with_last_occurrence(intent_b0200_values)
                
                # 验证众数是否在有效范围内
                if mode_b0100 is not None and 0 <= mode_b0100 <= 5:
                    # 生成对应的独热编码
                    b0100_one_hot = class_to_one_hot(mode_b0100)
                    # 更新所有时间步的标签
                    for t in range(num_time_steps):
                        new_labels[window_idx, t, 30] = mode_b0100
                        new_labels[window_idx, t, 31:37] = b0100_one_hot
                
                if mode_b0200 is not None and 0 <= mode_b0200 <= 5:
                    # 生成对应的独热编码
                    b0200_one_hot = class_to_one_hot(mode_b0200)
                    # 更新所有时间步的标签
                    for t in range(num_time_steps):
                        new_labels[window_idx, t, 37] = mode_b0200
                        new_labels[window_idx, t, 38:43] = b0200_one_hot
                
                # 转换为整数键用于统计
                b0100_key = int(mode_b0100) if mode_b0100 is not None else 'None'
                b0200_key = int(mode_b0200) if mode_b0200 is not None else 'None'
                
                # 更新统计计数
                intent_b0100_counts[b0100_key] = intent_b0100_counts.get(b0100_key, 0) + 1
                intent_b0200_counts[b0200_key] = intent_b0200_counts.get(b0200_key, 0) + 1
                
                success_count += 1
            else:
                print(f"窗口 {window_idx} 标签维度不符合预期: {window_labels.shape}")
                error_count += 1
                
        except Exception as e:
            print(f"处理窗口 {window_idx} 时出错: {str(e)}")
            error_count += 1
            # 继续处理下一个窗口
            continue
    
    # 保存处理后的数据集
    try:
        np.savez(output_path, samples=samples, labels=new_labels)
        print(f"处理后的数据集已保存到: {output_path}")
    except Exception as e:
        print(f"保存数据集时出错: {str(e)}")
        return None
    
    # 计算占比
    total_processed = success_count
    intent_b0100_stats = {
        "unique_labels": list(intent_b0100_counts.keys()),
        "label_counts": intent_b0100_counts,
        "label_percentages": {k: round(v/total_processed*100, 2) if total_processed > 0 else 0 for k, v in intent_b0100_counts.items()},
        "total_samples": total_processed
    }
    
    intent_b0200_stats = {
        "unique_labels": list(intent_b0200_counts.keys()),
        "label_counts": intent_b0200_counts,
        "label_percentages": {k: round(v/total_processed*100, 2) if total_processed > 0 else 0 for k, v in intent_b0200_counts.items()},
        "total_samples": total_processed
    }
    
    # 保存统计信息
    stats = {
        "intent_B0100": intent_b0100_stats,
        "intent_B0200": intent_b0200_stats,
        "total_windows": total_windows,
        "success_count": success_count,
        "error_count": error_count
    }
    
    stats_file = os.path.join(os.path.dirname(output_path), "processed_label_statistics.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        # 确保所有numpy类型都转换为Python原生类型
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return int(obj) if isinstance(obj, np.integer) else float(obj)
            elif isinstance(obj, dict):
                return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        converted_stats = convert_numpy_types(stats)
        json.dump(converted_stats, f, ensure_ascii=False, indent=2)
    
    print(f"统计信息已保存到: {stats_file}")
    
    # 打印统计结果
    print("\n===== 处理结果统计 =====")
    print(f"总窗口数: {total_windows}")
    print(f"成功处理: {success_count}")
    print(f"处理失败: {error_count}")
    
    print("\nintent_B0100 标签分布:")
    for label in intent_b0100_stats["unique_labels"]:
        count = intent_b0100_stats["label_counts"][label]
        percentage = intent_b0100_stats["label_percentages"][label]
        print(f"  标签 {label}: {count}个窗口 ({percentage}%)")
    
    print("\nintent_B0200 标签分布:")
    for label in intent_b0200_stats["unique_labels"]:
        count = intent_b0200_stats["label_counts"][label]
        percentage = intent_b0200_stats["label_percentages"][label]
        print(f"  标签 {label}: {count}个窗口 ({percentage}%)")
    
    return stats

def verify_processing(input_path, output_path):
    """
    验证处理结果
    
    Args:
        input_path: 原始数据集路径
        output_path: 处理后的数据集路径
    """
    print(f"\n正在验证处理结果...")
    
    try:
        # 加载原始数据
        orig_data = np.load(input_path)
        orig_labels = orig_data['labels']
        
        # 加载处理后的数据
        proc_data = np.load(output_path)
        proc_labels = proc_data['labels']
        
        # 随机选择几个窗口进行验证
        num_windows_to_check = min(5, orig_labels.shape[0])
        for i in range(num_windows_to_check):
            print(f"\n验证窗口 {i}:")
            
            # 检查每个时间步的标签是否一致
            all_b0100_same = np.all(proc_labels[i, :, 30] == proc_labels[i, 0, 30])
            all_b0200_same = np.all(proc_labels[i, :, 37] == proc_labels[i, 0, 30])
            
            print(f"  所有时间步intent_B0100标签一致: {all_b0100_same}")
            print(f"  所有时间步intent_B0200标签一致: {all_b0200_same}")
            print(f"  intent_B0100值: {proc_labels[i, 0, 30]}")
            print(f"  intent_B0100独热编码: {proc_labels[i, 0, 31:37]}")
            print(f"  intent_B0200值: {proc_labels[i, 0, 37]}")
            print(f"  intent_B0200独热编码: {proc_labels[i, 0, 38:43]}")
    
    except Exception as e:
        print(f"验证过程中出错: {str(e)}")

if __name__ == "__main__":
    print("开始处理数据集标签...")
    
    # 定义文件路径
    input_dataset = "C:\\Users\\SONG\\Desktop\\yitushibie\\chuli3\\带动作独热编码的数据集\\enhanced_time_series_data_with_actions.npz"
    output_dataset = "C:\\Users\\SONG\\Desktop\\yitushibie\\chuli4\\processed_data_with_unified_labels.npz"
    
    # 处理数据集
    stats = process_dataset_labels(input_dataset, output_dataset)
    
    # 验证处理结果
    if stats:
        verify_processing(input_dataset, output_dataset)
    
    print("\n标签处理完成!")