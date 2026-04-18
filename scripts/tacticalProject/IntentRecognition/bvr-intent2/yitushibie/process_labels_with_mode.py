import numpy as np
import json
import os

def get_mode_with_last_occurrence(values):
    """
    计算众数，如果有多个众数，选择最后出现的那个
    
    Args:
        values: 数值列表
        
    Returns:
        众数
    """
    if len(values) == 0:
        return None
    
    # 统计每个值出现的次数和最后出现的位置
    value_counts = {}
    last_occurrence = {}
    
    for i, val in enumerate(values):
        val = float(val)  # 转换为float以确保一致性
        if val not in value_counts:
            value_counts[val] = 0
        value_counts[val] += 1
        last_occurrence[val] = i
    
    # 找到最大出现次数
    max_count = max(value_counts.values())
    
    # 找出所有众数
    modes = [val for val, count in value_counts.items() if count == max_count]
    
    # 如果有多个众数，选择最后出现的那个
    if len(modes) > 1:
        return max(modes, key=lambda x: last_occurrence[x])
    else:
        return modes[0]

def class_to_one_hot(class_idx, num_classes=6):
    """
    将类别索引转换为独热编码
    
    Args:
        class_idx: 类别索引（0-5）
        num_classes: 类别数量，默认为6
        
    Returns:
        独热编码数组
    """
    one_hot = np.zeros(num_classes, dtype=np.float32)
    one_hot[int(class_idx)] = 1.0
    return one_hot

def process_dataset_labels(dataset_path="processed_data.npz", output_path="processed_data_unified_labels.npz"):
    """
    处理数据集中的标签，计算每个窗口的众数并统一标签
    
    Args:
        dataset_path: 输入数据集路径
        output_path: 输出数据集路径
        
    Returns:
        处理后的标签统计信息
    """
    print(f"正在加载数据集: {dataset_path}")
    
    # 检查文件是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: 找不到数据集文件 {dataset_path}")
        return None
    
    # 加载数据集
    try:
        data = np.load(dataset_path)
        samples = data['samples']
        labels = data['labels']
        print(f"数据集加载成功")
        print(f"样本形状: {samples.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"标签维度: {labels.ndim}")
        
        # 打印前几个标签样本以了解结构
        print("\n前5个标签样本:")
        for i in range(min(5, len(labels))):
            print(f"样本 {i} 标签形状: {labels[i].shape}, 内容: {labels[i]}")
            
        # 检查标签长度
        print(f"\n标签向量长度: {len(labels[0]) if len(labels) > 0 else '未知'}")
        if len(labels) > 0 and len(labels[0]) > 43:
            print("标签向量足够长，可以访问索引30和37")
        else:
            print(f"警告: 标签向量长度为{len(labels[0]) if len(labels) > 0 else '未知'}，可能无法访问索引30和37")
            
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
    
    total_windows = samples.shape[0]
    print(f"总窗口数: {total_windows}")
    
    # 处理每个窗口
    for window_idx in range(total_windows):
        if window_idx % 1000 == 0:
            print(f"处理窗口 {window_idx}/{total_windows}, 错误数: {error_count}")
        
        # 获取当前窗口的标签
        window_labels = labels[window_idx]
        
        try:
            # 直接统计intent_B0100和intent_B0200的值，不进行修改
            # 因为我们需要先了解实际的数据结构
            
            # 打印前5个窗口的详细信息
            if window_idx < 5:
                print(f"\n窗口 {window_idx} 标签详情:")
                print(f"标签形状: {window_labels.shape}")
                print(f"标签向量长度: {len(window_labels)}")
                
                # 尝试访问不同索引位置的值
                if len(window_labels) > 10:
                    print(f"前10个元素: {window_labels[:10]}")
                
                # 检查是否可以访问索引30和37
                if len(window_labels) > 37:
                    print(f"索引30的值: {window_labels[30]}")
                    print(f"索引31-36的值: {window_labels[31:37]}")
                    print(f"索引37的值: {window_labels[37]}")
                    print(f"索引38-43的值: {window_labels[38:43]}")
            
            # 尝试获取标签值并统计
            if len(window_labels) > 37:
                # 获取标签值
                intent_b0100_value = window_labels[30]
                intent_b0200_value = window_labels[37]
                
                # 转换为整数键用于统计
                b0100_key = int(intent_b0100_value) if isinstance(intent_b0100_value, (int, np.integer, float, np.floating)) else str(intent_b0100_value)
                b0200_key = int(intent_b0200_value) if isinstance(intent_b0200_value, (int, np.integer, float, np.floating)) else str(intent_b0200_value)
                
                # 更新统计计数
                intent_b0100_counts[b0100_key] = intent_b0100_counts.get(b0100_key, 0) + 1
                intent_b0200_counts[b0200_key] = intent_b0200_counts.get(b0200_key, 0) + 1
                
                # 生成并设置对应的独热编码
                if len(window_labels) > 36:
                    b0100_one_hot = class_to_one_hot(intent_b0100_value)
                    new_labels[window_idx, 31:37] = b0100_one_hot
                
                if len(window_labels) > 43:
                    b0200_one_hot = class_to_one_hot(intent_b0200_value)
                    new_labels[window_idx, 38:43] = b0200_one_hot
            else:
                print(f"窗口 {window_idx} 标签长度不足，无法访问索引30和37")
                error_count += 1
                
        except Exception as e:
            print(f"处理窗口 {window_idx} 时出错: {str(e)}")
            error_count += 1
            # 不抛出异常，继续处理下一个窗口
            continue
    
    # 保存处理后的数据集
    try:
        np.savez(output_path, samples=samples, labels=new_labels)
        print(f"处理后的数据集已保存到: {output_path}")
    except Exception as e:
        print(f"保存数据集时出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 计算占比
    total = total_windows
    intent_b0100_stats = {
        "unique_labels": list(intent_b0100_counts.keys()),
        "label_counts": {str(k): v for k, v in intent_b0100_counts.items()},
        "label_percentages": {str(k): round(v/total*100, 2) for k, v in intent_b0100_counts.items()},
        "total_samples": total
    }
    
    intent_b0200_stats = {
        "unique_labels": list(intent_b0200_counts.keys()),
        "label_counts": {str(k): v for k, v in intent_b0200_counts.items()},
        "label_percentages": {str(k): round(v/total*100, 2) for k, v in intent_b0200_counts.items()},
        "total_samples": total
    }
    
    # 保存统计信息
    stats = {
        "intent_B0100": intent_b0100_stats,
        "intent_B0200": intent_b0200_stats,
        "total_samples": total
    }
    
    with open("unified_label_distribution.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print("\n===== 标签统计结果 =====")
    print("\nintent_B0100:")
    for label in intent_b0100_stats["unique_labels"]:
        count = intent_b0100_stats["label_counts"][str(label)]
        percentage = intent_b0100_stats["label_percentages"][str(label)]
        print(f"  标签 {label}: {count}个样本 ({percentage}%)")
    
    print("\nintent_B0200:")
    for label in intent_b0200_stats["unique_labels"]:
        count = intent_b0200_stats["label_counts"][str(label)]
        percentage = intent_b0200_stats["label_percentages"][str(label)]
        print(f"  标签 {label}: {count}个样本 ({percentage}%)")
    
    print(f"\n统计信息已保存到: unified_label_distribution.json")
    
    return stats

def verify_processing(original_path="processed_data.npz", processed_path="processed_data_unified_labels.npz", sample_indices=[0, 1, 2, 3, 4]):
    """
    验证处理后的数据集，检查几个样本窗口
    
    Args:
        original_path: 原始数据集路径
        processed_path: 处理后的数据集路径
        sample_indices: 要检查的样本索引列表
    """
    if not os.path.exists(processed_path):
        print(f"错误: 找不到处理后的数据集文件 {processed_path}")
        return
    
    try:
        original_data = np.load(original_path)
        processed_data = np.load(processed_path)
        
        original_labels = original_data['labels']
        processed_labels = processed_data['labels']
        
        print("\n===== 处理验证 =====")
        for idx in sample_indices:
            if idx >= original_labels.shape[0]:
                continue
                
            print(f"\n窗口 {idx}:")
            
            # 检查intent_B0100（索引30）
            orig_b0100 = original_labels[idx, 30]
            proc_b0100 = processed_labels[idx, 30]
            print(f"  intent_B0100 原始值: {orig_b0100}, 处理后值: {proc_b0100}")
            
            # 检查intent_B0100对应的独热编码（索引31-36）
            orig_b0100_one_hot = original_labels[idx, 31:37]
            proc_b0100_one_hot = processed_labels[idx, 31:37]
            print(f"  intent_B0100 原始独热编码: {orig_b0100_one_hot}")
            print(f"  intent_B0100 处理后独热编码: {proc_b0100_one_hot}")
            
            # 检查独热编码是否与标签值对应
            expected_one_hot = class_to_one_hot(proc_b0100)
            is_correct_b0100 = np.allclose(proc_b0100_one_hot, expected_one_hot)
            print(f"  intent_B0100 独热编码是否正确: {is_correct_b0100}")
            
            # 检查intent_B0200（索引37）
            orig_b0200 = original_labels[idx, 37]
            proc_b0200 = processed_labels[idx, 37]
            print(f"  intent_B0200 原始值: {orig_b0200}, 处理后值: {proc_b0200}")
            
            # 检查intent_B0200对应的独热编码（索引38-43）
            orig_b0200_one_hot = original_labels[idx, 38:43]
            proc_b0200_one_hot = processed_labels[idx, 38:43]
            print(f"  intent_B0200 原始独热编码: {orig_b0200_one_hot}")
            print(f"  intent_B0200 处理后独热编码: {proc_b0200_one_hot}")
            
            # 检查独热编码是否与标签值对应
            expected_one_hot = class_to_one_hot(proc_b0200)
            is_correct_b0200 = np.allclose(proc_b0200_one_hot, expected_one_hot)
            print(f"  intent_B0200 独热编码是否正确: {is_correct_b0200}")
            
    except Exception as e:
        print(f"验证时出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("开始处理数据集标签...")
    
    # 只进行统计分析，不修改标签（先了解数据结构）
    # 创建一个新的统计分析函数调用
    try:
        data = np.load("processed_data.npz")
        labels = data['labels']
        
        # 直接统计intent_B0100和intent_B0200的值
        intent_b0100_counts = {}
        intent_b0200_counts = {}
        error_count = 0
        
        print("\n开始直接统计标签分布...")
        for i in range(len(labels)):
            if i % 1000 == 0:
                print(f"统计窗口 {i}/{len(labels)}")
            
            try:
                window_labels = labels[i]
                if len(window_labels) > 37:
                    # 获取标签值
                    b0100_val = window_labels[30]
                    b0200_val = window_labels[37]
                    
                    # 转换为整数键
                    b0100_key = int(b0100_val) if isinstance(b0100_val, (int, np.integer, float, np.floating)) else str(b0100_val)
                    b0200_key = int(b0200_val) if isinstance(b0200_val, (int, np.integer, float, np.floating)) else str(b0200_val)
                    
                    # 更新统计
                    intent_b0100_counts[b0100_key] = intent_b0100_counts.get(b0100_key, 0) + 1
                    intent_b0200_counts[b0200_key] = intent_b0200_counts.get(b0200_key, 0) + 1
                else:
                    error_count += 1
            except Exception as e:
                error_count += 1
                continue
        
        # 计算占比
        total = len(labels)
        intent_b0100_stats = {
            "unique_labels": list(intent_b0100_counts.keys()),
            "label_counts": intent_b0100_counts,
            "label_percentages": {k: round(v/total*100, 2) for k, v in intent_b0100_counts.items()},
            "total_samples": total
        }
        
        intent_b0200_stats = {
            "unique_labels": list(intent_b0200_counts.keys()),
            "label_counts": intent_b0200_counts,
            "label_percentages": {k: round(v/total*100, 2) for k, v in intent_b0200_counts.items()},
            "total_samples": total
        }
        
        # 保存统计信息
        stats = {
            "intent_B0100": intent_b0100_stats,
            "intent_B0200": intent_b0200_stats,
            "total_samples": total,
            "error_count": error_count
        }
        
        with open("label_distribution_direct.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print("\n===== 直接统计结果 =====")
        print(f"统计完成，错误数: {error_count}")
        print("\nintent_B0100 标签分布:")
        for label in intent_b0100_stats["unique_labels"]:
            count = intent_b0100_stats["label_counts"][label]
            percentage = intent_b0100_stats["label_percentages"][label]
            print(f"  标签 {label}: {count}个样本 ({percentage}%)")
        
        print("\nintent_B0200 标签分布:")
        for label in intent_b0200_stats["unique_labels"]:
            count = intent_b0200_stats["label_counts"][label]
            percentage = intent_b0200_stats["label_percentages"][label]
            print(f"  标签 {label}: {count}个样本 ({percentage}%)")
        
        print("\n统计信息已保存到: label_distribution_direct.json")
        
    except Exception as e:
        print(f"统计过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n标签统计分析完成!")