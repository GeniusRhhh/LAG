import numpy as np
import json

def one_hot_to_class_index(one_hot_vector):
    """
    将独热编码向量转换为类别索引
    """
    if isinstance(one_hot_vector, np.ndarray) and len(one_hot_vector.shape) > 0:
        # 找到最大值的索引
        max_index = np.argmax(one_hot_vector)
        # 检查是否真的是独热编码（最大值为1，其余为0）
        if one_hot_vector[max_index] == 1.0 and np.sum(one_hot_vector) == 1.0:
            return max_index
    # 如果不是标准独热编码，返回最大值索引
    return np.argmax(one_hot_vector)

def get_mode_with_last_occurrence(labels, is_one_hot=False):
    """
    计算标签序列的众数，如果有多个众数（相同最高频率），则返回最后出现的标签
    
    参数:
    labels: 标签序列
    is_one_hot: 是否为独热编码格式
    """
    if len(labels) == 0:
        return None
    
    # 统计每个标签出现的次数和最后出现的位置
    label_counts = {}
    last_occurrence = {}
    
    for idx, label in enumerate(labels):
        # 如果是独热编码，先转换为类别索引
        if is_one_hot:
            processed_label = one_hot_to_class_index(label)
        else:
            processed_label = label
        
        if processed_label not in label_counts:
            label_counts[processed_label] = 0
        label_counts[processed_label] += 1
        last_occurrence[processed_label] = idx
    
    # 找出最高频率
    max_count = max(label_counts.values())
    
    # 找出所有具有最高频率的标签
    mode_candidates = [label for label, count in label_counts.items() if count == max_count]
    
    # 如果只有一个众数，直接返回
    if len(mode_candidates) == 1:
        return mode_candidates[0]
    
    # 如果有多个众数，返回最后出现的那个
    latest_label = None
    latest_pos = -1
    for label in mode_candidates:
        if last_occurrence[label] > latest_pos:
            latest_pos = last_occurrence[label]
            latest_label = label
    
    return latest_label

def load_and_analyze_dataset():
    """
    加载数据集并分析标签种类，特别关注intent_B0100和intent_B0200标签以及后面的6位独热编码
    """
    try:
        # 直接加载processed_data.npz数据集
        data_path = 'c:\\Users\\SONG\\Desktop\\yitushibie\\processed_data.npz'
        print(f"尝试加载数据集: {data_path}")
        data = np.load(data_path)
        
        print(f"成功加载数据集: {data_path}")
        print(f"数据集包含的键: {list(data.keys())}")
        
        # 详细打印数据信息
        for key in data.keys():
            value = data[key]
            print(f"键 '{key}': 形状 {value.shape}, 类型 {type(value)}, 数据类型 {value.dtype}")
            
            # 打印前几个元素以了解数据结构
            if len(value.shape) > 0:
                print(f"  前5个元素示例: {value[:5] if len(value) > 5 else value}")
        
        # 分析标签数据
        if 'labels' in data:
            # 处理标签数据
            y_data = data['labels']
            print(f"\n开始分析标签数据...")
            print(f"标签数据形状: {y_data.shape}")
            
            # 分析intent_B0100和intent_B0200标签以及后面的6位独热编码
            # 假设标签格式为 [samples, 11]，其中前两个是intent_B0100和intent_B0200，接下来6个是独热编码
            if len(y_data.shape) == 2 and y_data.shape[1] >= 8:
                print(f"\n开始分析intent_B0100、intent_B0200和独热编码标签...")
                print(f"标签维度: {y_data.shape[1]}")
                
                # 提取intent_B0100标签 (第1列)
                intent_b0100 = y_data[:, 0]
                # 提取intent_B0200标签 (第2列)
                intent_b0200 = y_data[:, 1]
                # 提取后面的6位独热编码 (第3-8列)
                one_hot_encoding = y_data[:, 2:8] if y_data.shape[1] >= 8 else y_data[:, 2:]
                
                # 分析intent_B0100
                print(f"\nIntent_B0100标签分析:")
                unique_b0100 = np.unique(intent_b0100)
                print(f"  唯一标签: {unique_b0100}")
                print(f"  标签种类数量: {len(unique_b0100)}")
                
                # 统计intent_B0100标签分布
                b0100_counts = {}
                for label in intent_b0100:
                    # 转换为Python原生类型
                    key = int(label) if isinstance(label, (np.integer, np.floating)) else label
                    if key not in b0100_counts:
                        b0100_counts[key] = 0
                    b0100_counts[key] += 1
                
                print("  标签分布:")
                for label, count in sorted(b0100_counts.items()):
                    percentage = (count / len(intent_b0100)) * 100
                    print(f"    标签 {label}: {count} 个样本 ({percentage:.2f}%)")
                
                # 分析intent_B0200
                print(f"\nIntent_B0200标签分析:")
                unique_b0200 = np.unique(intent_b0200)
                print(f"  唯一标签: {unique_b0200}")
                print(f"  标签种类数量: {len(unique_b0200)}")
                
                # 统计intent_B0200标签分布
                b0200_counts = {}
                for label in intent_b0200:
                    # 转换为Python原生类型
                    key = int(label) if isinstance(label, (np.integer, np.floating)) else label
                    if key not in b0200_counts:
                        b0200_counts[key] = 0
                    b0200_counts[key] += 1
                
                print("  标签分布:")
                for label, count in sorted(b0200_counts.items()):
                    percentage = (count / len(intent_b0200)) * 100
                    print(f"    标签 {label}: {count} 个样本 ({percentage:.2f}%)")
                
                # 分析独热编码部分
                print(f"\n6位独热编码分析:")
                print(f"  独热编码维度: {one_hot_encoding.shape[1]}")
                
                # 将独热编码转换为类别索引
                one_hot_classes = []
                for i, encoding in enumerate(one_hot_encoding):
                    class_idx = one_hot_to_class_index(encoding)
                    one_hot_classes.append(class_idx)
                    # 打印前几个样本的独热编码转换结果
                    if i < 5:
                        print(f"  样本 {i} 的独热编码: {encoding}")
                        print(f"  样本 {i} 的类别索引: {class_idx}")
                
                # 统计独热编码转换后的类别分布
                unique_one_hot = np.unique(one_hot_classes)
                print(f"  唯一类别索引: {unique_one_hot}")
                print(f"  类别数量: {len(unique_one_hot)}")
                
                one_hot_counts = {}
                for label in one_hot_classes:
                    # 转换为Python原生类型
                    key = int(label) if isinstance(label, (np.integer, np.floating)) else label
                    if key not in one_hot_counts:
                        one_hot_counts[key] = 0
                    one_hot_counts[key] += 1
                
                print("  类别分布:")
                for label, count in sorted(one_hot_counts.items()):
                    percentage = (count / len(one_hot_classes)) * 100
                    print(f"    类别 {label}: {count} 个样本 ({percentage:.2f}%)")
                
                # 分析所有11个标签维度的整体情况
                print(f"\n完整标签分析 (所有 {y_data.shape[1]} 个维度):")
                
                # 计算每个样本窗口的众数标签（取众数作为标签）
                all_mode_labels = []
                for i, window_labels in enumerate(y_data):
                    # 对于多维标签，我们对每个维度分别计算众数
                    # 这里简化处理，取intent_B0100的众数作为代表
                    mode_label = get_mode_with_last_occurrence([window_labels[0]])
                    all_mode_labels.append(mode_label)
                    # 打印前几个窗口的结果
                    if i < 5:
                        print(f"  窗口 {i} 的标签: {window_labels}")
                        print(f"  窗口 {i} 的intent_B0100标签: {window_labels[0]}")
                        print(f"  窗口 {i} 的intent_B0200标签: {window_labels[1]}")
                        print(f"  窗口 {i} 的独热编码部分: {window_labels[2:8] if y_data.shape[1] >= 8 else window_labels[2:]}")
                
                # 保存标签分布信息到文件
                # 确保所有numpy类型都转换为Python原生类型
                distribution_info = {
                    'intent_B0100': {
                        'unique_labels': [int(label) if isinstance(label, (np.integer, np.floating)) else label for label in unique_b0100.tolist()],
                        'label_counts': b0100_counts,
                        'total_samples': int(len(intent_b0100))
                    },
                    'intent_B0200': {
                        'unique_labels': [int(label) if isinstance(label, (np.integer, np.floating)) else label for label in unique_b0200.tolist()],
                        'label_counts': b0200_counts,
                        'total_samples': int(len(intent_b0200))
                    },
                    'one_hot_encoding': {
                        'unique_classes': [int(label) if isinstance(label, (np.integer, np.floating)) else label for label in unique_one_hot.tolist()],
                        'class_counts': one_hot_counts,
                        'total_samples': int(len(one_hot_classes)),
                        'encoding_dimension': int(one_hot_encoding.shape[1])
                    },
                    'total_samples': int(y_data.shape[0])
                }
                
                with open('dataset_label_distribution.json', 'w', encoding='utf-8') as f:
                    json.dump(distribution_info, f, ensure_ascii=False, indent=2)
                
                print("\n标签分布信息已保存到 dataset_label_distribution.json")
            else:
                # 处理其他格式的标签数据
                print(f"标签数据维度为 {y_data.shape}，不满足预期的[samples, 11]格式")
                
                # 按照原来的逻辑处理
                if len(y_data.shape) == 1:
                    # 一维数组 [samples,]
                    print("标签数据是一维数组格式")
                    all_labels = y_data.tolist()
                elif len(y_data.shape) == 2:
                    if y_data.shape[1] == 1:
                        # 二维数组但第二维只有1个元素 [samples, 1]
                        print("标签数据是二维数组格式，第二维长度为1")
                        all_labels = y_data.flatten().tolist()
                    else:
                        # 检查是否为独热编码格式（每行只有一个1，其余为0）
                        is_one_hot = True
                        for i in range(min(10, len(y_data))):  # 检查前10个样本
                            row_sum = np.sum(y_data[i])
                            ones_count = np.sum(y_data[i] == 1.0)
                            if row_sum != 1.0 or ones_count != 1:
                                is_one_hot = False
                                break
                        
                        if is_one_hot:
                            print(f"标签数据是独热编码格式，共有 {y_data.shape[0]} 个样本，每个标签维度为 {y_data.shape[1]}")
                            
                            all_labels = []
                            for i, one_hot_label in enumerate(y_data):
                                # 将独热编码转换为类别索引
                                class_index = one_hot_to_class_index(one_hot_label)
                                all_labels.append(class_index)
                        else:
                            # 假设是[samples, time_steps]格式，需要计算每个样本窗口的众数
                            print(f"标签数据是二维数组格式，假设为[samples, time_steps]")
                            
                            all_labels = []
                            for i, window_labels in enumerate(y_data):
                                # 计算每个窗口的众数标签
                                mode_label = get_mode_with_last_occurrence(window_labels)
                                if mode_label is not None:
                                    all_labels.append(mode_label)
                else:
                    print(f"警告: 标签数据维度为 {len(y_data.shape)}，可能不是预期格式")
                    all_labels = y_data.flatten().tolist()
            
            # 分析样本数据
            if 'samples' in data:
                samples = data['samples']
                print(f"\n样本数据形状: {samples.shape}")
                print(f"样本数量: {samples.shape[0]}")
                print(f"每个样本的特征维度: {samples.shape[1:]}")
        else:
            print("错误: 数据集中未找到 'labels' 键")
            
    except FileNotFoundError:
        print(f"错误: 找不到数据集文件 {data_path}")
    except Exception as e:
        import traceback
        print(f"分析数据集时出错: {str(e)}")
        print("错误详情:")
        traceback.print_exc()

def main():
    print("开始分析数据集标签分布...")
    load_and_analyze_dataset()
    print("\n数据集分析完成！")

if __name__ == "__main__":
    main()