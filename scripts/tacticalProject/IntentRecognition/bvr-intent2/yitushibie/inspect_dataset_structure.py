import numpy as np
import os

# 简单的数据集结构检查脚本
def inspect_dataset_structure(dataset_path="processed_data.npz"):
    """
    详细检查数据集的结构
    """
    print(f"正在检查数据集: {dataset_path}")
    
    # 检查文件是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: 找不到数据集文件 {dataset_path}")
        return
    
    # 加载数据集
    try:
        data = np.load(dataset_path)
        print(f"数据集加载成功，包含以下键: {list(data.keys())}")
        
        # 检查每个键的数据
        for key in data.keys():
            value = data[key]
            print(f"\n键 '{key}' 的信息:")
            print(f"  形状: {value.shape}")
            print(f"  维度: {value.ndim}")
            print(f"  数据类型: {value.dtype}")
            
            # 显示前几个样本的详细信息
            print(f"  前3个样本的形状:")
            if value.ndim >= 1:
                for i in range(min(3, len(value))):
                    if value.ndim == 1:
                        # 一维数据
                        print(f"    样本 {i}: 长度 {len(value[i]) if hasattr(value[i], '__len__') else '标量'}, 值: {value[i] if hasattr(value[i], '__len__') else value[i]}")
                    elif value.ndim == 2:
                        # 二维数据
                        print(f"    样本 {i}: 形状 {value[i].shape}")
                        if value[i].size <= 100:  # 避免打印过大的数据
                            print(f"      内容: {value[i]}")
                    else:
                        # 更高维数据
                        print(f"    样本 {i}: 形状 {value[i].shape}")
            
        # 特别详细检查labels
        if 'labels' in data:
            labels = data['labels']
            print(f"\n===== 标签数据详细检查 =====")
            print(f"标签数据类型: {labels.dtype}")
            print(f"标签总长度: {len(labels)}")
            
            # 检查前5个标签样本
            for i in range(min(5, len(labels))):
                label = labels[i]
                print(f"\n标签样本 {i}:")
                print(f"  类型: {type(label)}")
                print(f"  形状: {label.shape if hasattr(label, 'shape') else 'N/A'}")
                print(f"  大小: {label.size if hasattr(label, 'size') else 'N/A'}")
                
                # 显示一些值
                if hasattr(label, '__len__'):
                    print(f"  长度: {len(label)}")
                    # 尝试显示一些元素
                    max_display = min(20, len(label))
                    print(f"  前{max_display}个元素: {label[:max_display]}")
                else:
                    print(f"  值: {label}")
            
            # 尝试检查标签的实际内容
            print(f"\n===== 标签内容检查 =====")
            first_label = labels[0]
            if isinstance(first_label, np.ndarray):
                print(f"第一个标签是numpy数组，形状: {first_label.shape}")
                
                # 检查是否是时间序列数据
                if first_label.ndim == 2:
                    print(f"这可能是时间序列数据，每个窗口有 {first_label.shape[0]} 个时间步，每个时间步有 {first_label.shape[1]} 个特征")
                    # 显示第一个时间步的标签
                    print(f"第一个时间步的标签: {first_label[0]}")
                    # 检查每个时间步的标签是否相同
                    if first_label.shape[0] > 1:
                        all_same = np.all(first_label == first_label[0], axis=0)
                        print(f"所有时间步标签都相同: {all_same}")
                        if not np.all(all_same):
                            # 找出不同的位置
                            different_positions = np.where(all_same == False)[0]
                            print(f"不同的位置: {different_positions}")
                else:
                    print(f"标签维度为 {first_label.ndim}")
            
    except Exception as e:
        print(f"检查数据集时出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_dataset_structure()