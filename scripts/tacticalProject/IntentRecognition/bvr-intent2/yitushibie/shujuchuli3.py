import numpy as np
import os
import pandas as pd
import random

# 数据集路径
dataset_path = r'c:\Users\86188\Desktop\yitushibie\chuli3\带动作独热编码的数据集\enhanced_time_series_data_with_actions.npz'

print("正在查询chuli3中数据集的索引和形状信息...")
print(f"数据集路径: {dataset_path}")
print("-" * 50)

# 检查文件是否存在
if not os.path.exists(dataset_path):
    print(f"错误: 找不到数据集文件 '{dataset_path}'")
    print("请确认数据集路径是否正确")
    exit(1)

try:
    # 加载数据集
    data = np.load(dataset_path)
    
    # 显示所有可用的索引
    print("数据集包含的索引:")
    for key in data.keys():
        print(f"  - {key}")
    
    print("\n数据集各部分的形状信息:")
    print("-" * 50)
    
    # 显示每个索引对应的数据形状
    for key in data.keys():
        array = data[key]
        print(f"索引 '{key}' 的信息:")
        print(f"  形状: {array.shape}")
        print(f"  数据类型: {array.dtype}")
        print(f"  维度: {array.ndim}")
        
        # 如果是数组，显示一些基本统计信息
        if isinstance(array, np.ndarray):
            print(f"  元素数量: {array.size}")
            
            # 对于二维或更高维数组，显示前几行
            if array.ndim >= 2:
                print("\n  前3行数据的前10个特征示例:")
                for i in range(min(3, array.shape[0])):
                    # 只显示前10个特征
                    features_to_show = array[i, :min(10, array.shape[1])] if array.ndim >= 2 else array[i]
                    print(f"    行 {i}: {features_to_show}")
        
        print("-" * 50)
    
    # 特别处理feature_names（如果存在）
    if 'feature_names' in data:
        feature_names = data['feature_names']
        print("\n特征名称列表:")
        print(f"共有 {len(feature_names)} 个特征")
        print("\n前20个特征名称:")
        for i, name in enumerate(feature_names[:20]):
            print(f"  {i + 1}. {name}")
        
        if len(feature_names) > 20:
            print(f"\n后20个特征名称:")
            for i, name in enumerate(feature_names[-20:], len(feature_names) - 19):
                print(f"  {i}. {name}")
    
    # 特别处理data（如果存在）
    if 'data' in data:
        data_array = data['data']
        print("\n数据数组 'data' 的详细信息:")
        print(f"  形状: {data_array.shape}")
        print(f"  行数: {data_array.shape[0]}")
        print(f"  列数/特征数: {data_array.shape[1] if data_array.ndim >= 2 else 'N/A'}")
        
        # 计算并显示数据的基本统计信息
        if data_array.ndim >= 2:
            print("\n  数据统计信息:")
            print(f"    最小值: {np.min(data_array):.6f}")
            print(f"    最大值: {np.max(data_array):.6f}")
            print(f"    平均值: {np.mean(data_array):.6f}")
            print(f"    标准差: {np.std(data_array):.6f}")
    
    # 添加随机选择窗口并导出为Excel的功能
    print("\n正在添加随机选择窗口并导出为Excel的功能...")
    print("-" * 50)
    
    if 'data' in data and 'feature_names' in data:
        # 随机选择一个窗口索引
        random_window_idx = random.randint(0, data['data'].shape[0] - 1)
        print(f"随机选择的窗口索引: {random_window_idx}")
        
        # 获取该窗口的数据
        window_data = data['data'][random_window_idx]
        feature_names = data['feature_names']
        
        print(f"窗口形状: {window_data.shape}")
        print(f"包含 {window_data.shape[0]} 个时间步")
        
        # 创建DataFrame
        df = pd.DataFrame(window_data, columns=feature_names)
        
        # 创建输出目录
        output_dir = r'c:\Users\86188\Desktop\yitushibie\chuli3\窗口样本'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 生成Excel文件路径
        excel_file = os.path.join(output_dir, f'window_sample_{random_window_idx}.xlsx')
        
        # 导出到Excel
        try:
            df.to_excel(excel_file, index_label='时间步')
            print(f"\n成功导出窗口样本到Excel文件:")
            print(f"文件路径: {excel_file}")
            print(f"文件包含 {len(df)} 行数据和 {len(df.columns)} 列特征")
            
            # 显示导出的数据摘要
            print("\n导出数据摘要:")
            print(f"前5个特征: {list(df.columns[:5])}")
            print(f"最后5个特征: {list(df.columns[-5:])}")
            
        except Exception as e:
            print(f"\n导出Excel文件时出错: {e}")
            print("请确保已安装openpyxl库: pip install openpyxl")
    else:
        print("\n无法执行窗口导出功能，数据集中缺少必要的索引")
    
    print("\n数据集查询完成!")
    
finally:
    # 确保关闭文件
    if 'data' in locals():
        data.close()