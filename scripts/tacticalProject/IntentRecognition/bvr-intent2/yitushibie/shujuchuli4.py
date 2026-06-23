import numpy as np
import json

# 查看42维特征对应的索引名称
def analyze_features():
    """分析并显示42维特征对应的索引名称"""
    print("分析yitushibie.py中使用的42维特征索引")
    print("=" * 60)
    
    # 从yitushibie.py中提取的特征索引
    # 第8-27位（Python索引从0开始，所以是7-26）
    feature_indices_1 = list(range(7, 27))
    # 第44-65位（Python索引从0开始，所以是43-64）
    feature_indices_2 = list(range(43, 65))
    
    # 合并所有特征索引
    all_feature_indices = feature_indices_1 + feature_indices_2
    
    print(f"总特征维度: {len(all_feature_indices)}")
    print("\n特征索引详情:")
    print("-" * 60)
    print(f"{'序号':<8}{'原始数据索引':<15}{'特征组':<15}{'特征描述':<20}")
    print("-" * 60)
    
    # 为特征提供描述性名称
    # 注意：由于没有具体的特征名称信息，这里根据索引位置提供通用描述
    # 实际应用中，应该根据数据的具体含义修改这些描述
    
    # 定义特征描述
    feature_descriptions = []
    
    # 第8-27位特征的描述
    for i, idx in enumerate(feature_indices_1):
        # 假设这些是与飞机状态相关的特征
        feature_num = i + 8  # 转换回原始编号（1-based）
        desc = f"飞机状态特征_{feature_num}"
        feature_descriptions.append((i, idx, "组1 (8-27)", desc))
    
    # 第44-65位特征的描述
    offset = len(feature_indices_1)
    for i, idx in enumerate(feature_indices_2):
        # 假设这些是与动作相关的特征
        feature_num = i + 44  # 转换回原始编号（1-based）
        desc = f"动作相关特征_{feature_num}"
        feature_descriptions.append((i + offset, idx, "组2 (44-65)", desc))
    
    # 输出所有特征
    for idx, orig_idx, group, desc in feature_descriptions:
        print(f"{idx:<8}{orig_idx:<15}{group:<15}{desc:<20}")
    
    # 特别标记v3和v4的位置（根据之前的任务描述，它们是索引17和18）
    print("\n特别标记:")
    print("-" * 60)
    print("注意：根据之前的任务，v3和v4对应原始数据中的索引17和18")
    print("在提取的特征中：")
    
    # 检查v3和v4是否在特征索引中
    v3_idx = 17
    v4_idx = 18
    
    if v3_idx in all_feature_indices:
        pos = all_feature_indices.index(v3_idx)
        print(f"v3 (原始索引{17}) 对应提取特征中的位置: {pos}")
    else:
        print(f"v3 (原始索引{17}) 不在提取的特征中")
    
    if v4_idx in all_feature_indices:
        pos = all_feature_indices.index(v4_idx)
        print(f"v4 (原始索引{18}) 对应提取特征中的位置: {pos}")
    else:
        print(f"v4 (原始索引{18}) 不在提取的特征中")
    
    # 尝试加载数据集并显示更多信息（如果可能）
    try:
        dataset_path = r"c:\Users\86188\Desktop\yitushibie\chuli3\带动作独热编码的数据集\enhanced_time_series_data_with_actions.npz"
        if __name__ == "__main__":
            print("\n尝试加载数据集以获取更多信息...")
            data = np.load(dataset_path)
            X_data = data['data']
            print(f"数据集加载成功")
            print(f"数据形状: {X_data.shape}")
            print(f"原始特征总数: {X_data.shape[2]}")
            
            # 显示提取特征的部分统计信息
            print("\n提取特征的统计信息示例:")
            print("-" * 60)
            sample_idx = 0  # 第一个样本
            time_idx = 0    # 第一个时间步
            
            # 显示前5个特征和后5个特征的取值
            print(f"样本 {sample_idx} 在时间步 {time_idx} 的特征值:")
            for i, orig_idx in enumerate(all_feature_indices[:5]):
                print(f"特征 {i} (原始索引 {orig_idx}): {X_data[sample_idx, time_idx, orig_idx]:.4f}")
            
            print("...")
            
            for i, orig_idx in enumerate(all_feature_indices[-5:]):
                actual_i = len(all_feature_indices) - 5 + i
                print(f"特征 {actual_i} (原始索引 {orig_idx}): {X_data[sample_idx, time_idx, orig_idx]:.4f}")
    except Exception as e:
        print(f"\n无法加载数据集或显示统计信息: {str(e)}")
    
    # 保存特征信息到JSON文件
    feature_info = {
        "total_features": len(all_feature_indices),
        "feature_groups": [
            {
                "name": "组1",
                "original_range": "8-27",
                "indices": feature_indices_1
            },
            {
                "name": "组2",
                "original_range": "44-65",
                "indices": feature_indices_2
            }
        ],
        "all_indices": all_feature_indices,
        "feature_details": [
            {"position": idx, "original_index": orig_idx, "group": group, "description": desc}
            for idx, orig_idx, group, desc in feature_descriptions
        ],
        "v3_info": {
            "original_index": v3_idx,
            "in_features": v3_idx in all_feature_indices,
            "position": all_feature_indices.index(v3_idx) if v3_idx in all_feature_indices else None
        },
        "v4_info": {
            "original_index": v4_idx,
            "in_features": v4_idx in all_feature_indices,
            "position": all_feature_indices.index(v4_idx) if v4_idx in all_feature_indices else None
        }
    }
    
    # 保存到JSON文件
    json_path = "feature_indices_info.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n特征信息已保存到 {json_path}")

if __name__ == "__main__":
    analyze_features()