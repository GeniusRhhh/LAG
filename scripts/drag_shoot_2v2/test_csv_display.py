#!/usr/bin/env python3
"""
测试CSV文件显示效果
验证不同查看方式下的中文显示情况
"""

import pandas as pd
import os


def test_csv_display():
    """测试CSV文件的显示效果"""
    print("=" * 60)
    print("CSV文件中文显示测试")
    print("=" * 60)
    
    # 测试文件列表
    test_files = [
        'scripts/drag_shoot_2v2/pincer_attack_results/pincer_attack_trajectory_20250828_140149.csv',
        'scripts/drag_shoot_2v2/front_back_attack_results/front_back_attack_trajectory_20250828_135503.csv',
        'scripts/drag_shoot_2v2/high_low_attack_results/high_low_attack_trajectory_20250828_150015.csv'
    ]
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"⚠️ 文件不存在: {file_path}")
            continue
            
        print(f"\n=== 测试文件: {os.path.basename(file_path)} ===")
        
        # 测试不同编码读取
        for encoding in ['utf-8', 'utf-8-sig', 'gbk']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"\n✅ {encoding} 编码读取成功")
                
                if 'Action_Type' in df.columns:
                    # 显示Action_Type的唯一值
                    action_types = df['Action_Type'].unique()[:5]
                    print(f"Action_Type样本 ({encoding}):")
                    for i, action in enumerate(action_types):
                        print(f"  {i+1}. '{action}' (类型: {type(action).__name__})")
                    
                    # 显示Direction的唯一值
                    if 'Direction' in df.columns:
                        directions = df['Direction'].unique()[:5]
                        print(f"Direction样本 ({encoding}):")
                        for i, direction in enumerate(directions):
                            print(f"  {i+1}. '{direction}' (类型: {type(direction).__name__})")
                
                break  # 成功读取后跳出循环
                
            except Exception as e:
                print(f"❌ {encoding} 编码读取失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    
    # 额外测试：创建一个包含中文的测试CSV
    print("\n=== 创建测试CSV文件 ===")
    test_data = {
        'Agent_ID': ['A0100', 'A0200', 'B0100'],
        'Action_Type': ['战术crank', '平飞', 'Crank'],
        'Direction': ['左转', '无', '右转'],
        'Time_s': [10.0, 20.0, 30.0]
    }
    
    test_df = pd.DataFrame(test_data)
    test_file = 'scripts/drag_shoot_2v2/test_chinese_display.csv'
    
    # 用不同编码保存测试文件
    encodings_to_test = ['utf-8', 'utf-8-sig', 'gbk']
    
    for encoding in encodings_to_test:
        try:
            test_file_encoded = f'scripts/drag_shoot_2v2/test_chinese_{encoding.replace("-", "_")}.csv'
            test_df.to_csv(test_file_encoded, index=False, encoding=encoding)
            print(f"✅ 创建测试文件: {os.path.basename(test_file_encoded)} ({encoding})")
            
            # 立即读取验证
            df_read = pd.read_csv(test_file_encoded, encoding=encoding)
            print(f"   验证读取: Action_Type[0] = '{df_read['Action_Type'].iloc[0]}'")
            
        except Exception as e:
            print(f"❌ 创建 {encoding} 测试文件失败: {e}")


if __name__ == "__main__":
    test_csv_display()
