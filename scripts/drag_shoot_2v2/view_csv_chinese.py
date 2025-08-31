#!/usr/bin/env python3
"""
CSV中文查看器
确保正确显示中文内容，解决编码问题
"""

import pandas as pd
import os
import sys


def view_csv_with_chinese(file_path: str, max_rows: int = 20):
    """查看CSV文件，确保中文正确显示"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    print(f"📁 查看文件: {os.path.basename(file_path)}")
    print("=" * 80)
    
    # 尝试不同编码读取
    df = None
    encoding_used = None
    
    for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            encoding_used = encoding
            print(f"✅ 使用 {encoding} 编码成功读取")
            break
        except Exception as e:
            continue
    
    if df is None:
        print("❌ 无法读取文件")
        return
    
    print(f"📊 文件信息: {len(df)} 行, {len(df.columns)} 列")
    print(f"🔤 使用编码: {encoding_used}")
    
    # 显示列名
    print(f"\n📋 列名: {list(df.columns)}")
    
    # 如果有Action_Type和Direction列，特别显示
    if 'Action_Type' in df.columns:
        print(f"\n🎯 Action_Type 唯一值:")
        action_types = df['Action_Type'].value_counts()
        for action, count in action_types.head(10).items():
            print(f"   '{action}': {count}次")
    
    if 'Direction' in df.columns:
        print(f"\n🧭 Direction 唯一值:")
        directions = df['Direction'].value_counts()
        for direction, count in directions.head(10).items():
            print(f"   '{direction}': {count}次")
    
    # 显示前几行数据
    print(f"\n📄 前{min(max_rows, len(df))}行数据:")
    print("-" * 80)
    
    # 只显示关键列
    key_columns = ['Agent_ID', 'Time_s', 'Action_Type', 'Direction']
    display_columns = [col for col in key_columns if col in df.columns]
    
    if display_columns:
        display_df = df[display_columns].head(max_rows)
        for idx, row in display_df.iterrows():
            print(f"行{idx+1:3d}: ", end="")
            for col in display_columns:
                value = row[col]
                if pd.isna(value):
                    value = "NaN"
                print(f"{col}='{value}' ", end="")
            print()
    else:
        # 如果没有关键列，显示所有列
        for idx, row in df.head(max_rows).iterrows():
            print(f"行{idx+1}: {dict(row)}")
    
    print("=" * 80)


def main():
    """主函数"""
    print("🔍 CSV中文查看器")
    print("=" * 80)
    
    # 查看最新的轨迹文件
    latest_files = [
        ('钳形攻击', 'scripts/drag_shoot_2v2/pincer_attack_results/pincer_attack_trajectory_20250828_140149.csv'),
        ('前后攻击', 'scripts/drag_shoot_2v2/front_back_attack_results/front_back_attack_trajectory_20250828_135503.csv'),
        ('上下夹击', 'scripts/drag_shoot_2v2/high_low_attack_results/high_low_attack_trajectory_20250828_150015.csv'),
        ('拖曳射击', 'scripts/drag_shoot_2v2/air_combat_results/drag_shoot_trajectory_20250828_104722.csv')
    ]
    
    for tactic_name, file_path in latest_files:
        print(f"\n🎯 {tactic_name}战术")
        view_csv_with_chinese(file_path, max_rows=5)
        print()
    
    # 检查是否有用户指定的文件
    if len(sys.argv) > 1:
        user_file = sys.argv[1]
        print(f"\n👤 用户指定文件")
        view_csv_with_chinese(user_file, max_rows=10)


if __name__ == "__main__":
    main()
