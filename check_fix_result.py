#!/usr/bin/env python3
"""检查方向标注修复结果"""

import pandas as pd
from collections import Counter

def check_fix_result():
    """检查修复结果"""
    
    csv_file = 'scripts/drag_shoot_2v2/high_low_attack_results/high_low_attack_trajectory_20250829_222739.csv'
    
    try:
        df = pd.read_csv(csv_file)
        
        print('=== 方向标注修复验证结果 ===')
        print()
        
        # 统计所有动作标注
        action_counts = Counter(df['Action_Type'].dropna())
        print('修复后的动作标注统计:')
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            print(f'  {action}: {count}次')
        
        print()
        
        # 检查方向标注多样性
        direction_counts = Counter(df['Direction'].dropna())
        print('方向标注多样性:')
        for direction, count in sorted(direction_counts.items(), key=lambda x: x[1], reverse=True):
            print(f'  {direction}: {count}次')
        
        print()
        
        # 分析A0200的Short Skate方向标注
        a0200_short_skate = df[(df['Agent_ID'] == 'A0200') & (df['Action_Type'] == 'Short skate')]
        if len(a0200_short_skate) > 0:
            print('A0200 Short Skate方向标注:')
            a0200_directions = Counter(a0200_short_skate['Direction'])
            for direction, count in a0200_directions.items():
                print(f'  {direction}: {count}次')
            
            # 显示前几个数据点
            print()
            print('A0200 Short Skate前10个数据点:')
            for i, (idx, row) in enumerate(a0200_short_skate.head(10).iterrows()):
                print(f'  时间: {row["Time_s"]:6.1f}s, 航向: {row["Heading_deg"]:6.1f}°, 方向标注: {row["Direction"]}')
        else:
            print('未找到A0200的Short Skate数据')
        
        print()
        print('=== 修复验证完成 ===')
        
    except Exception as e:
        print(f'错误: {e}')

if __name__ == "__main__":
    check_fix_result()
