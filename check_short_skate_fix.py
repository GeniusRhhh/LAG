#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from collections import Counter

def check_short_skate_fix():
    """检查Short Skate修复效果"""
    
    # 读取最新的CSV文件
    csv_file = 'scripts/drag_shoot_2v2/high_low_attack_results/high_low_attack_trajectory_20250829_222739.csv'
    
    try:
        df = pd.read_csv(csv_file)
        print("=== Short Skate修复验证结果 ===")
        print()
        
        # 统计所有动作标注
        action_counts = Counter(df['Action_Type'].dropna())
        print("修复后的动作标注统计:")
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  '{action}': {count}次")

        print()

        # 检查Short Skate相关标注
        short_skate_actions = [action for action in action_counts.keys() if 'Short skate' in action]
        if short_skate_actions:
            print("✅ Short Skate标注修复成功:")
            for action in short_skate_actions:
                print(f"  '{action}': {action_counts[action]}次")
        else:
            print("❌ Short Skate标注未找到")

        print()

        # 检查方向标注多样性
        direction_counts = Counter(df['Direction'].dropna())
        print("方向标注多样性:")
        for direction, count in sorted(direction_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  '{direction}': {count}次")

        print()

        # 按智能体分析Short Skate
        print("按智能体分析Short Skate标注:")
        for agent in ['A0100', 'A0200', 'B0100', 'B0200']:
            agent_data = df[df['Agent_ID'] == agent]
            agent_short_skate = agent_data[agent_data['Action_Type'].str.contains('Short skate', na=False)]
            if len(agent_short_skate) > 0:
                print(f"  {agent}: {len(agent_short_skate)}次Short Skate标注")
                skate_actions = Counter(agent_short_skate['Action_Type'])
                for action, count in skate_actions.items():
                    print(f"    - '{action}': {count}次")
            else:
                print(f"  {agent}: 无Short Skate标注")
        
        print()
        print("=== 修复验证完成 ===")
        
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    check_short_skate_fix()
