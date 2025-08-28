#!/usr/bin/env python3
"""
动作标注验证脚本
验证修正后的动作标注系统的准确性
"""

import pandas as pd
import os

def validate_action_annotations():
    """验证动作标注结果"""
    
    # 查找最新的轨迹数据文件
    results_dir = "scripts/drag_shoot_2v2/air_combat_results"
    csv_files = [f for f in os.listdir(results_dir) if f.startswith('drag_shoot_trajectory_') and f.endswith('.csv')]
    
    if not csv_files:
        print("未找到轨迹数据文件")
        return
    
    # 选择最新的文件
    latest_file = sorted(csv_files)[-1]
    csv_path = os.path.join(results_dir, latest_file)
    
    print("=== 动作标注验证报告 ===")
    print(f"分析文件: {latest_file}")
    print()
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    print(f"数据总行数: {len(df)}")
    print(f"时间范围: {df['Time_s'].min():.1f}s - {df['Time_s'].max():.1f}s")
    print(f"飞机数量: {df['Agent_ID'].nunique()}")
    print(f"飞机ID: {list(df['Agent_ID'].unique())}")
    print()
    
    print("=== 动作类型统计 ===")
    action_counts = df['Action_Type'].value_counts()
    for action, count in action_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{action}: {count} ({percentage:.1f}%)")
    print()
    
    print("=== 各飞机动作分布 ===")
    for agent in df['Agent_ID'].unique():
        agent_data = df[df['Agent_ID'] == agent]
        print(f"{agent}:")
        agent_actions = agent_data['Action_Type'].value_counts()
        for action, count in agent_actions.items():
            percentage = (count / len(agent_data)) * 100
            print(f"  {action}: {count} ({percentage:.1f}%)")
        print()
    
    print("=== 动作时间序列分析 ===")
    # 找出每种动作的首次出现时间
    first_occurrence = {}
    for action in df['Action_Type'].unique():
        first_time = df[df['Action_Type'] == action]['Time_s'].min()
        first_occurrence[action] = first_time
    
    print("各动作首次出现时间:")
    for action, time in sorted(first_occurrence.items(), key=lambda x: x[1]):
        print(f"  {action}: {time:.1f}s")
    print()
    
    print("=== 关键发现 ===")
    
    # 检查Crank机动识别情况
    crank_data = df[df['Action_Type'].str.contains('Crank|crank')]
    if len(crank_data) > 0:
        print(f"✅ Crank机动识别成功: {len(crank_data)} 个数据点")
        crank_agents = crank_data['Agent_ID'].value_counts()
        for agent, count in crank_agents.items():
            print(f"   {agent}: {count} 个Crank动作")
    else:
        print("❌ 未识别到Crank机动")
    
    # 检查Short Skate机动识别情况
    skate_data = df[df['Action_Type'].str.contains('Short skate')]
    if len(skate_data) > 0:
        print(f"✅ Short Skate机动识别成功: {len(skate_data)} 个数据点")
        skate_agents = skate_data['Agent_ID'].value_counts()
        for agent, count in skate_agents.items():
            print(f"   {agent}: {count} 个Short Skate动作")
    else:
        print("❌ 未识别到Short Skate机动")
    
    # 检查战术机动识别情况
    tactical_data = df[df['Action_Type'].str.contains('战术')]
    if len(tactical_data) > 0:
        print(f"✅ 战术机动识别成功: {len(tactical_data)} 个数据点")
        tactical_agents = tactical_data['Agent_ID'].value_counts()
        for agent, count in tactical_agents.items():
            print(f"   {agent}: {count} 个战术动作")
    else:
        print("❌ 未识别到战术机动")
    
    print()
    print("=== 验证结论 ===")
    
    # 计算动作多样性
    action_diversity = len(action_counts)
    print(f"动作类型多样性: {action_diversity}/11 种基本动作")
    
    # 检查中文编码
    has_chinese = any('平飞' in action or 'Crank' in action or '战术' in action for action in action_counts.index)
    if has_chinese:
        print("✅ 中文编码显示正常")
    else:
        print("❌ 中文编码存在问题")
    
    # 检查动作标注覆盖率
    coverage = (len(df) - df['Action_Type'].isna().sum()) / len(df) * 100
    print(f"动作标注覆盖率: {coverage:.1f}%")
    
    if coverage == 100.0 and action_diversity >= 3 and has_chinese:
        print("🎉 动作标注系统验证通过！")
    else:
        print("⚠️  动作标注系统需要进一步优化")

if __name__ == "__main__":
    validate_action_annotations()
