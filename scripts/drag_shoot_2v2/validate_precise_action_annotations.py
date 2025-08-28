#!/usr/bin/env python3
"""
精确动作标注验证脚本
验证基于实际战术代码逻辑的动作标注系统
"""

import pandas as pd
import os
import numpy as np

def validate_precise_action_annotations():
    """验证精确动作标注结果"""
    
    # 查找最新的轨迹数据文件
    results_dir = "scripts/drag_shoot_2v2/air_combat_results"
    csv_files = [f for f in os.listdir(results_dir) if f.startswith('drag_shoot_trajectory_') and f.endswith('.csv')]
    
    if not csv_files:
        print("未找到轨迹数据文件")
        return
    
    # 选择最新的文件
    latest_file = sorted(csv_files)[-1]
    csv_path = os.path.join(results_dir, latest_file)
    
    print("=" * 80)
    print("🔍 精确动作标注验证报告")
    print("=" * 80)
    print(f"分析文件: {latest_file}")
    print()
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    print(f"📊 基础统计信息:")
    print(f"   数据总行数: {len(df)}")
    print(f"   时间范围: {df['Time_s'].min():.1f}s - {df['Time_s'].max():.1f}s")
    print(f"   飞机数量: {df['Agent_ID'].nunique()}")
    print(f"   飞机ID: {list(df['Agent_ID'].unique())}")
    print()
    
    print("📈 动作类型统计:")
    action_counts = df['Action_Type'].value_counts()
    for action, count in action_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {action}: {count} ({percentage:.1f}%)")
    print()
    
    print("🎯 各飞机动作分布分析:")
    for agent in sorted(df['Agent_ID'].unique()):
        agent_data = df[df['Agent_ID'] == agent]
        print(f"\n   {agent} ({len(agent_data)} 个数据点):")
        agent_actions = agent_data['Action_Type'].value_counts()
        for action, count in agent_actions.items():
            percentage = (count / len(agent_data)) * 100
            print(f"     {action}: {count} ({percentage:.1f}%)")
    
    print("\n" + "=" * 80)
    print("🔬 精确度验证分析")
    print("=" * 80)
    
    # 验证友方僚机A0200的crank机动
    print("\n1️⃣ 友方僚机A0200 Crank机动验证:")
    a0200_data = df[df['Agent_ID'] == 'A0200'].copy()
    if len(a0200_data) > 0:
        # 分析航向变化
        a0200_data['heading_change'] = a0200_data['Heading_deg'].diff().abs()
        crank_data = a0200_data[a0200_data['Action_Type'].str.contains('Crank|crank', na=False)]
        
        if len(crank_data) > 0:
            print(f"   ✅ 识别到Crank机动: {len(crank_data)} 个数据点")
            print(f"   📍 Crank时间范围: {crank_data['Time_s'].min():.1f}s - {crank_data['Time_s'].max():.1f}s")
            print(f"   🧭 Crank期间航向范围: {crank_data['Heading_deg'].min():.1f}° - {crank_data['Heading_deg'].max():.1f}°")
            
            # 检查是否有68°航向（右侧crank）
            heading_68_data = crank_data[(crank_data['Heading_deg'] >= 60) & (crank_data['Heading_deg'] <= 76)]
            if len(heading_68_data) > 0:
                print(f"   🎯 右侧crank (60°-76°): {len(heading_68_data)} 个数据点 ✅")
            else:
                print(f"   ❌ 未发现右侧crank (60°-76°航向)")
                
            # 检查左转crank（回到0°）
            heading_0_data = crank_data[((crank_data['Heading_deg'] >= 350) | (crank_data['Heading_deg'] <= 10))]
            if len(heading_0_data) > 0:
                print(f"   🎯 左转crank (350°-10°): {len(heading_0_data)} 个数据点 ✅")
            else:
                print(f"   ❌ 未发现左转crank (350°-10°航向)")
        else:
            print(f"   ❌ 未识别到A0200的Crank机动")
    
    # 验证敌方飞机动作识别
    print("\n2️⃣ 敌方飞机动作验证:")
    enemy_agents = [agent for agent in df['Agent_ID'].unique() if agent.startswith('B')]
    
    for enemy_id in enemy_agents:
        enemy_data = df[df['Agent_ID'] == enemy_id]
        if len(enemy_data) > 0:
            print(f"\n   {enemy_id}:")
            enemy_actions = enemy_data['Action_Type'].value_counts()
            
            # 检查是否只有平飞
            if len(enemy_actions) == 1 and '平飞' in enemy_actions.index:
                print(f"     ❌ 仅识别到平飞动作，可能存在识别缺失")
            else:
                print(f"     ✅ 识别到多种动作类型:")
                for action, count in enemy_actions.items():
                    percentage = (count / len(enemy_data)) * 100
                    print(f"       {action}: {count} ({percentage:.1f}%)")
            
            # 检查Short Skate机动
            skate_data = enemy_data[enemy_data['Action_Type'].str.contains('Short skate|战术crank', na=False)]
            if len(skate_data) > 0:
                print(f"     🎯 Short Skate/战术机动: {len(skate_data)} 个数据点 ✅")
            else:
                print(f"     ⚠️  未识别到Short Skate机动")
            
            # 检查航向分布
            south_heading = enemy_data[(enemy_data['Heading_deg'] >= 170) & (enemy_data['Heading_deg'] <= 190)]
            north_heading = enemy_data[((enemy_data['Heading_deg'] >= 350) | (enemy_data['Heading_deg'] <= 10))]
            
            print(f"     📊 航向分布:")
            print(f"       南向(170°-190°): {len(south_heading)} 个数据点 ({len(south_heading)/len(enemy_data)*100:.1f}%)")
            print(f"       北向(350°-10°): {len(north_heading)} 个数据点 ({len(north_heading)/len(enemy_data)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("📋 验证结论")
    print("=" * 80)
    
    # 计算动作多样性
    action_diversity = len(action_counts)
    print(f"✨ 动作类型多样性: {action_diversity}/11 种基本动作")
    
    # 检查中文编码
    has_chinese = any('平飞' in action or 'Crank' in action or '战术' in action for action in action_counts.index)
    if has_chinese:
        print("✅ 中文编码显示正常")
    else:
        print("❌ 中文编码存在问题")
    
    # 检查动作标注覆盖率
    coverage = (len(df) - df['Action_Type'].isna().sum()) / len(df) * 100
    print(f"📊 动作标注覆盖率: {coverage:.1f}%")
    
    # 检查时间精度
    time_intervals = df['Time_s'].diff().dropna().unique()
    time_precision = min(time_intervals[time_intervals > 0])
    print(f"⏱️  时间精度: {time_precision:.1f}秒间隔")
    
    # 综合评估
    print(f"\n🎯 综合评估:")
    
    # 友方动作识别评估
    friendly_crank_found = len(df[(df['Agent_ID'] == 'A0200') & 
                                 (df['Action_Type'].str.contains('Crank|crank', na=False))]) > 0
    if friendly_crank_found:
        print("   ✅ 友方僚机Crank机动识别: 通过")
    else:
        print("   ❌ 友方僚机Crank机动识别: 失败")
    
    # 敌方动作识别评估
    enemy_diversity = 0
    for enemy_id in enemy_agents:
        enemy_actions = df[df['Agent_ID'] == enemy_id]['Action_Type'].nunique()
        if enemy_actions > 1:
            enemy_diversity += 1
    
    if enemy_diversity > 0:
        print(f"   ✅ 敌方动作多样性识别: 通过 ({enemy_diversity}/{len(enemy_agents)} 个敌机)")
    else:
        print(f"   ❌ 敌方动作多样性识别: 失败")
    
    # 最终评分
    score = 0
    if coverage == 100.0:
        score += 25
    if action_diversity >= 5:
        score += 25
    if has_chinese:
        score += 25
    if friendly_crank_found and enemy_diversity > 0:
        score += 25
    
    print(f"\n🏆 最终评分: {score}/100")
    
    if score >= 80:
        print("🎉 精确动作标注系统验证通过！")
    elif score >= 60:
        print("⚠️  动作标注系统基本可用，但需要进一步优化")
    else:
        print("❌ 动作标注系统需要重大改进")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    validate_precise_action_annotations()
