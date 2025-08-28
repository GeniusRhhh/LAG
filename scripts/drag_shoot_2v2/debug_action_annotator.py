#!/usr/bin/env python3
"""
调试动作标注器
检查动作识别逻辑是否正常工作
"""

import pandas as pd
import os
import sys
sys.path.append('.')

from scripts.drag_shoot_2v2.action_annotator import ActionAnnotator

def debug_action_annotator():
    """调试动作标注器的核心问题"""
    
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
    print("🐛 动作标注器调试分析")
    print("=" * 80)
    print(f"分析文件: {latest_file}")
    print()
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 检查数据基本信息
    print(f"📊 数据基本信息:")
    print(f"   总数据点: {len(df)}")
    print(f"   时间范围: {df['Time_s'].min():.1f}s - {df['Time_s'].max():.1f}s")
    print(f"   飞机数量: {df['Agent_ID'].nunique()}")
    print(f"   飞机ID: {list(df['Agent_ID'].unique())}")
    print()
    
    # 检查动作标注分布
    print("🎯 动作标注分布:")
    action_counts = df['Action_Type'].value_counts()
    for action, count in action_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {action}: {count} ({percentage:.1f}%)")
    print()
    
    # 检查每个飞机的动作分布
    print("✈️ 各飞机动作分布:")
    for agent_id in df['Agent_ID'].unique():
        agent_data = df[df['Agent_ID'] == agent_id]
        agent_actions = agent_data['Action_Type'].value_counts()
        print(f"   {agent_id}:")
        for action, count in agent_actions.items():
            percentage = (count / len(agent_data)) * 100
            print(f"     {action}: {count} ({percentage:.1f}%)")
        print()
    
    # 重点分析A0200的航向变化
    print("🔍 A0200航向分析:")
    a0200_data = df[df['Agent_ID'] == 'A0200'].copy()
    if len(a0200_data) > 0:
        print(f"   数据点数量: {len(a0200_data)}")
        print(f"   航向范围: {a0200_data['Heading_deg'].min():.1f}° - {a0200_data['Heading_deg'].max():.1f}°")
        
        # 检查是否有右侧crank机动的航向特征
        right_crank_data = a0200_data[(a0200_data['Heading_deg'] >= 60) & (a0200_data['Heading_deg'] <= 76)]
        print(f"   右侧crank航向 (60°-76°): {len(right_crank_data)} 个数据点 ({len(right_crank_data)/len(a0200_data)*100:.1f}%)")
        
        if len(right_crank_data) > 0:
            print(f"   右侧crank时间范围: {right_crank_data['Time_s'].min():.1f}s - {right_crank_data['Time_s'].max():.1f}s")
            right_crank_actions = right_crank_data['Action_Type'].value_counts()
            print(f"   右侧crank期间动作分布:")
            for action, count in right_crank_actions.items():
                percentage = (count / len(right_crank_data)) * 100
                print(f"     {action}: {count} ({percentage:.1f}%)")
    else:
        print("   ❌ 未找到A0200数据")
    print()
    
    # 重点分析B0200的返航机动
    print("🔄 B0200返航分析:")
    b0200_data = df[df['Agent_ID'] == 'B0200'].copy()
    if len(b0200_data) > 0:
        print(f"   数据点数量: {len(b0200_data)}")
        print(f"   航向范围: {b0200_data['Heading_deg'].min():.1f}° - {b0200_data['Heading_deg'].max():.1f}°")
        
        # 检查南向和北向飞行
        south_data = b0200_data[(b0200_data['Heading_deg'] >= 170) & (b0200_data['Heading_deg'] <= 190)]
        north_data = b0200_data[((b0200_data['Heading_deg'] >= 350) | (b0200_data['Heading_deg'] <= 10))]
        
        print(f"   南向飞行 (170°-190°): {len(south_data)} 个数据点 ({len(south_data)/len(b0200_data)*100:.1f}%)")
        print(f"   北向飞行 (350°-10°): {len(north_data)} 个数据点 ({len(north_data)/len(b0200_data)*100:.1f}%)")
        
        if len(north_data) > 0:
            print(f"   返航时间范围: {north_data['Time_s'].min():.1f}s - {north_data['Time_s'].max():.1f}s")
            north_actions = north_data['Action_Type'].value_counts()
            print(f"   返航期间动作分布:")
            for action, count in north_actions.items():
                percentage = (count / len(north_data)) * 100
                print(f"     {action}: {count} ({percentage:.1f}%)")
    else:
        print("   ❌ 未找到B0200数据")
    print()
    
    # 检查动作标注器实例化
    print("🔧 动作标注器测试:")
    try:
        annotator = ActionAnnotator()
        print(f"   ✅ 动作标注器创建成功")
        print(f"   动作类型数量: {len(annotator.ACTION_TYPES)}")
        print(f"   动作类型: {list(annotator.ACTION_TYPES.values())}")
    except Exception as e:
        print(f"   ❌ 动作标注器创建失败: {e}")
    
    print()
    
    # 最终诊断
    print("🏥 问题诊断:")
    
    # 检查是否所有动作都是"平飞"
    all_level_flight = all(action == '平飞' for action in df['Action_Type'])
    if all_level_flight:
        print("   ❌ 严重问题: 所有动作都被标注为'平飞'")
        print("   💡 可能原因:")
        print("     1. 动作识别函数返回None，回退到默认值")
        print("     2. 动作识别条件过于严格，无法匹配")
        print("     3. 历史数据不足，无法计算航向变化率")
        print("     4. 战术任务对象传递有问题")
    else:
        print("   ✅ 动作标注有多样性")
    
    # 检查是否有预期的机动
    has_crank = any('Crank' in action or 'crank' in action for action in df['Action_Type'])
    if not has_crank:
        print("   ❌ 问题: 未检测到任何Crank机动")
        print("   💡 这表明航向变化识别逻辑可能有问题")
    else:
        print("   ✅ 检测到Crank机动")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    debug_action_annotator()
