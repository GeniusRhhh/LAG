#!/usr/bin/env python3
"""
最终动作标注验证脚本
专门验证敌方飞机B0200的返航机动识别
"""

import pandas as pd
import os
import numpy as np

def analyze_b0200_actions():
    """分析B0200的动作标注情况"""
    
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
    print("🔍 B0200动作标注最终验证")
    print("=" * 80)
    print(f"分析文件: {latest_file}")
    print()
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 筛选B0200的数据
    b0200_data = df[df['Agent_ID'] == 'B0200'].copy()
    
    if len(b0200_data) == 0:
        print("❌ 未找到B0200的数据")
        return
    
    print(f"📊 B0200基础信息:")
    print(f"   数据点数量: {len(b0200_data)}")
    print(f"   时间范围: {b0200_data['Time_s'].min():.1f}s - {b0200_data['Time_s'].max():.1f}s")
    print()
    
    # 分析航向变化
    print("🧭 航向分析:")
    south_data = b0200_data[(b0200_data['Heading_deg'] >= 170) & (b0200_data['Heading_deg'] <= 190)]
    north_data = b0200_data[((b0200_data['Heading_deg'] >= 350) | (b0200_data['Heading_deg'] <= 10))]
    
    print(f"   南向飞行 (170°-190°): {len(south_data)} 个数据点 ({len(south_data)/len(b0200_data)*100:.1f}%)")
    print(f"   北向飞行 (350°-10°): {len(north_data)} 个数据点 ({len(north_data)/len(b0200_data)*100:.1f}%)")
    
    if len(north_data) > 0:
        print(f"   北向飞行时间范围: {north_data['Time_s'].min():.1f}s - {north_data['Time_s'].max():.1f}s")
    
    print()
    
    # 分析动作标注
    print("🎯 动作标注分析:")
    action_counts = b0200_data['Action_Type'].value_counts()
    for action, count in action_counts.items():
        percentage = (count / len(b0200_data)) * 100
        print(f"   {action}: {count} ({percentage:.1f}%)")
    
    print()
    
    # 分析返航阶段的动作标注
    print("🔄 返航阶段分析:")
    if len(north_data) > 0:
        rtb_start_time = north_data['Time_s'].min()
        rtb_data = b0200_data[b0200_data['Time_s'] >= rtb_start_time]
        
        print(f"   返航开始时间: {rtb_start_time:.1f}s")
        print(f"   返航阶段数据点: {len(rtb_data)}")
        
        rtb_actions = rtb_data['Action_Type'].value_counts()
        print(f"   返航阶段动作分布:")
        for action, count in rtb_actions.items():
            percentage = (count / len(rtb_data)) * 100
            print(f"     {action}: {count} ({percentage:.1f}%)")
        
        # 检查返航转向阶段
        transition_data = rtb_data.head(50)  # 返航开始的前50个数据点
        print(f"   返航转向阶段 (前50个数据点):")
        transition_actions = transition_data['Action_Type'].value_counts()
        for action, count in transition_actions.items():
            percentage = (count / len(transition_data)) * 100
            print(f"     {action}: {count} ({percentage:.1f}%)")
    else:
        print("   ❌ 未检测到返航机动")
    
    print()
    
    # 分析航向变化率
    print("📈 航向变化率分析:")
    b0200_data['heading_change'] = b0200_data['Heading_deg'].diff().abs()
    
    # 处理跨越0°/360°的情况
    mask = b0200_data['heading_change'] > 180
    b0200_data.loc[mask, 'heading_change'] = 360 - b0200_data.loc[mask, 'heading_change']
    
    high_change_data = b0200_data[b0200_data['heading_change'] > 2.0]
    
    print(f"   高航向变化率 (>2°/step): {len(high_change_data)} 个数据点")
    if len(high_change_data) > 0:
        print(f"   高变化率时间范围: {high_change_data['Time_s'].min():.1f}s - {high_change_data['Time_s'].max():.1f}s")
        high_change_actions = high_change_data['Action_Type'].value_counts()
        print(f"   高变化率期间动作分布:")
        for action, count in high_change_actions.items():
            percentage = (count / len(high_change_data)) * 100
            print(f"     {action}: {count} ({percentage:.1f}%)")
    
    print()
    
    # 最终评估
    print("🏆 最终评估:")
    
    # 检查是否正确识别了返航机动
    has_north_flight = len(north_data) > 0
    has_crank_action = 'Crank' in ' '.join(action_counts.index) or 'crank' in ' '.join(action_counts.index)
    has_diverse_actions = len(action_counts) > 1
    
    print(f"   ✅ 检测到北向飞行: {'是' if has_north_flight else '否'}")
    print(f"   ✅ 识别到Crank机动: {'是' if has_crank_action else '否'}")
    print(f"   ✅ 动作类型多样性: {'是' if has_diverse_actions else '否'}")
    
    if has_north_flight and not has_crank_action:
        print("   ⚠️  问题: 检测到返航机动但未正确标注为Crank")
        print("   💡 建议: 检查动作标注器的敌方识别逻辑")
    
    if has_north_flight and has_diverse_actions:
        print("   🎉 B0200动作识别基本正确！")
    elif has_north_flight:
        print("   ⚠️  B0200动作识别部分正确，需要改进动作多样性")
    else:
        print("   ❌ B0200动作识别存在重大问题")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_b0200_actions()
