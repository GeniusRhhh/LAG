#!/usr/bin/env python3
"""
简单的动作标注器测试
直接测试动作识别逻辑
"""

import pandas as pd
import os
import sys
sys.path.append('.')

from scripts.drag_shoot_2v2.action_annotator import ActionAnnotator

def test_simple_action_annotation():
    """简单测试动作标注器"""
    
    # 查找最新的轨迹数据文件
    results_dir = "scripts/drag_shoot_2v2/air_combat_results"
    csv_files = [f for f in os.listdir(results_dir) if f.startswith('drag_shoot_trajectory_') and f.endswith('.csv')]
    
    if not csv_files:
        print("未找到轨迹数据文件")
        return
    
    # 选择最新的文件
    latest_file = sorted(csv_files)[-1]
    csv_path = os.path.join(results_dir, latest_file)
    
    print("=" * 60)
    print("🧪 简单动作标注器测试")
    print("=" * 60)
    print(f"测试文件: {latest_file}")
    print()
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 筛选B0200的数据
    b0200_data = df[df['Agent_ID'] == 'B0200'].copy()
    
    if len(b0200_data) == 0:
        print("❌ 未找到B0200的数据")
        return
    
    print(f"📊 B0200数据点数量: {len(b0200_data)}")
    print(f"📊 时间范围: {b0200_data['Time_s'].min():.1f}s - {b0200_data['Time_s'].max():.1f}s")
    print(f"📊 航向范围: {b0200_data['Heading_deg'].min():.1f}° - {b0200_data['Heading_deg'].max():.1f}°")
    print()
    
    # 检查航向变化
    print("🔍 航向变化分析:")
    
    # 早期南向飞行
    south_data = b0200_data[(b0200_data['Heading_deg'] >= 170) & (b0200_data['Heading_deg'] <= 190)]
    print(f"   南向飞行 (170°-190°): {len(south_data)} 个数据点")
    if len(south_data) > 0:
        print(f"   南向时间范围: {south_data['Time_s'].min():.1f}s - {south_data['Time_s'].max():.1f}s")
    
    # 后期北向飞行
    north_data = b0200_data[((b0200_data['Heading_deg'] >= 350) | (b0200_data['Heading_deg'] <= 10))]
    print(f"   北向飞行 (350°-10°): {len(north_data)} 个数据点")
    if len(north_data) > 0:
        print(f"   北向时间范围: {north_data['Time_s'].min():.1f}s - {north_data['Time_s'].max():.1f}s")
    
    print()
    
    # 检查动作标注分布
    print("🎯 动作标注分布:")
    action_counts = b0200_data['Action_Type'].value_counts()
    for action, count in action_counts.items():
        percentage = (count / len(b0200_data)) * 100
        print(f"   {action}: {count} ({percentage:.1f}%)")
    
    print()
    
    # 手动测试动作识别逻辑
    print("🔧 手动测试动作识别:")
    
    # 创建动作标注器
    annotator = ActionAnnotator()
    
    # 模拟一些历史数据
    print("   初始化历史数据...")
    
    # 添加一些南向飞行的历史数据
    for i in range(5):
        time_point = 100.0 + i * 2.0
        annotator.history['B0200'] = annotator.history.get('B0200', [])
        annotator.history['B0200'].append({
            'heading': 180.0,
            'pitch': 0.0,
            'roll': 0.0,
            'velocity': 300.0,
            'time': time_point
        })
    
    print(f"   历史数据点数: {len(annotator.history['B0200'])}")
    
    # 测试北向飞行状态
    current_state = {
        'heading': 0.0,  # 北向
        'pitch': 0.0,
        'roll': 0.0,
        'velocity': 300.0,
        'x': 50000.0,
        'y': 6000.0,
        'z': 6000.0
    }
    
    current_time = 290.0
    
    print(f"   测试状态: 航向={current_state['heading']:.1f}°, 时间={current_time:.1f}s")
    
    # 测试敌方动作识别
    try:
        enemy_action = annotator._identify_enemy_precise_action('B0200', current_state, current_time, None)
        print(f"   敌方动作识别结果: {enemy_action}")
        
        # 测试航向变化率计算
        heading_rate = annotator._calculate_heading_change_rate('B0200', current_state['heading'], current_time)
        print(f"   航向变化率: {heading_rate:.2f}°/s")
        
    except Exception as e:
        print(f"   ❌ 动作识别出错: {e}")
    
    print()
    print("=" * 60)

if __name__ == "__main__":
    test_simple_action_annotation()
