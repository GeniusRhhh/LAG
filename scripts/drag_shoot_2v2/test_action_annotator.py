#!/usr/bin/env python3
"""
测试动作标注器的逻辑
专门测试B0200的动作识别
"""

import pandas as pd
import os
import sys
sys.path.append('.')

from scripts.drag_shoot_2v2.action_annotator import ActionAnnotator

def test_action_annotator():
    """测试动作标注器对B0200的识别"""
    
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
    print("🧪 动作标注器测试")
    print("=" * 80)
    print(f"测试文件: {latest_file}")
    print()
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 筛选B0200的数据
    b0200_data = df[df['Agent_ID'] == 'B0200'].copy()
    
    if len(b0200_data) == 0:
        print("❌ 未找到B0200的数据")
        return
    
    # 创建动作标注器
    annotator = ActionAnnotator()
    
    # 测试几个关键时间点
    test_times = [0, 50, 100, 150, 200, 250, 260, 270, 280, 290]
    
    print("🔍 测试关键时间点的动作识别:")
    print()
    
    for test_time in test_times:
        # 找到最接近的数据点
        closest_idx = (b0200_data['Time_s'] - test_time).abs().idxmin()
        row = b0200_data.loc[closest_idx]
        
        # 构造状态字典
        current_state = {
            'heading': row['Heading_deg'],
            'pitch': row['Pitch_deg'],
            'roll': row['Roll_deg'],
            'velocity': row['Velocity_m_s'],
            'x': row['X_m'],
            'y': row['Y_m'],
            'z': row['Z_m']
        }
        
        # 模拟历史数据
        history_data = b0200_data[b0200_data['Time_s'] <= row['Time_s']].tail(10)
        annotator.history['B0200'] = []
        for _, hist_row in history_data.iterrows():
            annotator.history['B0200'].append({
                'heading': hist_row['Heading_deg'],
                'pitch': hist_row['Pitch_deg'],
                'roll': hist_row['Roll_deg'],
                'velocity': hist_row['Velocity_m_s'],
                'time': hist_row['Time_s']
            })
        
        # 测试敌方精确动作识别
        try:
            enemy_action = annotator._identify_enemy_precise_action('B0200', current_state, row['Time_s'], None)
            trajectory_action = annotator._identify_trajectory_based_action('B0200', current_state, row['Time_s'])
            
            print(f"t={row['Time_s']:6.1f}s: heading={row['Heading_deg']:6.1f}°")
            print(f"           敌方精确识别: {enemy_action}")
            print(f"           轨迹基础识别: {trajectory_action}")
            print(f"           实际标注: {row['Action_Type']}")
            print()
            
        except Exception as e:
            print(f"t={row['Time_s']:6.1f}s: 识别出错 - {e}")
            print()
    
    print("=" * 80)

if __name__ == "__main__":
    test_action_annotator()
