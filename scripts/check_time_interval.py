#!/usr/bin/env python3
"""检查CSV文件的时间间隔"""

import pandas as pd
import os

def check_time_intervals():
    """检查CSV文件的时间间隔"""
    print("🔍 检查CSV文件的时间间隔")
    print("=" * 50)
    
    csv_files = [
        'scripts/drag_shoot_2v2/basic_action_data/accelerate/accelerate_107mps_001.csv',
        'scripts/drag_shoot_2v2/basic_action_data/decelerate/decelerate_83mps_001.csv',
        'scripts/drag_shoot_2v2/basic_action_data/level_flight/level_flight_20s_001.csv'
    ]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f'\n📊 检查文件: {os.path.basename(csv_file)}')
            df = pd.read_csv(csv_file)
            
            print(f'总行数: {len(df)}')
            print(f'前10行时间戳:')
            for i in range(min(10, len(df))):
                time_val = df['Time_s'].iloc[i]
                print(f'  行{i}: {time_val:.1f}s')
            
            if len(df) > 1:
                print(f'\n时间间隔分析:')
                time_intervals = []
                for i in range(1, min(11, len(df))):
                    interval = df['Time_s'].iloc[i] - df['Time_s'].iloc[i-1]
                    time_intervals.append(interval)
                    print(f'  间隔 {i-1}->{i}: {interval:.1f}s')
                
                if time_intervals:
                    avg_interval = sum(time_intervals) / len(time_intervals)
                    print(f'\n平均时间间隔: {avg_interval:.1f}s')
                    
                    if abs(avg_interval - 0.2) < 0.05:
                        print("✅ 时间间隔正确 (0.2s)")
                    elif abs(avg_interval - 0.4) < 0.05:
                        print("❌ 时间间隔错误 (0.4s，应该是0.2s)")
                    else:
                        print(f"⚠️ 时间间隔异常 ({avg_interval:.1f}s)")
            break
    
    # 检查配置参数
    print(f'\n🔧 检查配置参数:')
    try:
        import sys
        sys.path.append('.')
        from envs.JSBSim.envs.env_base import BaseEnv
        from envs.JSBSim.utils.utils import parse_config
        
        config = parse_config('simple_maneuver_config')
        agent_interaction_steps = getattr(config, 'agent_interaction_steps', 12)
        sim_freq = getattr(config, 'sim_freq', 60)
        time_interval = agent_interaction_steps / sim_freq
        
        print(f'agent_interaction_steps: {agent_interaction_steps}')
        print(f'sim_freq: {sim_freq}')
        print(f'计算的time_interval: {time_interval:.1f}s')
        
        if abs(time_interval - 0.2) < 0.01:
            print("✅ 配置参数正确")
        else:
            print("❌ 配置参数错误")
            
    except Exception as e:
        print(f"配置检查失败: {e}")

if __name__ == "__main__":
    check_time_intervals()
