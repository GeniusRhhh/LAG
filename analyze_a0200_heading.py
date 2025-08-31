#!/usr/bin/env python3
"""分析A0200僚机Short Skate阶段的实际航向变化"""

import pandas as pd
import numpy as np

def analyze_a0200_heading():
    """分析A0200僚机的航向变化"""
    
    # 读取最新的CSV文件
    csv_file = 'scripts/drag_shoot_2v2/high_low_attack_results/high_low_attack_trajectory_20250829_222739.csv'
    
    try:
        df = pd.read_csv(csv_file)
        
        # 分析A0200僚机的Short Skate阶段数据
        a0200_data = df[df['Agent_ID'] == 'A0200'].copy()
        short_skate_data = a0200_data[a0200_data['Action_Type'] == 'Short skate'].copy()
        
        print('=== A0200僚机Short Skate阶段分析 ===')
        print(f'Short Skate数据点数量: {len(short_skate_data)}')
        
        if len(short_skate_data) > 0:
            print(f'Short Skate时间范围: {short_skate_data["Time_s"].min():.1f}s - {short_skate_data["Time_s"].max():.1f}s')
            
            # 分析航向变化
            short_skate_data = short_skate_data.sort_values('Time_s').reset_index(drop=True)
            
            # 计算航向变化
            heading_changes = []
            for i in range(1, len(short_skate_data)):
                current_heading = short_skate_data.loc[i, 'Heading_deg']
                prev_heading = short_skate_data.loc[i-1, 'Heading_deg']
                
                # 计算航向变化，处理360度跨越
                change = current_heading - prev_heading
                if change > 180:
                    change -= 360
                elif change < -180:
                    change += 360
                
                heading_changes.append(change)
            
            if heading_changes:
                print()
                print('航向变化分析:')
                print(f'平均航向变化: {np.mean(heading_changes):.2f}度')
                print(f'最大正向变化: {max(heading_changes):.2f}度')
                print(f'最大负向变化: {min(heading_changes):.2f}度')
                
                # 统计左转和右转的数量
                left_turns = sum(1 for change in heading_changes if change < -2)
                right_turns = sum(1 for change in heading_changes if change > 2)
                
                print(f'显著左转次数 (< -2度): {left_turns}')
                print(f'显著右转次数 (> 2度): {right_turns}')
                
                # 显示前10个数据点的详细信息
                print()
                print('前10个Short Skate数据点:')
                for i in range(min(10, len(short_skate_data))):
                    row = short_skate_data.iloc[i]
                    change = heading_changes[i-1] if i > 0 else 0.0
                    print(f'时间: {row["Time_s"]:6.1f}s, 航向: {row["Heading_deg"]:6.1f}°, 变化: {change:6.2f}°, 当前标注: {row["Direction"]}')
        
        else:
            print('未找到A0200的Short Skate数据')
            
        # 分析A0200在85秒后的所有数据
        print()
        print('=== A0200在85秒后的航向变化分析 ===')
        a0200_after_85 = a0200_data[a0200_data['Time_s'] >= 85.0].copy()
        
        if len(a0200_after_85) > 10:
            a0200_after_85 = a0200_after_85.sort_values('Time_s').reset_index(drop=True)
            
            print('85秒后前20个数据点:')
            for i in range(min(20, len(a0200_after_85))):
                row = a0200_after_85.iloc[i]
                if i > 0:
                    prev_row = a0200_after_85.iloc[i-1]
                    change = row['Heading_deg'] - prev_row['Heading_deg']
                    if change > 180:
                        change -= 360
                    elif change < -180:
                        change += 360
                else:
                    change = 0.0
                
                print(f'时间: {row["Time_s"]:6.1f}s, 航向: {row["Heading_deg"]:6.1f}°, 变化: {change:6.2f}°, 动作: {row["Action_Type"]}, 方向: {row["Direction"]}')
        
    except Exception as e:
        print(f'错误: {e}')

if __name__ == "__main__":
    analyze_a0200_heading()
