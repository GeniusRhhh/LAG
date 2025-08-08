#!/usr/bin/env python3
"""
分析编队间距 - 专门分析30-90秒时间段的编队间距数据
"""

import pandas as pd
import numpy as np
import sys
import os

def calculate_distance(x1, y1, x2, y2):
    """计算两点之间的距离（米）"""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def analyze_formation_spacing(csv_file):
    """分析编队间距"""
    print("🔍 分析编队间距数据...")
    
    # 读取轨迹数据
    df = pd.read_csv(csv_file)
    
    # 筛选30-90秒时间段
    target_df = df[(df['Time_s'] >= 30.0) & (df['Time_s'] <= 90.0)]
    
    if target_df.empty:
        print("❌ 没有找到30-90秒时间段的数据")
        return
    
    # 获取时间点列表
    time_points = sorted(target_df['Time_s'].unique())
    
    print(f"📊 找到{len(time_points)}个时间点的数据")
    print(f"⏰ 时间范围: {time_points[0]:.1f}s - {time_points[-1]:.1f}s")
    print()
    
    # 分析每个时间点的编队间距
    spacing_data = []
    
    for time_point in time_points:
        # 获取该时间点的数据
        time_data = target_df[target_df['Time_s'] == time_point]
        
        # 获取A0100和A0200的位置
        a0100_data = time_data[time_data['Agent_ID'] == 'A0100']
        a0200_data = time_data[time_data['Agent_ID'] == 'A0200']
        
        if len(a0100_data) == 1 and len(a0200_data) == 1:
            # 提取位置信息
            a0100_x = a0100_data.iloc[0]['X_m']
            a0100_y = a0100_data.iloc[0]['Y_m']
            a0100_heading = a0100_data.iloc[0]['Heading_deg']
            
            a0200_x = a0200_data.iloc[0]['X_m']
            a0200_y = a0200_data.iloc[0]['Y_m']
            a0200_heading = a0200_data.iloc[0]['Heading_deg']
            
            # 计算编队间距
            distance_m = calculate_distance(a0100_x, a0100_y, a0200_x, a0200_y)
            distance_km = distance_m / 1000.0
            distance_nm = distance_m / 1852.0  # 海里
            
            # 判断是否在标准范围内 (5-10海里)
            in_range = 5.0 <= distance_nm <= 10.0
            status_icon = "✅" if in_range else "⚠️"
            
            # 存储数据
            spacing_data.append({
                'time': time_point,
                'distance_km': distance_km,
                'distance_nm': distance_nm,
                'in_range': in_range,
                'a0100_heading': a0100_heading,
                'a0200_heading': a0200_heading
            })
            
            # 每3秒打印一次（模拟原始监控逻辑）
            if abs(time_point % 3.0) < 0.1:  # 允许小的浮点误差
                print(f"🔍 CRANK机动后编队间距监控:")
                print(f"{status_icon} [时间: {time_point:.1f}s] 编队间距: {distance_km:.1f}km ({distance_nm:.1f}海里)")
                print(f"   长机A0100航向: {a0100_heading:.1f}°, 僚机A0200航向: {a0200_heading:.1f}°")
                
                if not in_range:
                    if distance_nm < 5.0:
                        print(f"⚠️  编队间距过近！当前{distance_nm:.1f}海里 < 标准5海里")
                    else:
                        print(f"⚠️  编队间距过远！当前{distance_nm:.1f}海里 > 标准10海里")
                else:
                    print(f"✅  编队间距符合标准！{distance_nm:.1f}海里在5-10海里范围内")
                print()
    
    # 统计分析
    if spacing_data:
        distances_nm = [d['distance_nm'] for d in spacing_data]
        avg_distance = np.mean(distances_nm)
        min_distance = np.min(distances_nm)
        max_distance = np.max(distances_nm)
        
        in_range_count = sum(1 for d in spacing_data if d['in_range'])
        total_count = len(spacing_data)
        compliance_rate = (in_range_count / total_count) * 100
        
        print("=" * 60)
        print("📈 编队间距统计分析 (30-90秒时间段)")
        print("=" * 60)
        print(f"⏰ 分析时间段: 30.0s - 90.0s")
        print(f"📊 数据点数量: {total_count}")
        print(f"📏 平均间距: {avg_distance:.1f}海里 ({avg_distance*1.852:.1f}km)")
        print(f"📏 最小间距: {min_distance:.1f}海里 ({min_distance*1.852:.1f}km)")
        print(f"📏 最大间距: {max_distance:.1f}海里 ({max_distance*1.852:.1f}km)")
        print(f"✅ 标准符合率: {compliance_rate:.1f}% ({in_range_count}/{total_count})")
        print()
        
        if avg_distance < 5.0:
            print("🔍 结论: 编队间距过近，需要增加crank机动幅度")
        elif avg_distance > 10.0:
            print("🔍 结论: 编队间距过远，需要减少crank机动幅度")
        else:
            print("🔍 结论: 编队间距基本符合5-10海里标准要求")

if __name__ == "__main__":
    # 使用最新的轨迹文件
    csv_file = "scripts/drag_shoot_2v2/air_combat_results/trajectory_20250808_104248.csv"

    if os.path.exists(csv_file):
        analyze_formation_spacing(csv_file)
    else:
        print(f"❌ 找不到轨迹文件: {csv_file}")
