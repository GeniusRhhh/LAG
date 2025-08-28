#!/usr/bin/env python3
"""
调试B0200航向变化率计算
分析为什么航向变化率显示为0.00°/s
"""

import pandas as pd
import os
import numpy as np

def debug_heading_change():
    """调试B0200的航向变化率计算"""
    
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
    print("🔍 B0200航向变化率调试分析")
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
    
    # 按时间排序
    b0200_data = b0200_data.sort_values('Time_s').reset_index(drop=True)
    
    print(f"📊 B0200基础信息:")
    print(f"   数据点数量: {len(b0200_data)}")
    print(f"   时间范围: {b0200_data['Time_s'].min():.1f}s - {b0200_data['Time_s'].max():.1f}s")
    print()
    
    # 计算航向变化率
    b0200_data['heading_diff'] = b0200_data['Heading_deg'].diff()
    b0200_data['time_diff'] = b0200_data['Time_s'].diff()
    
    # 处理航向角度跨越问题
    mask_pos = b0200_data['heading_diff'] > 180
    mask_neg = b0200_data['heading_diff'] < -180
    b0200_data.loc[mask_pos, 'heading_diff'] -= 360
    b0200_data.loc[mask_neg, 'heading_diff'] += 360
    
    # 计算航向变化率 (度/秒)
    b0200_data['heading_rate'] = b0200_data['heading_diff'] / b0200_data['time_diff']
    
    print("🧭 航向变化分析:")
    print(f"   最大航向变化率: {b0200_data['heading_rate'].max():.2f}°/s")
    print(f"   最小航向变化率: {b0200_data['heading_rate'].min():.2f}°/s")
    print(f"   平均航向变化率: {b0200_data['heading_rate'].mean():.2f}°/s")
    print(f"   标准差: {b0200_data['heading_rate'].std():.2f}°/s")
    print()
    
    # 分析返航阶段
    print("🔄 返航阶段分析:")
    rtb_start_time = 117.8  # 从之前的分析得出
    rtb_data = b0200_data[b0200_data['Time_s'] >= rtb_start_time].copy()
    
    if len(rtb_data) > 0:
        print(f"   返航阶段数据点: {len(rtb_data)}")
        print(f"   返航阶段航向范围: {rtb_data['Heading_deg'].min():.1f}° - {rtb_data['Heading_deg'].max():.1f}°")
        print(f"   返航阶段最大航向变化率: {rtb_data['heading_rate'].max():.2f}°/s")
        print(f"   返航阶段最小航向变化率: {rtb_data['heading_rate'].min():.2f}°/s")
        print(f"   返航阶段平均航向变化率: {rtb_data['heading_rate'].mean():.2f}°/s")
        print()
        
        # 分析返航转向阶段（前50个数据点）
        transition_data = rtb_data.head(50)
        print(f"   返航转向阶段 (前50个数据点):")
        print(f"     时间范围: {transition_data['Time_s'].min():.1f}s - {transition_data['Time_s'].max():.1f}s")
        print(f"     航向范围: {transition_data['Heading_deg'].min():.1f}° - {transition_data['Heading_deg'].max():.1f}°")
        print(f"     最大航向变化率: {transition_data['heading_rate'].max():.2f}°/s")
        print(f"     最小航向变化率: {transition_data['heading_rate'].min():.2f}°/s")
        print(f"     平均航向变化率: {transition_data['heading_rate'].mean():.2f}°/s")
        print()
    
    # 分析高航向变化率的时间点
    high_rate_data = b0200_data[abs(b0200_data['heading_rate']) > 1.0].copy()
    
    print("📈 高航向变化率分析 (>1.0°/s):")
    if len(high_rate_data) > 0:
        print(f"   高变化率数据点: {len(high_rate_data)}")
        print(f"   时间范围: {high_rate_data['Time_s'].min():.1f}s - {high_rate_data['Time_s'].max():.1f}s")
        print(f"   最大变化率: {high_rate_data['heading_rate'].max():.2f}°/s")
        print(f"   最小变化率: {high_rate_data['heading_rate'].min():.2f}°/s")
        print()
        
        # 显示前10个高变化率的时间点
        print("   前10个高变化率时间点:")
        for i, row in high_rate_data.head(10).iterrows():
            print(f"     t={row['Time_s']:.1f}s: {row['Heading_deg']:.1f}° -> 变化率={row['heading_rate']:.2f}°/s")
    else:
        print("   ❌ 未发现高航向变化率数据点")
    
    print()
    
    # 分析调试时间段 (123-125s)
    debug_data = b0200_data[(b0200_data['Time_s'] >= 123) & (b0200_data['Time_s'] <= 125)].copy()
    
    print("🐛 调试时间段分析 (123-125s):")
    if len(debug_data) > 0:
        print(f"   调试时间段数据点: {len(debug_data)}")
        print("   详细数据:")
        for i, row in debug_data.iterrows():
            print(f"     t={row['Time_s']:.1f}s: heading={row['Heading_deg']:.1f}°, rate={row['heading_rate']:.2f}°/s")
    else:
        print("   ❌ 调试时间段无数据")
    
    print()
    
    # 最终评估
    print("🏆 最终评估:")
    
    # 检查是否有显著的航向变化
    total_heading_change = abs(b0200_data['Heading_deg'].iloc[-1] - b0200_data['Heading_deg'].iloc[0])
    if total_heading_change > 180:
        total_heading_change = 360 - total_heading_change
    
    print(f"   总航向变化: {total_heading_change:.1f}°")
    
    has_significant_change = total_heading_change > 90
    has_high_rate = len(high_rate_data) > 0
    
    print(f"   ✅ 有显著航向变化 (>90°): {'是' if has_significant_change else '否'}")
    print(f"   ✅ 有高变化率时刻 (>1.0°/s): {'是' if has_high_rate else '否'}")
    
    if has_significant_change and not has_high_rate:
        print("   ⚠️  问题: 有显著航向变化但无高变化率时刻")
        print("   💡 可能原因: 航向变化过于缓慢，分布在很长时间内")
    elif has_significant_change and has_high_rate:
        print("   🎉 B0200航向变化正常，应该能被识别为Crank机动！")
    else:
        print("   ❌ B0200航向变化异常")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    debug_heading_change()
