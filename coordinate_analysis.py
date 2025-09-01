#!/usr/bin/env python3
"""
坐标和角度问题分析脚本
分析CSV数据中的坐标计算错误和角度处理问题
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple

def analyze_coordinate_issues():
    """分析坐标问题"""
    print("🔍 **坐标和角度问题分析**")
    print("=" * 80)
    
    # 分析问题样本
    problem_samples = [
        ("climb_left", "climb_left_1517m_239deg_004.csv", "1517米高度增益，239度转弯"),
        ("climb", "climb_1640m_001.csv", "1640米高度增益，基础爬升"),
        ("turn_left", "turn_left_204deg_001.csv", "204度左转"),
        ("turn_right", "turn_right_214deg_003.csv", "214度右转")
    ]
    
    base_dir = "scripts/drag_shoot_2v2/basic_action_data"
    
    for action_type, filename, description in problem_samples:
        file_path = os.path.join(base_dir, action_type, filename)
        
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            continue
            
        print(f"\n📊 **分析 {action_type}: {description}**")
        print("-" * 60)
        
        try:
            df = pd.read_csv(file_path)
            
            if len(df) == 0:
                print("❌ 文件为空")
                continue
            
            # 基本信息
            print(f"数据行数: {len(df)}")
            print(f"时间范围: {df['Time_s'].min():.1f}s - {df['Time_s'].max():.1f}s")
            print(f"持续时间: {df['Time_s'].max() - df['Time_s'].min():.1f}s")
            
            # 代理过滤检查
            unique_agents = df['Agent_ID'].unique()
            print(f"代理数量: {len(unique_agents)} ({', '.join(unique_agents)})")
            if len(unique_agents) > 1:
                print("⚠️ 包含多个代理数据，应该只有A0100")
            
            # 坐标分析
            initial_pos = (df.iloc[0]['X_m'], df.iloc[0]['Y_m'], df.iloc[0]['Z_m'])
            final_pos = (df.iloc[-1]['X_m'], df.iloc[-1]['Y_m'], df.iloc[-1]['Z_m'])
            
            print(f"初始位置: X={initial_pos[0]:.1f}, Y={initial_pos[1]:.1f}, Z={initial_pos[2]:.1f}")
            print(f"最终位置: X={final_pos[0]:.1f}, Y={final_pos[1]:.1f}, Z={final_pos[2]:.1f}")
            
            # 位置变化
            dx = final_pos[0] - initial_pos[0]
            dy = final_pos[1] - initial_pos[1]
            dz = final_pos[2] - initial_pos[2]
            
            print(f"位置变化: ΔX={dx:.1f}m, ΔY={dy:.1f}m, ΔZ={dz:.1f}m")
            
            # 水平距离和方向
            horizontal_distance = np.sqrt(dx**2 + dy**2)
            if horizontal_distance > 0:
                bearing = np.rad2deg(np.arctan2(dx, dy))
                print(f"水平移动: {horizontal_distance:.1f}m, 方位角: {bearing:.1f}度")
            else:
                print("水平移动: 0m (无水平移动)")
            
            # 航向分析
            initial_heading = df.iloc[0]['Heading_deg']
            final_heading = df.iloc[-1]['Heading_deg']
            heading_change = final_heading - initial_heading
            
            # 处理角度包装问题
            if heading_change > 180:
                heading_change -= 360
            elif heading_change < -180:
                heading_change += 360
                
            print(f"航向变化: {initial_heading:.1f}° → {final_heading:.1f}° (变化: {heading_change:.1f}°)")
            
            # 高度分析
            if abs(dz) > 100:  # 显著高度变化
                print(f"高度变化: {dz:.1f}m ({'爬升' if dz > 0 else '下降'})")
                
                # 检查参数匹配
                if "climb" in action_type and "m" in filename:
                    param_height = int(filename.split('m')[0].split('_')[-1])
                    print(f"参数高度: {param_height}m")
                    print(f"实际高度变化: {dz:.1f}m")
                    if abs(abs(dz) - param_height) > 500:
                        print(f"⚠️ 参数不匹配！差异: {abs(abs(dz) - param_height):.1f}m")
                    else:
                        print("✅ 参数匹配良好")
            
            # 转弯角度分析
            if "deg" in filename:
                param_angle = int(filename.split('deg')[0].split('_')[-1])
                print(f"参数转弯角度: {param_angle}度")
                print(f"实际航向变化: {abs(heading_change):.1f}度")
                
                if abs(abs(heading_change) - param_angle) > 30:
                    print(f"⚠️ 转弯角度不匹配！差异: {abs(abs(heading_change) - param_angle):.1f}度")
                else:
                    print("✅ 转弯角度匹配良好")
            
            # 坐标合理性检查
            print(f"\n🔍 **坐标合理性检查**:")
            
            # X,Y坐标范围检查
            x_range = df['X_m'].max() - df['X_m'].min()
            y_range = df['Y_m'].max() - df['Y_m'].min()
            z_range = df['Z_m'].max() - df['Z_m'].min()
            
            print(f"坐标变化范围: X={x_range:.1f}m, Y={y_range:.1f}m, Z={z_range:.1f}m")
            
            # 检查坐标是否过大
            if initial_pos[0] > 10000000 or initial_pos[1] > 10000000:
                print("⚠️ X,Y坐标值过大，可能存在单位转换错误")
                print(f"   X坐标: {initial_pos[0]:.1f} (应该在合理范围内)")
                print(f"   Y坐标: {initial_pos[1]:.1f} (应该在合理范围内)")
            else:
                print("✅ X,Y坐标值在合理范围内")
            
            # 速度分析
            initial_velocity = df.iloc[0]['Velocity_m_s']
            final_velocity = df.iloc[-1]['Velocity_m_s']
            velocity_change = final_velocity - initial_velocity
            
            print(f"速度变化: {initial_velocity:.1f} → {final_velocity:.1f} m/s (变化: {velocity_change:.1f} m/s)")
            
        except Exception as e:
            print(f"❌ 分析失败: {e}")

def identify_coordinate_calculation_errors():
    """识别坐标计算错误"""
    print(f"\n🔧 **坐标计算错误识别**")
    print("=" * 60)
    
    # 检查一个样本的原始经纬度数据
    sample_file = "scripts/drag_shoot_2v2/basic_action_data/level_flight/level_flight_250mps_001.csv"
    
    if os.path.exists(sample_file):
        df = pd.read_csv(sample_file)
        
        print("分析level_flight样本的坐标计算:")
        print(f"X坐标范围: {df['X_m'].min():.1f} - {df['X_m'].max():.1f}")
        print(f"Y坐标范围: {df['Y_m'].min():.1f} - {df['Y_m'].max():.1f}")
        print(f"Z坐标范围: {df['Z_m'].min():.1f} - {df['Z_m'].max():.1f}")
        
        # 计算坐标变化
        x_change = df['X_m'].max() - df['X_m'].min()
        y_change = df['Y_m'].max() - df['Y_m'].min()
        
        print(f"水平移动距离: {np.sqrt(x_change**2 + y_change**2):.1f}m")
        
        # 检查坐标是否合理
        if df['X_m'].iloc[0] > 1000000:
            print("⚠️ 坐标值过大，可能的问题:")
            print("   1. 经纬度转换公式错误")
            print("   2. 单位转换错误")
            print("   3. 坐标系选择错误")
            
            # 推测正确的坐标计算方法
            print(f"\n💡 **坐标修复建议**:")
            print("当前可能使用的转换:")
            print("  pos_x = lon_deg * 111320 * cos(lat_rad)")
            print("  pos_y = lat_deg * 111320")
            print("建议修复:")
            print("  1. 使用相对坐标而非绝对坐标")
            print("  2. 或使用UTM坐标系")
            print("  3. 或使用局部坐标系（以初始位置为原点）")

def analyze_large_angle_issues():
    """分析大角度转弯问题"""
    print(f"\n🔄 **大角度转弯问题分析**")
    print("=" * 60)
    
    large_angle_samples = [
        "turn_left/turn_left_204deg_001.csv",
        "turn_right/turn_right_214deg_003.csv",
        "climb_left/climb_left_1517m_239deg_004.csv"
    ]
    
    base_dir = "scripts/drag_shoot_2v2/basic_action_data"
    
    for sample_path in large_angle_samples:
        file_path = os.path.join(base_dir, sample_path)
        
        if not os.path.exists(file_path):
            continue
            
        print(f"\n分析大角度样本: {sample_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            # 提取参数角度
            filename = os.path.basename(sample_path)
            if "deg" in filename:
                param_angle = int(filename.split('deg')[0].split('_')[-1])
                print(f"参数角度: {param_angle}度")
                
                if param_angle > 180:
                    print(f"⚠️ 大角度转弯 (>{180}度)")
                    print("可能的问题:")
                    print("  1. 角度包装处理不正确")
                    print("  2. 转弯方向计算错误")
                    print("  3. 多圈转弯实现问题")
            
            # 分析航向变化
            headings = df['Heading_deg'].values
            heading_changes = []
            
            for i in range(1, len(headings)):
                change = headings[i] - headings[i-1]
                # 处理角度包装
                if change > 180:
                    change -= 360
                elif change < -180:
                    change += 360
                heading_changes.append(change)
            
            total_heading_change = sum(heading_changes)
            print(f"实际总航向变化: {total_heading_change:.1f}度")
            
            # 检查是否有异常的角度跳跃
            large_jumps = [abs(change) for change in heading_changes if abs(change) > 30]
            if large_jumps:
                print(f"⚠️ 发现异常角度跳跃: {len(large_jumps)}次，最大: {max(large_jumps):.1f}度")
            
        except Exception as e:
            print(f"❌ 分析失败: {e}")

if __name__ == "__main__":
    analyze_coordinate_issues()
    identify_coordinate_calculation_errors()
    analyze_large_angle_issues()
    
    print(f"\n🎯 **问题总结和修复建议**")
    print("=" * 80)
    print("发现的主要问题:")
    print("1. ⚠️ X,Y坐标值过大 - 坐标转换公式需要修复")
    print("2. ⚠️ 大角度转弯处理 - 需要正确的角度包装和多圈转弯支持")
    print("3. ⚠️ 参数与实际行为不匹配 - 需要验证动作执行逻辑")
    print("4. ⚠️ 可能包含多代理数据 - 需要确保只记录A0100")
    
    print(f"\n修复优先级:")
    print("1. 🔧 修复坐标计算公式（使用相对坐标或局部坐标系）")
    print("2. 🔧 修复大角度转弯处理（正确的角度包装）")
    print("3. 🔧 验证动作参数与实际行为的一致性")
    print("4. 🔧 确保CSV数据只包含A0100代理")
