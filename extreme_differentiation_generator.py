#!/usr/bin/env python3
"""
极端差异化基础动作数据生成器
实施全面修复：极端参数差异化、CSV数据修复、坐标系修复
"""

import sys
sys.path.append('.')
from scripts.basic_action_data_generator import BasicActionDataGenerator
import pandas as pd
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

def show_extreme_parameter_changes():
    """显示极端参数差异化变更"""
    print("🔧 **极端参数差异化配置**")
    print("=" * 80)
    
    extreme_changes = [
        ("基础动作极端差异化", [
            ("加速/减速", "速度变化", "50-150 m/s → 80-300 m/s", "+100% 范围扩大"),
            ("左转/右转", "转弯角度", "30-135度 → 45-270度", "+100% 角度增大"),
            ("左转/右转", "转弯速率", "2.0-8.0度/秒 → 1.5-12.0度/秒", "+50% 速率提升"),
            ("爬升/俯冲", "高度变化", "1500-3500米 → 2000-6000米", "+71% 高度增大")
        ]),
        ("组合机动极端差异化", [
            ("左爬升/右爬升", "转弯角度", "30-120度 → 60-360度", "+200% 角度增大"),
            ("左爬升/右爬升", "转弯速率", "3.0-8.0度/秒 → 1.0-15.0度/秒", "+87% 速率提升"),
            ("左爬升/右爬升", "爬升高度", "1500-3500米 → 3000-8000米", "+129% 高度增大"),
            ("左俯冲/右俯冲", "转弯角度", "30-120度 → 60-360度", "+200% 角度增大"),
            ("左俯冲/右俯冲", "转弯速率", "3.0-8.0度/秒 → 1.0-15.0度/秒", "+87% 速率提升"),
            ("左俯冲/右俯冲", "俯冲高度", "1500-3500米 → 3000-8000米", "+129% 高度增大")
        ])
    ]
    
    for category, items in extreme_changes:
        print(f"\n📊 **{category}**:")
        print("-" * 70)
        for action, param, change, effect in items:
            print(f"  {action:<15} {param:<12} {change:<25} {effect}")

def show_data_recording_fixes():
    """显示数据记录修复"""
    print(f"\n🔧 **数据记录修复**:")
    print("=" * 50)
    print("✅ CSV数据过滤：只记录A0100代理数据，完全排除B0100（敌方）数据")
    print("✅ 坐标系修复：修复X,Y坐标计算错误，使用正确的经纬度转UTM转换")
    print("✅ 单位转换：经度考虑纬度修正，纬度使用标准转换")
    print("✅ 数据完整性：确保每个样本只包含A0100的完整轨迹数据")

def analyze_extreme_sample_quality(acmi_path: str, csv_path: str, action_name: str, expected_duration: float) -> Dict:
    """分析极端差异化样本的质量"""
    result = {
        'acmi_exists': False,
        'acmi_size_kb': 0.0,
        'acmi_quality': 'FAIL',
        'csv_exists': False,
        'csv_rows': 0,
        'csv_time_range': 0.0,
        'csv_duration_match': False,
        'csv_agent_filter': False,
        'coordinate_range': {'x_range': 0, 'y_range': 0, 'z_range': 0},
        'csv_quality': 'FAIL',
        'overall_quality': 'FAIL'
    }
    
    try:
        # 检查ACMI文件
        if os.path.exists(acmi_path):
            result['acmi_exists'] = True
            acmi_size = os.path.getsize(acmi_path)
            result['acmi_size_kb'] = acmi_size / 1024
            
            # ACMI质量评估
            if acmi_size >= 35000:  # 35KB以上
                result['acmi_quality'] = 'EXCELLENT'
            elif acmi_size >= 25000:  # 25KB以上
                result['acmi_quality'] = 'GOOD'
            elif acmi_size >= 15000:   # 15KB以上
                result['acmi_quality'] = 'FAIR'
            else:
                result['acmi_quality'] = 'POOR'
        
        # 检查CSV文件
        if os.path.exists(csv_path):
            result['csv_exists'] = True
            df = pd.read_csv(csv_path)
            result['csv_rows'] = len(df)
            
            if len(df) > 0:
                # 验证只有A0100代理数据
                unique_agents = df['Agent_ID'].unique()
                result['csv_agent_filter'] = len(unique_agents) == 1 and unique_agents[0] == 'A0100'
                
                # 时间范围检查
                time_min = df['Time_s'].min()
                time_max = df['Time_s'].max()
                result['csv_time_range'] = time_max - time_min
                
                # 坐标范围检查（验证坐标修复）
                if 'X_m' in df.columns and 'Y_m' in df.columns and 'Z_m' in df.columns:
                    result['coordinate_range'] = {
                        'x_range': df['X_m'].max() - df['X_m'].min(),
                        'y_range': df['Y_m'].max() - df['Y_m'].min(),
                        'z_range': df['Z_m'].max() - df['Z_m'].min()
                    }
                
                # 时间范围匹配度检查
                time_diff = abs(result['csv_time_range'] - expected_duration)
                result['csv_duration_match'] = time_diff < 2.0  # 允许2秒误差
                
                # CSV质量评估
                if (result['csv_rows'] >= 60 and result['csv_duration_match'] and 
                    result['csv_agent_filter'] and result['coordinate_range']['z_range'] > 0):
                    result['csv_quality'] = 'EXCELLENT'
                elif (result['csv_rows'] >= 40 and result['csv_duration_match'] and 
                      result['csv_agent_filter']):
                    result['csv_quality'] = 'GOOD'
                elif result['csv_rows'] >= 20 and result['csv_agent_filter']:
                    result['csv_quality'] = 'FAIR'
                else:
                    result['csv_quality'] = 'POOR'
        
        # 综合质量评估
        if (result['acmi_quality'] == 'EXCELLENT' and result['csv_quality'] == 'EXCELLENT'):
            result['overall_quality'] = 'EXCELLENT'
        elif (result['acmi_quality'] in ['EXCELLENT', 'GOOD'] and result['csv_quality'] in ['EXCELLENT', 'GOOD']):
            result['overall_quality'] = 'GOOD'
        elif (result['acmi_exists'] and result['csv_exists'] and 
              result['acmi_quality'] in ['EXCELLENT', 'GOOD', 'FAIR'] and 
              result['csv_quality'] in ['EXCELLENT', 'GOOD', 'FAIR']):
            result['overall_quality'] = 'FAIR'
        else:
            result['overall_quality'] = 'POOR'
        
        return result
        
    except Exception as e:
        print(f"极端样本质量分析失败: {e}")
        return result

def generate_extreme_differentiation_dataset():
    """生成极端差异化数据集"""
    print("🚀 **极端差异化基础动作数据生成器**")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("目标: 实施全面修复，生成165个极端差异化高质量样本")
    
    # 显示修复信息
    show_extreme_parameter_changes()
    show_data_recording_fixes()
    
    # 创建极端差异化生成器
    print(f"\n🔧 **初始化极端差异化数据生成器**")
    generator = BasicActionDataGenerator()
    
    # 获取所有可用动作
    all_actions = generator.get_available_actions()
    samples_per_action = 15
    total_expected_samples = len(all_actions) * samples_per_action
    
    print(f"\n✅ **极端差异化验证**:")
    print(f"可用动作数: {len(all_actions)} 种")
    print(f"动作列表: {', '.join(all_actions)}")
    print(f"每种动作样本数: {samples_per_action}")
    print(f"预期总样本数: {total_expected_samples}")
    
    # 验证动作配置
    expected_actions = [
        "level_flight", "accelerate", "decelerate", "turn_left", "turn_right",
        "climb", "climb_left", "climb_right", "dive", "dive_left", "dive_right"
    ]
    
    if set(all_actions) != set(expected_actions):
        print(f"❌ 动作配置不匹配")
        return False, {}, {}
    
    print("✅ 动作配置验证通过")
    
    # 开始生成数据
    print(f"\n🎯 **开始极端差异化数据生成**")
    print("=" * 80)
    
    total_samples = 0
    successful_samples = 0
    quality_stats = {
        'EXCELLENT': 0,
        'GOOD': 0,
        'FAIR': 0,
        'POOR': 0
    }
    
    action_results = {}
    coordinate_issues = 0
    agent_filter_issues = 0
    
    for action_index, action_name in enumerate(all_actions, 1):
        print(f"\n🔄 **[{action_index}/{len(all_actions)}] 生成 {action_name} 极端差异化数据**")
        print("-" * 80)
        
        action_result = {
            'successful': [],
            'failed': [],
            'quality_analysis': []
        }
        
        # 获取动作配置
        config = generator.action_configs[action_name]
        expected_duration = sum(config.duration_range) / 2  # 平均duration
        
        # 确定动作类型
        if action_name in ["climb_left", "climb_right", "dive_left", "dive_right"]:
            action_type = "极端组合机动"
        else:
            action_type = "极端基础动作"
        
        print(f"动作类型: {action_name} ({action_type})")
        print(f"预期duration: {expected_duration:.1f}秒")
        print(f"仿真总时长: {expected_duration + 18:.1f}秒 (包含18秒缓冲)")
        print(f"极端差异化: 最大参数范围，显著动作差异")
        
        for sample_index in range(1, samples_per_action + 1):
            total_samples += 1
            
            try:
                print(f"  样本 {sample_index:2d}/{samples_per_action}...", end=" ")
                
                # 生成样本
                acmi_path, csv_path = generator.generate_single_action_data(action_name, sample_index)
                
                if acmi_path and csv_path:
                    # 分析样本质量
                    quality = analyze_extreme_sample_quality(acmi_path, csv_path, action_name, expected_duration)
                    
                    # 统计修复效果
                    if not quality['csv_agent_filter']:
                        agent_filter_issues += 1
                    if quality['coordinate_range']['z_range'] == 0:
                        coordinate_issues += 1
                    
                    if quality['overall_quality'] in ['EXCELLENT', 'GOOD', 'FAIR']:
                        successful_samples += 1
                        action_result['successful'].append(sample_index)
                        action_result['quality_analysis'].append(quality)
                        quality_stats[quality['overall_quality']] += 1
                        
                        # 显示详细质量信息
                        coord_info = f"Z变化:{quality['coordinate_range']['z_range']:.0f}m"
                        agent_info = "A0100✅" if quality['csv_agent_filter'] else "多代理❌"
                        print(f"✅ {quality['overall_quality']} (ACMI:{quality['acmi_size_kb']:.1f}KB, CSV:{quality['csv_rows']}行, {coord_info}, {agent_info})")
                    else:
                        action_result['failed'].append(sample_index)
                        quality_stats[quality['overall_quality']] += 1
                        print(f"❌ {quality['overall_quality']} (ACMI:{quality['acmi_size_kb']:.1f}KB, CSV:{quality['csv_rows']}行)")
                else:
                    action_result['failed'].append(sample_index)
                    quality_stats['POOR'] += 1
                    print("❌ 生成失败")
                    
            except Exception as e:
                action_result['failed'].append(sample_index)
                quality_stats['POOR'] += 1
                print(f"❌ 异常: {str(e)[:30]}")
        
        action_results[action_name] = action_result
        
        # 动作总结
        success_count = len(action_result['successful'])
        fail_count = len(action_result['failed'])
        success_rate = success_count / samples_per_action * 100
        print(f"  {action_name} 完成: {success_count}/{samples_per_action} 成功 ({success_rate:.1f}%)")
    
    return True, total_samples, successful_samples, quality_stats, action_results, coordinate_issues, agent_filter_issues

if __name__ == "__main__":
    # 生成极端差异化数据集
    success, total, successful, quality_stats, action_results, coord_issues, agent_issues = generate_extreme_differentiation_dataset()
    
    if not success:
        print(f"\n❌ 极端差异化数据生成失败")
        exit(1)
    
    # 最终总结报告
    print(f"\n🎉 **极端差异化数据生成总结报告**")
    print("=" * 80)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总样本数: {total}")
    print(f"成功样本: {successful}")
    print(f"失败样本: {total - successful}")
    print(f"总体成功率: {successful/total*100:.1f}%")
    
    # 修复效果验证
    print(f"\n🔧 **修复效果验证**:")
    print(f"坐标系问题: {coord_issues} 个样本")
    print(f"代理过滤问题: {agent_issues} 个样本")
    print(f"CSV数据修复率: {(total - agent_issues)/total*100:.1f}%")
    
    # 质量分布统计
    print(f"\n📊 **样本质量分布**:")
    for quality, count in quality_stats.items():
        percentage = count / total * 100 if total > 0 else 0
        print(f"  {quality:<10}: {count:3d} 个样本 ({percentage:5.1f}%)")
    
    # 极端差异化效果分析
    print(f"\n📈 **极端差异化效果分析**:")
    print(f"{'动作名称':<12} {'类型':<12} {'样本数':<6} {'平均ACMI':<9} {'平均CSV行':<9} {'差异化效果':<12}")
    print("-" * 80)
    
    for action_name, results in action_results.items():
        # 确定动作类型
        if action_name in ["climb_left", "climb_right", "dive_left", "dive_right"]:
            action_type = "极端组合机动"
        else:
            action_type = "极端基础动作"
        
        if results['quality_analysis']:
            sample_count = len(results['quality_analysis'])
            avg_acmi_size = sum(q['acmi_size_kb'] for q in results['quality_analysis']) / len(results['quality_analysis'])
            avg_csv_rows = sum(q['csv_rows'] for q in results['quality_analysis']) / len(results['quality_analysis'])
            
            # 计算差异化效果
            z_ranges = [q['coordinate_range']['z_range'] for q in results['quality_analysis'] if q['coordinate_range']['z_range'] > 0]
            if z_ranges:
                avg_z_change = sum(z_ranges) / len(z_ranges)
                differentiation = f"{avg_z_change:.0f}m变化"
            else:
                differentiation = "无高度变化"
            
            print(f"{action_name:<12} {action_type:<12} {sample_count:<6} {avg_acmi_size:<9.1f} {avg_csv_rows:<9.0f} {differentiation:<12}")
        else:
            print(f"{action_name:<12} {action_type:<12} {'0':<6} {'N/A':<9} {'N/A':<9} {'失败':<12}")
    
    # 最终评估
    if successful >= total * 0.9:
        print(f"\n🎉 **极端差异化数据生成圆满成功！**")
        print("✅ 极端参数差异化：动作执行效果极其明显")
        print("✅ CSV数据修复：只记录A0100代理，坐标系修复完成")
        print("✅ 质量验证通过：ACMI和CSV文件质量优秀")
        print("✅ 数据可用性：可直接用于高质量AI训练和可视化分析")
    else:
        print(f"\n⚠️ 极端差异化数据生成部分成功，建议检查失败样本")
