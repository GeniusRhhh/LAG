#!/usr/bin/env python3
"""
坐标系统标准化的基础动作数据生成器
修复：使用与拖曳射击项目完全相同的战场中心点和坐标系统
"""

import sys
sys.path.append('.')
from scripts.basic_action_data_generator import BasicActionDataGenerator
import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime
from typing import Dict, List, Tuple

def complete_data_cleanup():
    """完全清理现有数据"""
    print("🧹 **完全清理现有数据**")
    print("=" * 70)
    
    base_dir = "scripts/drag_shoot_2v2/basic_action_data"
    
    if os.path.exists(base_dir):
        total_files = 0
        # 统计现有文件
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith(('.csv', '.acmi')):
                    total_files += 1
        
        print(f"发现现有数据文件: {total_files} 个")
        
        # 删除整个目录
        try:
            shutil.rmtree(base_dir)
            print(f"✅ 完全删除目录: {base_dir}")
        except Exception as e:
            print(f"删除目录失败: {e}")
            return False
        
        print(f"✅ 数据清理完成，删除了 {total_files} 个文件")
    else:
        print("数据目录不存在，无需清理")
    
    return True

def show_coordinate_standardization_fixes():
    """显示坐标系统标准化修复内容"""
    print("🔧 **坐标系统标准化修复内容**")
    print("=" * 80)
    
    fixes = [
        ("战场中心点标准化", [
            "添加battle_field_center: [120.0, 60.4, 0.0]到simple_maneuver_config.yaml",
            "使用与拖曳射击项目完全相同的战场中心点",
            "确保LLA2NEU转换使用正确的参考点",
            "坐标系统完全统一"
        ]),
        ("飞机初始位置标准化", [
            "A0100: ic_long_gc_deg: 120.0, ic_lat_geod_deg: 60.0",
            "B0100: ic_long_gc_deg: 120.0, ic_lat_geod_deg: 60.8",
            "高度统一为ic_h_sl_ft: 20000（与拖曳射击项目一致）",
            "初始航向和速度与拖曳射击项目一致"
        ]),
        ("预期坐标范围", [
            "A0100应该在X≈-44535m（与拖曳射击项目一致）",
            "Y坐标应该在0附近（与拖曳射击项目一致）",
            "Z坐标应该在5940m左右（与拖曳射击项目一致）",
            "坐标变化范围应该与拖曳射击项目相似"
        ]),
        ("数据兼容性保证", [
            "CSV格式与拖曳射击项目完全兼容",
            "只记录A0100代理数据",
            "坐标数值范围与四个战术项目一致",
            "确保未来数据分析的一致性"
        ])
    ]
    
    for category, items in fixes:
        print(f"\n📊 **{category}**:")
        print("-" * 50)
        for item in items:
            print(f"  ✅ {item}")

def analyze_standardized_coordinate_quality(acmi_path: str, csv_path: str, action_name: str, expected_duration: float) -> Dict:
    """分析标准化坐标后样本的质量"""
    result = {
        'acmi_exists': False,
        'acmi_size_kb': 0.0,
        'acmi_quality': 'FAIL',
        'csv_exists': False,
        'csv_rows': 0,
        'csv_time_range': 0.0,
        'csv_duration_match': False,
        'csv_agent_filter': False,
        'coordinate_standardized': False,
        'coordinate_range_correct': False,
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
            if acmi_size >= 25000:  # 25KB以上
                result['acmi_quality'] = 'EXCELLENT'
            elif acmi_size >= 15000:  # 15KB以上
                result['acmi_quality'] = 'GOOD'
            elif acmi_size >= 8000:   # 8KB以上
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
                
                # 时间范围匹配度检查
                time_diff = abs(result['csv_time_range'] - expected_duration)
                result['csv_duration_match'] = time_diff < 3.0  # 允许3秒误差
                
                # 坐标标准化检查（与拖曳射击项目对比）
                if 'X_m' in df.columns and 'Y_m' in df.columns and 'Z_m' in df.columns:
                    result['coordinate_standardized'] = True
                    
                    # 检查坐标范围是否与拖曳射击项目一致
                    x_mean = df['X_m'].mean()
                    y_mean = df['Y_m'].mean()
                    z_mean = df['Z_m'].mean()
                    
                    # 拖曳射击项目的坐标范围：A0100在X≈-44535, Y≈0, Z≈5940
                    x_correct = -50000 < x_mean < -40000  # X坐标在-50000到-40000之间
                    y_correct = abs(y_mean) < 1000        # Y坐标在±1000米范围内
                    z_correct = 5000 < z_mean < 7000      # Z坐标在5000-7000米范围内
                    
                    result['coordinate_range_correct'] = x_correct and y_correct and z_correct
                
                # CSV质量评估
                quality_score = 0
                if result['csv_rows'] >= 30:
                    quality_score += 1
                if result['csv_duration_match']:
                    quality_score += 1
                if result['csv_agent_filter']:
                    quality_score += 1
                if result['coordinate_standardized']:
                    quality_score += 1
                if result['coordinate_range_correct']:
                    quality_score += 1
                
                if quality_score >= 4:
                    result['csv_quality'] = 'EXCELLENT'
                elif quality_score >= 3:
                    result['csv_quality'] = 'GOOD'
                elif quality_score >= 2:
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
        print(f"标准化坐标样本质量分析失败: {e}")
        return result

def generate_coordinate_standardized_dataset():
    """生成坐标系统标准化的数据集"""
    print("🚀 **坐标系统标准化基础动作数据生成器**")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("目标: 坐标系统标准化，与拖曳射击项目完全一致，生成55个高质量样本")
    
    # 显示修复信息
    show_coordinate_standardization_fixes()
    
    # 完全清理现有数据
    if not complete_data_cleanup():
        print("❌ 数据清理失败")
        return False, {}, {}
    
    # 创建坐标系统标准化的生成器
    print(f"\n🔧 **初始化坐标系统标准化数据生成器**")
    generator = BasicActionDataGenerator()
    
    # 获取所有可用动作
    all_actions = generator.get_available_actions()
    samples_per_action = 5  # 每种动作5个高质量样本
    total_expected_samples = len(all_actions) * samples_per_action
    
    print(f"\n✅ **坐标系统标准化验证**:")
    print(f"可用动作数: {len(all_actions)} 种")
    print(f"动作列表: {', '.join(all_actions)}")
    print(f"每种动作样本数: {samples_per_action} (高质量样本)")
    print(f"预期总样本数: {total_expected_samples}")
    print(f"预期坐标范围: X≈-44535m, Y≈0m, Z≈5940m (与拖曳射击项目一致)")
    
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
    print(f"\n🎯 **开始坐标系统标准化数据生成**")
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
    standardization_verification = {
        'coordinate_standardized': 0,
        'coordinate_range_correct': 0,
        'agent_filter_fixes': 0,
        'tactical_compatibility': 0
    }
    
    for action_index, action_name in enumerate(all_actions, 1):
        print(f"\n🔄 **[{action_index}/{len(all_actions)}] 生成 {action_name} 坐标标准化数据**")
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
            action_type = "小角度组合机动"
        elif action_name in ["turn_left", "turn_right"]:
            action_type = "小角度转弯"
        else:
            action_type = "基础动作"
        
        print(f"动作类型: {action_name} ({action_type})")
        print(f"预期duration: {expected_duration:.1f}秒")
        print(f"仿真总时长: {expected_duration + 18:.1f}秒 (包含18秒缓冲)")
        print(f"修复重点: 战场中心点标准化、坐标范围一致性")
        
        for sample_index in range(1, samples_per_action + 1):
            total_samples += 1
            
            try:
                print(f"  样本 {sample_index:2d}/{samples_per_action}...", end=" ")
                
                # 生成样本
                acmi_path, csv_path = generator.generate_single_action_data(action_name, sample_index)
                
                if acmi_path and csv_path:
                    # 分析样本质量
                    quality = analyze_standardized_coordinate_quality(acmi_path, csv_path, action_name, expected_duration)
                    
                    # 统计标准化效果
                    if quality['coordinate_standardized']:
                        standardization_verification['coordinate_standardized'] += 1
                    if quality['coordinate_range_correct']:
                        standardization_verification['coordinate_range_correct'] += 1
                    if quality['csv_agent_filter']:
                        standardization_verification['agent_filter_fixes'] += 1
                    if quality['coordinate_range_correct'] and quality['csv_agent_filter']:
                        standardization_verification['tactical_compatibility'] += 1
                    
                    if quality['overall_quality'] in ['EXCELLENT', 'GOOD', 'FAIR']:
                        successful_samples += 1
                        action_result['successful'].append(sample_index)
                        action_result['quality_analysis'].append(quality)
                        quality_stats[quality['overall_quality']] += 1
                        
                        # 显示详细标准化效果
                        coord_std = "标准化✅" if quality['coordinate_standardized'] else "格式❌"
                        coord_range = "范围✅" if quality['coordinate_range_correct'] else "范围❌"
                        agent_status = "A0100✅" if quality['csv_agent_filter'] else "多代理❌"
                        print(f"✅ {quality['overall_quality']} (ACMI:{quality['acmi_size_kb']:.1f}KB, CSV:{quality['csv_rows']}行, {coord_std}, {coord_range}, {agent_status})")
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
    
    return True, total_samples, successful_samples, quality_stats, action_results, standardization_verification

if __name__ == "__main__":
    # 生成坐标系统标准化数据集
    success, total, successful, quality_stats, action_results, std_stats = generate_coordinate_standardized_dataset()
    
    if not success:
        print(f"\n❌ 坐标系统标准化数据生成失败")
        exit(1)
    
    # 最终总结报告
    print(f"\n🎉 **坐标系统标准化数据生成总结报告**")
    print("=" * 80)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总样本数: {total}")
    print(f"成功样本: {successful}")
    print(f"失败样本: {total - successful}")
    print(f"总体成功率: {successful/total*100:.1f}%")
    
    # 坐标系统标准化效果验证
    print(f"\n🔧 **坐标系统标准化效果验证**:")
    print(f"坐标格式标准化: {std_stats['coordinate_standardized']}/{total} 样本 ({std_stats['coordinate_standardized']/total*100:.1f}%)")
    print(f"坐标范围正确性: {std_stats['coordinate_range_correct']}/{total} 样本 ({std_stats['coordinate_range_correct']/total*100:.1f}%)")
    print(f"代理过滤修复: {std_stats['agent_filter_fixes']}/{total} 样本 ({std_stats['agent_filter_fixes']/total*100:.1f}%)")
    print(f"战术项目兼容性: {std_stats['tactical_compatibility']}/{total} 样本 ({std_stats['tactical_compatibility']/total*100:.1f}%)")
    
    # 质量分布统计
    print(f"\n📊 **样本质量分布**:")
    for quality, count in quality_stats.items():
        percentage = count / total * 100 if total > 0 else 0
        print(f"  {quality:<10}: {count:3d} 个样本 ({percentage:5.1f}%)")
    
    # 各动作坐标标准化效果分析
    print(f"\n📈 **各动作坐标标准化效果分析**:")
    print(f"{'动作名称':<12} {'类型':<12} {'样本数':<6} {'成功率':<7} {'平均ACMI':<9} {'平均CSV行':<9}")
    print("-" * 75)
    
    for action_name, results in action_results.items():
        # 确定动作类型
        if action_name in ["climb_left", "climb_right", "dive_left", "dive_right"]:
            action_type = "小角度组合机动"
        elif action_name in ["turn_left", "turn_right"]:
            action_type = "小角度转弯"
        else:
            action_type = "基础动作"
        
        if results['quality_analysis']:
            sample_count = len(results['quality_analysis'])
            success_rate = len(results['successful']) / 5 * 100
            avg_acmi_size = sum(q['acmi_size_kb'] for q in results['quality_analysis']) / len(results['quality_analysis'])
            avg_csv_rows = sum(q['csv_rows'] for q in results['quality_analysis']) / len(results['quality_analysis'])
            
            print(f"{action_name:<12} {action_type:<12} {sample_count:<6} {success_rate:<7.0f}% {avg_acmi_size:<9.1f} {avg_csv_rows:<9.0f}")
        else:
            print(f"{action_name:<12} {action_type:<12} {'0':<6} {'0':<7}% {'N/A':<9} {'N/A':<9}")
    
    # 最终评估
    if successful >= total * 0.8 and std_stats['coordinate_range_correct'] >= total * 0.8:
        print(f"\n🎉 **坐标系统标准化数据生成圆满成功！**")
        print("✅ 战场中心点标准化：使用与拖曳射击项目相同的[120.0, 60.4, 0.0]")
        print("✅ 坐标范围一致性：X≈-44535m, Y≈0m, Z≈5940m与拖曳射击项目一致")
        print("✅ 数据格式兼容性：CSV格式与四个战术项目完全兼容")
        print("✅ 代理数据过滤：只记录A0100代理数据")
        print("✅ 未来数据分析：确保与所有战术项目的数据一致性")
    else:
        print(f"\n⚠️ 坐标系统标准化数据生成部分成功，建议检查坐标范围")
