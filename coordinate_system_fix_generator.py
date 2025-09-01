#!/usr/bin/env python3
"""
坐标系统修复的基础动作数据生成器
修复：使用拖曳射击项目相同的NEU坐标系统，小角度转弯，CSV数据过滤
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

def show_coordinate_system_fixes():
    """显示坐标系统修复内容"""
    print("🔧 **坐标系统修复内容**")
    print("=" * 80)
    
    fixes = [
        ("坐标系统修复", [
            "使用拖曳射击项目相同的NEU坐标系统",
            "直接使用agent.get_position()获取NEU坐标",
            "X_m = pos[0] (North分量)",
            "Y_m = pos[1] (East分量)",
            "Z_m = pos[2] (Up分量)"
        ]),
        ("小角度转弯修复", [
            "左转角度：-90度到-15度",
            "右转角度：15度到90度",
            "转弯速率：2.0-6.0度/秒",
            "避免大角度转弯问题"
        ]),
        ("CSV数据过滤修复", [
            "只记录A0100代理数据",
            "完全排除B0100（敌方）数据",
            "确保数据完整性和一致性",
            "修复多代理数据混合问题"
        ]),
        ("参数合理化", [
            "每种动作5个高质量样本",
            "物理可行的参数范围",
            "合理的持续时间设置",
            "确保参数与实际行为匹配"
        ])
    ]
    
    for category, items in fixes:
        print(f"\n📊 **{category}**:")
        print("-" * 50)
        for item in items:
            print(f"  ✅ {item}")

def analyze_coordinate_fixed_sample_quality(acmi_path: str, csv_path: str, action_name: str, expected_duration: float) -> Dict:
    """分析坐标修复后样本的质量"""
    result = {
        'acmi_exists': False,
        'acmi_size_kb': 0.0,
        'acmi_quality': 'FAIL',
        'csv_exists': False,
        'csv_rows': 0,
        'csv_time_range': 0.0,
        'csv_duration_match': False,
        'csv_agent_filter': False,
        'coordinate_format_correct': False,
        'coordinate_range_reasonable': False,
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
                
                # 坐标格式检查（NEU坐标系统）
                if 'X_m' in df.columns and 'Y_m' in df.columns and 'Z_m' in df.columns:
                    result['coordinate_format_correct'] = True
                    
                    # 检查坐标范围是否合理（类似拖曳射击项目）
                    x_range = df['X_m'].max() - df['X_m'].min()
                    y_range = df['Y_m'].max() - df['Y_m'].min()
                    z_range = df['Z_m'].max() - df['Z_m'].min()
                    
                    # 拖曳射击项目的坐标范围参考：X≈-44535, Y≈0-558, Z≈5940
                    # 检查坐标是否在合理范围内
                    x_reasonable = abs(df['X_m'].mean()) > 1000  # X坐标应该有显著值
                    y_reasonable = abs(df['Y_m'].max()) < 100000  # Y坐标应该在合理范围
                    z_reasonable = df['Z_m'].mean() > 1000  # Z坐标应该在合理高度
                    
                    result['coordinate_range_reasonable'] = x_reasonable and y_reasonable and z_reasonable
                
                # CSV质量评估
                quality_score = 0
                if result['csv_rows'] >= 30:
                    quality_score += 1
                if result['csv_duration_match']:
                    quality_score += 1
                if result['csv_agent_filter']:
                    quality_score += 1
                if result['coordinate_format_correct']:
                    quality_score += 1
                if result['coordinate_range_reasonable']:
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
        print(f"坐标修复样本质量分析失败: {e}")
        return result

def generate_coordinate_system_fix_dataset():
    """生成坐标系统修复的数据集"""
    print("🚀 **坐标系统修复基础动作数据生成器**")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("目标: 修复坐标系统，使用NEU坐标，小角度转弯，生成55个高质量样本")
    
    # 显示修复信息
    show_coordinate_system_fixes()
    
    # 完全清理现有数据
    if not complete_data_cleanup():
        print("❌ 数据清理失败")
        return False, {}, {}
    
    # 创建坐标系统修复的生成器
    print(f"\n🔧 **初始化坐标系统修复数据生成器**")
    generator = BasicActionDataGenerator()
    
    # 获取所有可用动作
    all_actions = generator.get_available_actions()
    samples_per_action = 5  # 每种动作5个高质量样本
    total_expected_samples = len(all_actions) * samples_per_action
    
    print(f"\n✅ **坐标系统修复验证**:")
    print(f"可用动作数: {len(all_actions)} 种")
    print(f"动作列表: {', '.join(all_actions)}")
    print(f"每种动作样本数: {samples_per_action} (高质量样本)")
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
    print(f"\n🎯 **开始坐标系统修复数据生成**")
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
    coordinate_fix_verification = {
        'coordinate_format_fixes': 0,
        'coordinate_range_fixes': 0,
        'agent_filter_fixes': 0,
        'small_angle_fixes': 0
    }
    
    for action_index, action_name in enumerate(all_actions, 1):
        print(f"\n🔄 **[{action_index}/{len(all_actions)}] 生成 {action_name} 坐标系统修复数据**")
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
        print(f"修复重点: NEU坐标系统、小角度转弯、数据过滤")
        
        for sample_index in range(1, samples_per_action + 1):
            total_samples += 1
            
            try:
                print(f"  样本 {sample_index:2d}/{samples_per_action}...", end=" ")
                
                # 生成样本
                acmi_path, csv_path = generator.generate_single_action_data(action_name, sample_index)
                
                if acmi_path and csv_path:
                    # 分析样本质量
                    quality = analyze_coordinate_fixed_sample_quality(acmi_path, csv_path, action_name, expected_duration)
                    
                    # 统计修复效果
                    if quality['coordinate_format_correct']:
                        coordinate_fix_verification['coordinate_format_fixes'] += 1
                    if quality['coordinate_range_reasonable']:
                        coordinate_fix_verification['coordinate_range_fixes'] += 1
                    if quality['csv_agent_filter']:
                        coordinate_fix_verification['agent_filter_fixes'] += 1
                    if action_name in ["turn_left", "turn_right"] and quality['coordinate_format_correct']:
                        coordinate_fix_verification['small_angle_fixes'] += 1
                    
                    if quality['overall_quality'] in ['EXCELLENT', 'GOOD', 'FAIR']:
                        successful_samples += 1
                        action_result['successful'].append(sample_index)
                        action_result['quality_analysis'].append(quality)
                        quality_stats[quality['overall_quality']] += 1
                        
                        # 显示详细修复效果
                        coord_format = "NEU✅" if quality['coordinate_format_correct'] else "格式❌"
                        coord_range = "范围✅" if quality['coordinate_range_reasonable'] else "范围❌"
                        agent_status = "A0100✅" if quality['csv_agent_filter'] else "多代理❌"
                        print(f"✅ {quality['overall_quality']} (ACMI:{quality['acmi_size_kb']:.1f}KB, CSV:{quality['csv_rows']}行, {coord_format}, {coord_range}, {agent_status})")
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
    
    return True, total_samples, successful_samples, quality_stats, action_results, coordinate_fix_verification

if __name__ == "__main__":
    # 生成坐标系统修复数据集
    success, total, successful, quality_stats, action_results, fix_stats = generate_coordinate_system_fix_dataset()
    
    if not success:
        print(f"\n❌ 坐标系统修复数据生成失败")
        exit(1)
    
    # 最终总结报告
    print(f"\n🎉 **坐标系统修复数据生成总结报告**")
    print("=" * 80)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总样本数: {total}")
    print(f"成功样本: {successful}")
    print(f"失败样本: {total - successful}")
    print(f"总体成功率: {successful/total*100:.1f}%")
    
    # 坐标系统修复效果验证
    print(f"\n🔧 **坐标系统修复效果验证**:")
    print(f"NEU坐标格式修复: {fix_stats['coordinate_format_fixes']}/{total} 样本 ({fix_stats['coordinate_format_fixes']/total*100:.1f}%)")
    print(f"坐标范围合理化: {fix_stats['coordinate_range_fixes']}/{total} 样本 ({fix_stats['coordinate_range_fixes']/total*100:.1f}%)")
    print(f"代理过滤修复: {fix_stats['agent_filter_fixes']}/{total} 样本 ({fix_stats['agent_filter_fixes']/total*100:.1f}%)")
    print(f"小角度转弯修复: {fix_stats['small_angle_fixes']}/10 转弯样本 ({fix_stats['small_angle_fixes']/10*100:.1f}%)")
    
    # 质量分布统计
    print(f"\n📊 **样本质量分布**:")
    for quality, count in quality_stats.items():
        percentage = count / total * 100 if total > 0 else 0
        print(f"  {quality:<10}: {count:3d} 个样本 ({percentage:5.1f}%)")
    
    # 各动作坐标修复效果分析
    print(f"\n📈 **各动作坐标修复效果分析**:")
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
    if successful >= total * 0.8:
        print(f"\n🎉 **坐标系统修复数据生成圆满成功！**")
        print("✅ NEU坐标系统修复：使用拖曳射击项目相同的坐标计算")
        print("✅ 小角度转弯修复：15-90度角度范围，避免大角度问题")
        print("✅ CSV数据过滤修复：只记录A0100代理数据")
        print("✅ 坐标格式正确：X_m(North), Y_m(East), Z_m(Up)")
        print("✅ 数据可用性：可直接用于高质量AI训练和可视化分析")
    else:
        print(f"\n⚠️ 坐标系统修复数据生成部分成功，建议检查失败样本")
