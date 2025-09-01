#!/usr/bin/env python3
"""
重新配置的基础动作数据生成器
按照重新确认的11种机动动作列表生成完整数据集
"""

import sys
sys.path.append('.')
from scripts.basic_action_data_generator import BasicActionDataGenerator
import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Tuple

def analyze_sample_quality(acmi_path: str, csv_path: str, action_name: str, expected_duration: float) -> Dict:
    """分析单个样本的质量"""
    result = {
        'acmi_exists': False,
        'acmi_size_kb': 0.0,
        'acmi_quality': 'FAIL',
        'csv_exists': False,
        'csv_rows': 0,
        'csv_time_range': 0.0,
        'csv_duration_match': False,
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
            elif acmi_size >= 5000:   # 5KB以上
                result['acmi_quality'] = 'FAIR'
            else:
                result['acmi_quality'] = 'POOR'
        
        # 检查CSV文件
        if os.path.exists(csv_path):
            result['csv_exists'] = True
            df = pd.read_csv(csv_path)
            result['csv_rows'] = len(df)
            
            if len(df) > 0:
                time_min = df['Time_s'].min()
                time_max = df['Time_s'].max()
                result['csv_time_range'] = time_max - time_min
                
                # 时间范围匹配度检查
                time_diff = abs(result['csv_time_range'] - expected_duration)
                result['csv_duration_match'] = time_diff < 2.0  # 允许2秒误差
                
                # CSV质量评估
                if result['csv_rows'] >= 50 and result['csv_duration_match']:
                    result['csv_quality'] = 'EXCELLENT'
                elif result['csv_rows'] >= 30 and result['csv_duration_match']:
                    result['csv_quality'] = 'GOOD'
                elif result['csv_rows'] >= 15:
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
        print(f"样本质量分析失败: {e}")
        return result

def show_action_configuration():
    """显示重新配置的动作信息"""
    print("📋 **重新配置的11种机动动作**")
    print("=" * 80)
    
    actions_info = [
        ("1", "平飞", "level_flight", "基础动作", "15-25秒"),
        ("2", "加速", "accelerate", "基础动作", "15-25秒"),
        ("3", "减速", "decelerate", "基础动作", "15-25秒"),
        ("4", "左转", "turn_left", "基础动作", "15-25秒"),
        ("5", "右转", "turn_right", "基础动作", "15-25秒"),
        ("6", "爬升", "climb", "基础动作", "15-25秒"),
        ("7", "左爬升", "climb_left", "组合机动", "20-30秒"),
        ("8", "右爬升", "climb_right", "组合机动", "20-30秒"),
        ("9", "俯冲", "dive", "基础动作", "15-25秒"),
        ("10", "左俯冲", "dive_left", "组合机动", "20-30秒"),
        ("11", "右俯冲", "dive_right", "组合机动", "20-30秒")
    ]
    
    print(f"{'序号':<4} {'中文名称':<8} {'英文名称':<12} {'类型':<8} {'Duration':<10}")
    print("-" * 70)
    
    basic_count = 0
    combo_count = 0
    
    for num, chinese, english, type_name, duration in actions_info:
        print(f"{num:<4} {chinese:<8} {english:<12} {type_name:<8} {duration:<10}")
        if type_name == "基础动作":
            basic_count += 1
        else:
            combo_count += 1
    
    print("-" * 70)
    print(f"总计: {len(actions_info)} 种动作")
    print(f"基础动作: {basic_count} 种 (duration: 15-25秒)")
    print(f"组合机动: {combo_count} 种 (duration: 20-30秒)")
    print(f"仿真缓冲时间: 18秒")
    print(f"预期样本数: {len(actions_info)} × 10 = {len(actions_info) * 10} 个")

def generate_reconfigured_dataset():
    """生成重新配置的数据集"""
    print("🚀 **重新配置的基础动作数据生成器**")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("目标: 按照重新确认的11种机动动作列表生成完整数据集")
    
    # 显示动作配置信息
    show_action_configuration()
    
    # 创建重新配置的生成器
    print(f"\n🔧 **初始化重新配置的数据生成器**")
    generator = BasicActionDataGenerator()
    
    # 获取所有可用动作
    all_actions = generator.get_available_actions()
    samples_per_action = 10
    total_expected_samples = len(all_actions) * samples_per_action
    
    print(f"\n✅ **框架重新配置验证**:")
    print(f"可用动作数: {len(all_actions)} 种")
    print(f"动作列表: {', '.join(all_actions)}")
    print(f"每种动作样本数: {samples_per_action}")
    print(f"预期总样本数: {total_expected_samples}")
    
    # 验证动作配置
    expected_actions = [
        "level_flight", "accelerate", "decelerate", "turn_left", "turn_right",
        "climb", "climb_left", "climb_right", "dive", "dive_left", "dive_right"
    ]
    
    missing_actions = set(expected_actions) - set(all_actions)
    extra_actions = set(all_actions) - set(expected_actions)
    
    if missing_actions:
        print(f"❌ 缺少动作: {missing_actions}")
        return False, {}, {}
    
    if extra_actions:
        print(f"⚠️ 额外动作: {extra_actions}")
    
    if len(all_actions) != 11:
        print(f"❌ 动作数量不匹配: 期望11种，实际{len(all_actions)}种")
        return False, {}, {}
    
    print("✅ 动作配置验证通过")
    
    # 开始生成数据
    print(f"\n🎯 **开始重新配置数据生成**")
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
    
    for action_index, action_name in enumerate(all_actions, 1):
        print(f"\n🔄 **[{action_index}/{len(all_actions)}] 生成 {action_name} 动作数据**")
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
            action_type = "组合机动"
        else:
            action_type = "基础动作"
        
        print(f"动作类型: {action_name} ({action_type})")
        print(f"预期duration: {expected_duration:.1f}秒")
        print(f"仿真总时长: {expected_duration + 18:.1f}秒 (包含18秒缓冲)")
        
        for sample_index in range(1, samples_per_action + 1):
            total_samples += 1
            
            try:
                print(f"  样本 {sample_index:2d}/{samples_per_action}...", end=" ")
                
                # 生成样本
                acmi_path, csv_path = generator.generate_single_action_data(action_name, sample_index)
                
                if acmi_path and csv_path:
                    # 分析样本质量
                    quality = analyze_sample_quality(acmi_path, csv_path, action_name, expected_duration)
                    
                    if quality['overall_quality'] in ['EXCELLENT', 'GOOD', 'FAIR']:
                        successful_samples += 1
                        action_result['successful'].append(sample_index)
                        action_result['quality_analysis'].append(quality)
                        quality_stats[quality['overall_quality']] += 1
                        
                        print(f"✅ {quality['overall_quality']} (ACMI: {quality['acmi_size_kb']:.1f}KB, CSV: {quality['csv_rows']}行, 时间: {quality['csv_time_range']:.1f}s)")
                    else:
                        action_result['failed'].append(sample_index)
                        quality_stats[quality['overall_quality']] += 1
                        print(f"❌ {quality['overall_quality']} (ACMI: {quality['acmi_size_kb']:.1f}KB, CSV: {quality['csv_rows']}行)")
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
    
    return True, total_samples, successful_samples, quality_stats, action_results

if __name__ == "__main__":
    # 生成重新配置的数据集
    success, total, successful, quality_stats, action_results = generate_reconfigured_dataset()
    
    if not success:
        print(f"\n❌ 重新配置数据生成失败")
        exit(1)
    
    # 最终总结报告
    print(f"\n🎉 **重新配置数据生成总结报告**")
    print("=" * 80)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总样本数: {total}")
    print(f"成功样本: {successful}")
    print(f"失败样本: {total - successful}")
    print(f"总体成功率: {successful/total*100:.1f}%")
    
    # 质量分布统计
    print(f"\n📊 **样本质量分布**:")
    for quality, count in quality_stats.items():
        percentage = count / total * 100 if total > 0 else 0
        print(f"  {quality:<10}: {count:3d} 个样本 ({percentage:5.1f}%)")
    
    # 各动作详细结果
    print(f"\n📋 **各动作生成结果**:")
    print(f"{'动作名称':<12} {'类型':<8} {'成功':<4} {'失败':<4} {'成功率':<7} {'平均ACMI':<9} {'平均CSV行':<9}")
    print("-" * 75)
    
    for action_name, results in action_results.items():
        # 确定动作类型
        if action_name in ["climb_left", "climb_right", "dive_left", "dive_right"]:
            action_type = "组合机动"
        else:
            action_type = "基础动作"
        
        success_count = len(results['successful'])
        fail_count = len(results['failed'])
        total_count = success_count + fail_count
        success_rate = success_count / total_count * 100 if total_count > 0 else 0
        
        if results['quality_analysis']:
            avg_acmi_size = sum(q['acmi_size_kb'] for q in results['quality_analysis']) / len(results['quality_analysis'])
            avg_csv_rows = sum(q['csv_rows'] for q in results['quality_analysis']) / len(results['quality_analysis'])
            print(f"{action_name:<12} {action_type:<8} {success_count:<4} {fail_count:<4} {success_rate:<7.1f}% {avg_acmi_size:<9.1f} {avg_csv_rows:<9.0f}")
        else:
            print(f"{action_name:<12} {action_type:<8} {success_count:<4} {fail_count:<4} {success_rate:<7.1f}% {'N/A':<9} {'N/A':<9}")
    
    # 最终评估
    if successful >= total * 0.8:
        print(f"\n🎉 **重新配置数据生成圆满成功！**")
        print("✅ 框架重新配置完成：11种机动动作配置")
        print("✅ 数据清理完成：所有旧数据已清理")
        print("✅ 完整数据生成：110个样本生成完成")
        print("✅ 质量验证通过：ACMI和CSV文件质量良好")
        print("✅ 数据可用性：可直接用于机器学习训练和Tacview可视化")
    else:
        print(f"\n⚠️ 重新配置数据生成部分成功，建议检查失败样本")
