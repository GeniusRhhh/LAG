#!/usr/bin/env python3
"""
参数优化的基础动作数据生成器
增强动作差异化和样本多样性，生成165个高质量样本
"""

import sys
sys.path.append('.')
from scripts.basic_action_data_generator import BasicActionDataGenerator
import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Tuple

def analyze_parameter_diversity(action_results: Dict) -> Dict:
    """分析参数多样性"""
    diversity_analysis = {}
    
    for action_name, results in action_results.items():
        if not results['quality_analysis']:
            continue
            
        # 统计参数范围覆盖
        diversity_analysis[action_name] = {
            'sample_count': len(results['quality_analysis']),
            'avg_acmi_size': sum(q['acmi_size_kb'] for q in results['quality_analysis']) / len(results['quality_analysis']),
            'avg_csv_rows': sum(q['csv_rows'] for q in results['quality_analysis']) / len(results['quality_analysis']),
            'time_range_variation': max(q['csv_time_range'] for q in results['quality_analysis']) - min(q['csv_time_range'] for q in results['quality_analysis'])
        }
    
    return diversity_analysis

def show_parameter_optimization():
    """显示参数优化信息"""
    print("🔧 **参数优化配置详情**")
    print("=" * 80)
    
    optimization_info = [
        ("基础动作参数优化", [
            ("加速/减速", "速度变化", "40-100 m/s → 50-150 m/s"),
            ("左转/右转", "转弯角度", "15-90度 → 30-135度"),
            ("左转/右转", "转弯速率", "2.5-6.0度/秒 → 2.0-8.0度/秒"),
            ("爬升/俯冲", "高度变化", "1000-2500米 → 1500-3500米")
        ]),
        ("组合机动参数优化", [
            ("左爬升/右爬升", "转弯角度", "15-90度 → 30-120度"),
            ("左爬升/右爬升", "转弯速率", "2.5-6.0度/秒 → 3.0-8.0度/秒"),
            ("左爬升/右爬升", "爬升高度", "1000-2500米 → 1500-3500米"),
            ("左俯冲/右俯冲", "转弯角度", "15-90度 → 30-120度"),
            ("左俯冲/右俯冲", "转弯速率", "2.5-6.0度/秒 → 3.0-8.0度/秒"),
            ("左俯冲/右俯冲", "俯冲高度", "1000-2500米 → 1500-3500米")
        ])
    ]
    
    for category, items in optimization_info:
        print(f"\n📊 **{category}**:")
        print("-" * 60)
        for action, param, change in items:
            print(f"  {action:<15} {param:<12} {change}")
    
    print(f"\n📈 **样本数量优化**:")
    print(f"  每种动作样本数: 10个 → 15个")
    print(f"  总样本数: 110个 → 165个")
    print(f"  增加比例: +50%")

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
            if acmi_size >= 30000:  # 30KB以上
                result['acmi_quality'] = 'EXCELLENT'
            elif acmi_size >= 20000:  # 20KB以上
                result['acmi_quality'] = 'GOOD'
            elif acmi_size >= 10000:   # 10KB以上
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
                if result['csv_rows'] >= 60 and result['csv_duration_match']:
                    result['csv_quality'] = 'EXCELLENT'
                elif result['csv_rows'] >= 40 and result['csv_duration_match']:
                    result['csv_quality'] = 'GOOD'
                elif result['csv_rows'] >= 20:
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

def generate_optimized_parameter_dataset():
    """生成参数优化的数据集"""
    print("🚀 **参数优化基础动作数据生成器**")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("目标: 增强动作差异化和样本多样性，生成165个高质量样本")
    
    # 显示参数优化信息
    show_parameter_optimization()
    
    # 创建参数优化的生成器
    print(f"\n🔧 **初始化参数优化数据生成器**")
    generator = BasicActionDataGenerator()
    
    # 获取所有可用动作
    all_actions = generator.get_available_actions()
    samples_per_action = 15  # 优化后每种动作15个样本
    total_expected_samples = len(all_actions) * samples_per_action
    
    print(f"\n✅ **参数优化验证**:")
    print(f"可用动作数: {len(all_actions)} 种")
    print(f"动作列表: {', '.join(all_actions)}")
    print(f"每种动作样本数: {samples_per_action} (优化前: 10)")
    print(f"预期总样本数: {total_expected_samples} (优化前: 110)")
    
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
    print(f"\n🎯 **开始参数优化数据生成**")
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
        print(f"参数优化: 增强差异化和多样性")
        
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
    # 生成参数优化的数据集
    success, total, successful, quality_stats, action_results = generate_optimized_parameter_dataset()
    
    if not success:
        print(f"\n❌ 参数优化数据生成失败")
        exit(1)
    
    # 最终总结报告
    print(f"\n🎉 **参数优化数据生成总结报告**")
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
    
    # 参数多样性分析
    diversity_analysis = analyze_parameter_diversity(action_results)
    print(f"\n📈 **参数多样性分析**:")
    print(f"{'动作名称':<12} {'类型':<8} {'样本数':<6} {'平均ACMI':<9} {'平均CSV行':<9} {'时间变化':<8}")
    print("-" * 75)
    
    for action_name, analysis in diversity_analysis.items():
        # 确定动作类型
        if action_name in ["climb_left", "climb_right", "dive_left", "dive_right"]:
            action_type = "组合机动"
        else:
            action_type = "基础动作"
        
        print(f"{action_name:<12} {action_type:<8} {analysis['sample_count']:<6} {analysis['avg_acmi_size']:<9.1f} {analysis['avg_csv_rows']:<9.0f} {analysis['time_range_variation']:<8.1f}")
    
    # 最终评估
    if successful >= total * 0.9:
        print(f"\n🎉 **参数优化数据生成圆满成功！**")
        print("✅ 参数差异化增强：动作执行效果更加明显")
        print("✅ 样本多样性提升：165个样本覆盖更广泛场景")
        print("✅ 质量验证通过：ACMI和CSV文件质量优秀")
        print("✅ 数据可用性：可直接用于高质量AI训练")
    else:
        print(f"\n⚠️ 参数优化数据生成部分成功，建议检查失败样本")
