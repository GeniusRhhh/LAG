#!/usr/bin/env python3
"""
完整基础和战术动作数据生成器
- 框架改进：18秒缓冲时间，优化duration参数范围
- 完整数据生成：12种动作，每种10个样本，总计120个样本
- 质量验证：ACMI文件和CSV数据完整性验证
"""

import sys
sys.path.append('.')
from scripts.basic_action_data_generator import BasicActionDataGenerator
import pandas as pd
import os
import shutil
from datetime import datetime
from typing import Dict, List, Tuple

def clean_all_existing_data(generator):
    """完全清理现有数据"""
    print("🧹 **完全清理现有数据**")
    print("=" * 70)
    
    base_dir = generator.output_dir
    if os.path.exists(base_dir):
        total_files = 0
        # 统计现有文件
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith(('.csv', '.acmi')):
                    total_files += 1
        
        print(f"发现现有数据文件: {total_files} 个")
        
        # 删除所有CSV和ACMI文件
        deleted_files = 0
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith(('.csv', '.acmi')):
                    try:
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
                        deleted_files += 1
                        if deleted_files % 10 == 0:
                            print(f"已删除 {deleted_files}/{total_files} 个文件...")
                    except Exception as e:
                        print(f"无法删除 {file}: {e}")
        
        print(f"✅ 数据清理完成，删除了 {deleted_files} 个文件")
    else:
        print("数据目录不存在，无需清理")
    
    # 重新创建目录结构
    print("\n📁 **重新创建目录结构**")
    generator.setup_output_directories()
    print("✅ 目录结构重新创建完成")

def analyze_file_quality(acmi_path: str, csv_path: str, action_name: str, expected_duration: float) -> Dict:
    """分析单个样本的文件质量"""
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
                result['acmi_quality'] = 'GOOD'
            elif acmi_size >= 10000:  # 10KB以上
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
                    result['csv_quality'] = 'GOOD'
                elif result['csv_rows'] >= 20:
                    result['csv_quality'] = 'FAIR'
                else:
                    result['csv_quality'] = 'POOR'
        
        # 综合质量评估
        if (result['acmi_quality'] == 'GOOD' and result['csv_quality'] == 'GOOD'):
            result['overall_quality'] = 'EXCELLENT'
        elif (result['acmi_quality'] in ['GOOD', 'FAIR'] and result['csv_quality'] in ['GOOD', 'FAIR']):
            result['overall_quality'] = 'GOOD'
        elif (result['acmi_exists'] and result['csv_exists']):
            result['overall_quality'] = 'FAIR'
        else:
            result['overall_quality'] = 'POOR'
        
        return result
        
    except Exception as e:
        print(f"文件质量分析失败: {e}")
        return result

def generate_complete_action_dataset():
    """生成完整的基础和战术动作数据集"""
    print("🚀 **完整基础和战术动作数据生成器**")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("目标: 生成12种动作，每种10个样本，总计120个高质量样本")
    
    # 创建改进版生成器
    print("\n🔧 **初始化改进版数据生成器**")
    generator = BasicActionDataGenerator()
    
    # 显示框架改进信息
    print("✅ 框架改进:")
    print("  - 仿真缓冲时间: 10秒 → 18秒")
    print("  - 基础动作duration: 15-25秒")
    print("  - 战术动作duration: 20-35秒")
    print("  - 每种动作样本数: 5个 → 10个")
    
    # 获取所有可用动作
    all_actions = generator.get_available_actions()
    samples_per_action = 10
    total_expected_samples = len(all_actions) * samples_per_action
    
    print(f"\n📋 **生成配置**:")
    print(f"动作类型: {len(all_actions)}种")
    print(f"动作列表: {', '.join(all_actions)}")
    print(f"每种动作样本数: {samples_per_action}")
    print(f"预期总样本数: {total_expected_samples}")
    
    # 完全清理现有数据
    clean_all_existing_data(generator)
    
    # 开始生成数据
    print(f"\n🎯 **开始完整数据生成**")
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
        
        print(f"动作类型: {action_name}")
        print(f"预期duration: {expected_duration:.1f}秒")
        print(f"仿真总时长: {expected_duration + 18:.1f}秒 (包含18秒缓冲)")
        
        for sample_index in range(1, samples_per_action + 1):
            total_samples += 1
            
            try:
                print(f"  样本 {sample_index:2d}/{samples_per_action}...", end=" ")
                
                # 生成样本
                acmi_path, csv_path = generator.generate_single_action_data(action_name, sample_index)
                
                if acmi_path and csv_path:
                    # 分析文件质量
                    quality = analyze_file_quality(acmi_path, csv_path, action_name, expected_duration)
                    
                    if quality['overall_quality'] in ['EXCELLENT', 'GOOD']:
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
    
    return total_samples, successful_samples, quality_stats, action_results

if __name__ == "__main__":
    # 生成完整数据集
    total, successful, quality_stats, action_results = generate_complete_action_dataset()
    
    # 最终总结报告
    print(f"\n🎉 **完整数据生成总结报告**")
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
    print(f"{'动作名称':<15} {'成功':<4} {'失败':<4} {'成功率':<7} {'平均ACMI':<9} {'平均CSV行':<9}")
    print("-" * 70)
    
    for action_name, results in action_results.items():
        success_count = len(results['successful'])
        fail_count = len(results['failed'])
        total_count = success_count + fail_count
        success_rate = success_count / total_count * 100 if total_count > 0 else 0
        
        if results['quality_analysis']:
            avg_acmi_size = sum(q['acmi_size_kb'] for q in results['quality_analysis']) / len(results['quality_analysis'])
            avg_csv_rows = sum(q['csv_rows'] for q in results['quality_analysis']) / len(results['quality_analysis'])
            print(f"{action_name:<15} {success_count:<4} {fail_count:<4} {success_rate:<7.1f}% {avg_acmi_size:<9.1f} {avg_csv_rows:<9.0f}")
        else:
            print(f"{action_name:<15} {success_count:<4} {fail_count:<4} {success_rate:<7.1f}% {'N/A':<9} {'N/A':<9}")
    
    # 最终评估
    if successful >= total * 0.8:
        print(f"\n🎉 **数据生成圆满成功！**")
        print("✅ 框架改进完成：18秒缓冲时间，优化duration参数")
        print("✅ 完整数据生成：12种动作数据集生成完成")
        print("✅ 质量验证通过：ACMI和CSV文件质量良好")
        print("✅ 数据可用性：可直接用于机器学习训练和Tacview可视化")
    else:
        print(f"\n⚠️ 数据生成部分成功，建议检查失败样本")
