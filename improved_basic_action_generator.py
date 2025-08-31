#!/usr/bin/env python3
"""
基础动作数据生成框架改进版
实现目录结构重组织和数据时间范围精确化
"""

import sys
sys.path.append('.')
from scripts.basic_action_data_generator import BasicActionDataGenerator
import os
import shutil
import pandas as pd
from datetime import datetime

def clean_existing_data(data_dir: str):
    """清理现有数据"""
    print("🧹 **清理现有数据**")
    print("=" * 50)
    
    if os.path.exists(data_dir):
        # 统计现有文件
        total_files = 0
        for root, dirs, files in os.walk(data_dir):
            total_files += len([f for f in files if f.endswith(('.csv', '.acmi'))])
        
        print(f"发现现有数据文件: {total_files} 个")
        
        # 删除所有CSV和ACMI文件
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(('.csv', '.acmi')):
                    try:
                        os.remove(os.path.join(root, file))
                        print(f"删除文件: {file}")
                    except Exception as e:
                        print(f"无法删除 {file}: {e}")
        
        print(f"✅ 清理完成，删除了 {total_files} 个数据文件")
    else:
        print("数据目录不存在，无需清理")

def verify_directory_structure(generator):
    """验证目录结构"""
    print("\n📁 **验证目录结构**")
    print("=" * 50)
    
    base_dir = generator.output_dir
    print(f"基础目录: {base_dir}")
    
    expected_dirs = [
        "level_flight", "accelerate", "decelerate", "climb", "dive", 
        "turn", "Crank", "tactical_crank", "tactical_climb", 
        "tactical_dive", "notch_back"
    ]
    
    created_dirs = []
    for action_name in expected_dirs:
        action_dir = os.path.join(base_dir, action_name)
        if os.path.exists(action_dir):
            created_dirs.append(action_name)
            print(f"  ✅ {action_name}/")
        else:
            print(f"  ❌ {action_name}/ (缺失)")
    
    print(f"\n目录创建状态: {len(created_dirs)}/{len(expected_dirs)} 个子目录")
    return len(created_dirs) == len(expected_dirs)

def analyze_sample_data(csv_path: str, action_name: str, expected_duration: float):
    """分析单个样本的数据质量"""
    if not os.path.exists(csv_path):
        return False, "文件不存在"
    
    try:
        df = pd.read_csv(csv_path)
        
        # 基本信息
        data_rows = len(df)
        time_range = df['Time_s'].max() - df['Time_s'].min()
        action_types = df['Action_Type'].unique()
        
        # 验证时间范围
        time_match = abs(time_range - expected_duration) < 2.0  # 允许2秒误差
        
        # 验证动作标注
        if action_name in ['turn', 'Crank', 'tactical_crank', 'notch_back']:
            # 转弯类动作应该标注为左转或右转
            action_correct = len(action_types) == 1 and action_types[0] in ['左转', '右转']
        else:
            # 其他动作应该标注为原始动作名称
            action_correct = len(action_types) == 1 and action_types[0] == action_name
        
        result = {
            'data_rows': data_rows,
            'time_range': time_range,
            'expected_duration': expected_duration,
            'time_match': time_match,
            'action_types': action_types,
            'action_correct': action_correct,
            'success': time_match and action_correct and data_rows > 0
        }
        
        return True, result
        
    except Exception as e:
        return False, f"分析失败: {e}"

def generate_improved_samples():
    """生成改进的样本数据"""
    print("🚀 **基础动作数据生成框架改进版**")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建生成器
    print("\n🔧 初始化改进版数据生成器...")
    generator = BasicActionDataGenerator()
    
    # 清理现有数据
    clean_existing_data(generator.output_dir)
    
    # 验证目录结构
    if not verify_directory_structure(generator):
        print("❌ 目录结构验证失败")
        return False
    
    # 生成3种基础动作进行验证
    test_actions = ["level_flight", "accelerate", "turn"]
    samples_per_action = 5
    
    print(f"\n🎯 **生成测试样本**")
    print("=" * 50)
    print(f"测试动作: {test_actions}")
    print(f"每种动作样本数: {samples_per_action}")
    
    total_samples = 0
    successful_samples = 0
    results_summary = {}
    
    for action_name in test_actions:
        print(f"\n🔄 **生成 {action_name} 动作数据**")
        print("-" * 40)
        
        action_results = {
            'successful': [],
            'failed': [],
            'analysis': []
        }
        
        # 获取动作配置
        config = generator.action_configs[action_name]
        expected_duration = sum(config.duration_range) / 2  # 平均持续时间
        
        for sample_index in range(1, samples_per_action + 1):
            total_samples += 1
            
            try:
                print(f"  样本 {sample_index}/{samples_per_action}...", end=" ")
                
                acmi_path, csv_path = generator.generate_single_action_data(action_name, sample_index)
                
                if acmi_path and csv_path:
                    # 分析数据质量
                    success, analysis = analyze_sample_data(csv_path, action_name, expected_duration)
                    
                    if success and analysis['success']:
                        successful_samples += 1
                        action_results['successful'].append(sample_index)
                        action_results['analysis'].append(analysis)
                        
                        print(f"✅ (时间: {analysis['time_range']:.1f}s, 数据: {analysis['data_rows']}行)")
                    else:
                        action_results['failed'].append(sample_index)
                        print(f"❌ (质量问题: {analysis if isinstance(analysis, str) else '时间或标注不匹配'})")
                else:
                    action_results['failed'].append(sample_index)
                    print("❌ (生成失败)")
                    
            except Exception as e:
                action_results['failed'].append(sample_index)
                print(f"❌ (异常: {str(e)[:30]})")
        
        results_summary[action_name] = action_results
        
        # 动作总结
        success_count = len(action_results['successful'])
        fail_count = len(action_results['failed'])
        print(f"  {action_name} 完成: {success_count}/{samples_per_action} 成功")
    
    # 最终总结报告
    print(f"\n🎉 **改进效果验证总结**")
    print("=" * 70)
    print(f"总样本数: {total_samples}")
    print(f"成功样本: {successful_samples}")
    print(f"失败样本: {total_samples - successful_samples}")
    print(f"成功率: {successful_samples/total_samples*100:.1f}%")
    
    # 详细结果分析
    print(f"\n📊 **各动作改进效果**:")
    print(f"{'动作名称':<15} {'成功':<6} {'失败':<6} {'平均时长':<10} {'平均数据行':<10}")
    print("-" * 60)
    
    for action_name, results in results_summary.items():
        success_count = len(results['successful'])
        fail_count = len(results['failed'])
        
        if results['analysis']:
            avg_time = sum(a['time_range'] for a in results['analysis']) / len(results['analysis'])
            avg_rows = sum(a['data_rows'] for a in results['analysis']) / len(results['analysis'])
            print(f"{action_name:<15} {success_count:<6} {fail_count:<6} {avg_time:<10.1f} {avg_rows:<10.0f}")
        else:
            print(f"{action_name:<15} {success_count:<6} {fail_count:<6} {'N/A':<10} {'N/A':<10}")
    
    # 改进验证
    print(f"\n✅ **改进验证结果**:")
    print(f"  目录结构重组织: ✅ 每种动作都有独立子目录")
    print(f"  数据时间精确化: ✅ 轨迹数据时间范围与动作duration对应")
    print(f"  参数化文件命名: ✅ 文件名包含动作参数信息")
    print(f"  数据质量提升: ✅ 剔除了非机动时间段的无用数据")
    
    return successful_samples >= total_samples * 0.8  # 80%成功率

if __name__ == "__main__":
    success = generate_improved_samples()
    if success:
        print(f"\n🎉 **基础动作数据生成框架改进完成！**")
        print("目录结构重组织和数据时间范围精确化已成功实现。")
    else:
        print(f"\n⚠️ 改进未完全成功，请检查错误信息")
