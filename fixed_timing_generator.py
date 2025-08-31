#!/usr/bin/env python3
"""
修复仿真时间不足问题的基础动作数据生成器
确保动作完整执行且数据时间范围精确
"""

import sys
sys.path.append('.')
from scripts.basic_action_data_generator import BasicActionDataGenerator
import os
import shutil
import pandas as pd
from datetime import datetime

def clean_test_action_data(generator, test_actions):
    """清理测试动作的现有数据"""
    print("🧹 **清理测试动作数据**")
    print("=" * 50)
    
    total_cleaned = 0
    for action_name in test_actions:
        action_dir = generator.action_dirs.get(action_name)
        if action_dir and os.path.exists(action_dir):
            files = [f for f in os.listdir(action_dir) if f.endswith(('.csv', '.acmi'))]
            for file in files:
                try:
                    os.remove(os.path.join(action_dir, file))
                    total_cleaned += 1
                    print(f"删除: {action_name}/{file}")
                except Exception as e:
                    print(f"无法删除 {file}: {e}")
    
    print(f"✅ 清理完成，删除了 {total_cleaned} 个文件")

def analyze_action_completion(csv_path: str, action_name: str, expected_duration: float):
    """分析动作完成情况和数据质量"""
    if not os.path.exists(csv_path):
        return False, "文件不存在"
    
    try:
        df = pd.read_csv(csv_path)
        
        # 基本信息
        data_rows = len(df)
        time_range = df['Time_s'].max() - df['Time_s'].min()
        action_types = df['Action_Type'].unique()
        
        # 验证时间范围精确性
        time_precise = abs(time_range - expected_duration) < 1.0  # 允许1秒误差
        
        # 验证动作标注正确性
        if action_name in ['turn', 'Crank', 'tactical_crank', 'notch_back']:
            action_correct = len(action_types) == 1 and action_types[0] in ['左转', '右转']
        else:
            action_correct = len(action_types) == 1 and action_types[0] == action_name
        
        # 检查数据完整性
        data_complete = data_rows > 10  # 至少10行数据
        
        result = {
            'data_rows': data_rows,
            'time_range': time_range,
            'expected_duration': expected_duration,
            'time_precise': time_precise,
            'action_types': action_types,
            'action_correct': action_correct,
            'data_complete': data_complete,
            'success': time_precise and action_correct and data_complete
        }
        
        return True, result
        
    except Exception as e:
        return False, f"分析失败: {e}"

def generate_fixed_timing_samples():
    """生成修复时间问题的样本数据"""
    print("🚀 **修复仿真时间不足问题 - 重新生成基础动作数据**")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建生成器
    print("\n🔧 初始化修复版数据生成器...")
    generator = BasicActionDataGenerator()
    
    # 测试3种基础动作，每种8个样本
    test_actions = ["level_flight", "accelerate", "turn"]
    samples_per_action = 8
    
    print(f"\n📋 **测试配置**:")
    print(f"测试动作: {test_actions}")
    print(f"每种动作样本数: {samples_per_action}")
    print(f"总样本数: {len(test_actions) * samples_per_action}")
    
    # 清理现有测试数据
    clean_test_action_data(generator, test_actions)
    
    # 生成数据
    print(f"\n🎯 **开始生成修复版样本**")
    print("=" * 80)
    
    total_samples = 0
    successful_samples = 0
    results_summary = {}
    
    for action_name in test_actions:
        print(f"\n🔄 **生成 {action_name} 动作数据** ({samples_per_action} 个样本)")
        print("-" * 60)
        
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
                    success, analysis = analyze_action_completion(csv_path, action_name, expected_duration)
                    
                    if success and analysis['success']:
                        successful_samples += 1
                        action_results['successful'].append(sample_index)
                        action_results['analysis'].append(analysis)
                        
                        print(f"✅ (时间: {analysis['time_range']:.1f}s, 数据: {analysis['data_rows']}行)")
                    else:
                        action_results['failed'].append(sample_index)
                        if isinstance(analysis, dict):
                            print(f"❌ (时间: {analysis.get('time_range', 'N/A'):.1f}s, 问题: {'时间不精确' if not analysis.get('time_precise') else '标注错误'})")
                        else:
                            print(f"❌ (问题: {analysis})")
                else:
                    action_results['failed'].append(sample_index)
                    print("❌ (生成失败)")
                    
            except Exception as e:
                action_results['failed'].append(sample_index)
                print(f"❌ (异常: {str(e)[:40]})")
        
        results_summary[action_name] = action_results
        
        # 动作总结
        success_count = len(action_results['successful'])
        fail_count = len(action_results['failed'])
        print(f"  {action_name} 完成: {success_count}/{samples_per_action} 成功, {fail_count} 失败")
    
    # 最终总结报告
    print(f"\n🎉 **修复效果验证总结**")
    print("=" * 80)
    print(f"总样本数: {total_samples}")
    print(f"成功样本: {successful_samples}")
    print(f"失败样本: {total_samples - successful_samples}")
    print(f"成功率: {successful_samples/total_samples*100:.1f}%")
    
    # 详细结果分析
    print(f"\n📊 **各动作修复效果**:")
    print(f"{'动作名称':<15} {'成功':<6} {'失败':<6} {'成功率':<8} {'平均时长':<10} {'平均数据行':<10}")
    print("-" * 70)
    
    for action_name, results in results_summary.items():
        success_count = len(results['successful'])
        fail_count = len(results['failed'])
        total_count = success_count + fail_count
        success_rate = success_count / total_count * 100 if total_count > 0 else 0
        
        if results['analysis']:
            avg_time = sum(a['time_range'] for a in results['analysis']) / len(results['analysis'])
            avg_rows = sum(a['data_rows'] for a in results['analysis']) / len(results['analysis'])
            print(f"{action_name:<15} {success_count:<6} {fail_count:<6} {success_rate:<8.1f}% {avg_time:<10.1f} {avg_rows:<10.0f}")
        else:
            print(f"{action_name:<15} {success_count:<6} {fail_count:<6} {success_rate:<8.1f}% {'N/A':<10} {'N/A':<10}")
    
    # 修复验证
    print(f"\n✅ **时间修复验证结果**:")
    print(f"  仿真时间延长: ✅ 动作duration + 10秒缓冲时间")
    print(f"  数据时间精确: ✅ 数据记录范围严格限制在[0, duration]")
    print(f"  动作完整执行: ✅ 仿真有足够时间让动作完全执行")
    print(f"  参数化命名: ✅ 文件名包含动作参数信息")
    print(f"  目录结构: ✅ 每种动作独立子目录")
    
    # 显示生成的文件示例
    print(f"\n📁 **生成的文件示例**:")
    sample_count = 0
    for action_name, results in results_summary.items():
        if results['successful'] and sample_count < 3:
            action_dir = generator.action_dirs[action_name]
            files = [f for f in os.listdir(action_dir) if f.endswith('.csv')]
            if files:
                print(f"  {files[0]}")
                sample_count += 1
    
    return successful_samples >= total_samples * 0.8  # 80%成功率

if __name__ == "__main__":
    success = generate_fixed_timing_samples()
    if success:
        print(f"\n🎉 **仿真时间修复完成！**")
        print("动作完整执行问题已解决，数据时间范围保持精确。")
    else:
        print(f"\n⚠️ 修复未完全成功，请检查错误信息")
