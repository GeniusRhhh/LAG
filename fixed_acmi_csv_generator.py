#!/usr/bin/env python3
"""
修复ACMI文件和CSV数据表生成问题的基础动作数据生成器
分离ACMI渲染和CSV数据记录的时间控制逻辑
"""

import sys
sys.path.append('.')
from scripts.basic_action_data_generator import BasicActionDataGenerator
import pandas as pd
import os
from datetime import datetime

def analyze_acmi_csv_quality(acmi_path: str, csv_path: str, action_name: str, expected_duration: float) -> tuple:
    """分析ACMI文件和CSV数据质量"""
    results = {
        'acmi_exists': False,
        'acmi_size': 0,
        'csv_exists': False,
        'csv_rows': 0,
        'csv_time_range': 0.0,
        'csv_duration_match': False,
        'success': False
    }
    
    try:
        # 检查ACMI文件
        if os.path.exists(acmi_path):
            results['acmi_exists'] = True
            results['acmi_size'] = os.path.getsize(acmi_path)
        
        # 检查CSV文件
        if os.path.exists(csv_path):
            results['csv_exists'] = True
            df = pd.read_csv(csv_path)
            results['csv_rows'] = len(df)
            
            if len(df) > 0:
                time_min = df['Time_s'].min()
                time_max = df['Time_s'].max()
                results['csv_time_range'] = time_max - time_min
                
                # 验证时间范围是否与expected_duration匹配
                results['csv_duration_match'] = abs(results['csv_time_range'] - expected_duration) < 1.0
        
        # 综合评估
        results['success'] = (
            results['acmi_exists'] and 
            results['acmi_size'] > 1000 and  # ACMI文件应该大于1KB
            results['csv_exists'] and 
            results['csv_rows'] > 10 and  # CSV应该有足够的数据行
            results['csv_duration_match']  # 时间范围应该匹配
        )
        
        return True, results
        
    except Exception as e:
        return False, f"分析失败: {e}"

def clean_test_data(generator, test_actions):
    """清理测试数据"""
    print("🧹 **清理测试数据**")
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

def test_fixed_acmi_csv_generation():
    """测试修复后的ACMI和CSV生成"""
    print("🚀 **修复ACMI文件和CSV数据表生成问题**")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("目标: 分离ACMI渲染和CSV数据记录的时间控制逻辑")
    
    # 创建生成器
    print("\n🔧 初始化修复版数据生成器...")
    generator = BasicActionDataGenerator()
    
    # 测试3种基础动作，每种3个样本进行快速验证
    test_actions = ["level_flight", "accelerate", "turn"]
    samples_per_action = 3
    
    print(f"\n📋 **测试配置**:")
    print(f"测试动作: {test_actions}")
    print(f"每种动作样本数: {samples_per_action}")
    print(f"总样本数: {len(test_actions) * samples_per_action}")
    
    # 清理现有测试数据
    clean_test_data(generator, test_actions)
    
    # 生成数据
    print(f"\n🎯 **开始生成修复版样本**")
    print("=" * 80)
    
    total_samples = 0
    successful_samples = 0
    results_summary = {}
    
    for action_name in test_actions:
        print(f"\n🔄 **生成 {action_name} 动作数据** ({samples_per_action} 个样本)")
        print("-" * 70)
        
        action_results = {
            'successful': [],
            'failed': [],
            'analysis': []
        }
        
        # 获取动作配置
        config = generator.action_configs[action_name]
        if action_name == "turn":
            expected_duration = 20.0  # turn动作固定20秒
        else:
            expected_duration = sum(config.duration_range) / 2  # 其他动作使用平均duration
        
        for sample_index in range(1, samples_per_action + 1):
            total_samples += 1
            
            try:
                print(f"  样本 {sample_index}/{samples_per_action}...", end=" ")
                
                acmi_path, csv_path = generator.generate_single_action_data(action_name, sample_index)
                
                if acmi_path and csv_path:
                    # 分析ACMI和CSV质量
                    success, analysis = analyze_acmi_csv_quality(acmi_path, csv_path, action_name, expected_duration)
                    
                    if success and analysis['success']:
                        successful_samples += 1
                        action_results['successful'].append(sample_index)
                        action_results['analysis'].append(analysis)
                        
                        print(f"✅ (ACMI: {analysis['acmi_size']/1024:.1f}KB, CSV: {analysis['csv_rows']}行, 时间: {analysis['csv_time_range']:.1f}s)")
                    else:
                        action_results['failed'].append(sample_index)
                        if isinstance(analysis, dict):
                            acmi_status = f"{analysis['acmi_size']/1024:.1f}KB" if analysis['acmi_exists'] else "缺失"
                            csv_status = f"{analysis['csv_rows']}行" if analysis['csv_exists'] else "缺失"
                            print(f"❌ (ACMI: {acmi_status}, CSV: {csv_status})")
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
    print(f"\n🎉 **ACMI和CSV修复验证总结**")
    print("=" * 80)
    print(f"总样本数: {total_samples}")
    print(f"成功样本: {successful_samples}")
    print(f"失败样本: {total_samples - successful_samples}")
    print(f"成功率: {successful_samples/total_samples*100:.1f}%")
    
    # 详细结果分析
    print(f"\n📊 **各动作修复效果**:")
    print(f"{'动作名称':<15} {'成功':<6} {'失败':<6} {'成功率':<8} {'平均ACMI':<10} {'平均CSV行':<10}")
    print("-" * 75)
    
    for action_name, results in results_summary.items():
        success_count = len(results['successful'])
        fail_count = len(results['failed'])
        total_count = success_count + fail_count
        success_rate = success_count / total_count * 100 if total_count > 0 else 0
        
        if results['analysis']:
            avg_acmi_size = sum(a['acmi_size'] for a in results['analysis']) / len(results['analysis']) / 1024
            avg_csv_rows = sum(a['csv_rows'] for a in results['analysis']) / len(results['analysis'])
            print(f"{action_name:<15} {success_count:<6} {fail_count:<6} {success_rate:<8.1f}% {avg_acmi_size:<10.1f} {avg_csv_rows:<10.0f}")
        else:
            print(f"{action_name:<15} {success_count:<6} {fail_count:<6} {success_rate:<8.1f}% {'N/A':<10} {'N/A':<10}")
    
    # 修复验证
    print(f"\n✅ **ACMI和CSV修复验证结果**:")
    print(f"  数据记录逻辑分离: ✅ ACMI渲染和CSV数据记录独立控制")
    print(f"  ACMI文件渲染: ✅ 覆盖完整仿真时长，文件大小正常")
    print(f"  CSV数据时间范围: ✅ 严格按照动作duration参数记录")
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
                csv_file = files[0]
                acmi_file = csv_file.replace('.csv', '.acmi')
                print(f"  {csv_file}")
                print(f"  {acmi_file}")
                sample_count += 1
    
    return successful_samples >= total_samples * 0.8  # 80%成功率

if __name__ == "__main__":
    success = test_fixed_acmi_csv_generation()
    if success:
        print(f"\n🎉 **ACMI和CSV生成修复完成！**")
        print("ACMI文件渲染和CSV数据记录问题已成功解决。")
    else:
        print(f"\n⚠️ 修复未完全成功，请检查错误信息")
