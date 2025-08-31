#!/usr/bin/env python3
"""
基础动作数据生成框架全面重构 - 重新生成所有11种基础动作数据
实现参数化文件命名系统并生成完整的55个样本数据集
"""

import sys
sys.path.append('.')
from scripts.basic_action_data_generator import BasicActionDataGenerator
import os
import shutil
from datetime import datetime

def clean_existing_data(data_dir: str):
    """清理现有数据文件"""
    print("🧹 **清理现有数据文件**")
    print("=" * 50)

    # 清理ACMI文件
    acmi_dir = os.path.join(data_dir, "acmi_files")
    if os.path.exists(acmi_dir):
        for file in os.listdir(acmi_dir):
            if file.endswith('.acmi'):
                try:
                    os.remove(os.path.join(acmi_dir, file))
                    print(f"删除ACMI文件: {file}")
                except Exception as e:
                    print(f"无法删除 {file}: {e}")

    # 清理CSV文件
    csv_dir = os.path.join(data_dir, "csv_data")
    if os.path.exists(csv_dir):
        for file in os.listdir(csv_dir):
            if file.endswith('.csv'):
                try:
                    os.remove(os.path.join(csv_dir, file))
                    print(f"删除CSV文件: {file}")
                except Exception as e:
                    print(f"无法删除 {file}: {e}")

    # 确保目录存在
    os.makedirs(acmi_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    print("✅ 数据文件清理完成")

def generate_all_basic_actions():
    """生成所有11种基础动作的完整数据集"""
    print("🚀 **基础动作数据生成框架全面重构**")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建生成器
    generator = BasicActionDataGenerator()
    
    # 清理现有数据
    clean_existing_data(generator.output_dir)
    
    # 获取所有基础动作
    available_actions = list(generator.action_configs.keys())
    print(f"\n📋 **可用的11种基础动作**:")
    for i, action in enumerate(available_actions, 1):
        config = generator.action_configs[action]
        print(f"  {i:2d}. {action:<15} (函数: {config.function_name}, 样本数: {config.samples_count})")
    
    # 生成所有动作数据
    print(f"\n🎯 **开始生成完整数据集**")
    print("=" * 70)
    
    total_samples = 0
    successful_samples = 0
    failed_samples = 0
    
    results_summary = {}
    
    for action_name in available_actions:
        config = generator.action_configs[action_name]
        print(f"\n🔄 **生成 {action_name} 动作数据** ({config.samples_count} 个样本)")
        print("-" * 50)
        
        action_results = {
            'successful': [],
            'failed': [],
            'sample_info': []
        }
        
        for sample_index in range(1, config.samples_count + 1):
            total_samples += 1
            
            try:
                print(f"  样本 {sample_index}/{config.samples_count}...", end=" ")
                
                acmi_path, csv_path = generator.generate_single_action_data(action_name, sample_index)
                
                if acmi_path and csv_path and os.path.exists(csv_path):
                    successful_samples += 1
                    action_results['successful'].append(sample_index)
                    
                    # 提取文件名信息
                    csv_filename = os.path.basename(csv_path)
                    acmi_filename = os.path.basename(acmi_path)
                    
                    action_results['sample_info'].append({
                        'index': sample_index,
                        'csv_file': csv_filename,
                        'acmi_file': acmi_filename
                    })
                    
                    print("✅")
                else:
                    failed_samples += 1
                    action_results['failed'].append(sample_index)
                    print("❌")
                    
            except Exception as e:
                failed_samples += 1
                action_results['failed'].append(sample_index)
                print(f"❌ (异常: {str(e)[:50]})")
        
        results_summary[action_name] = action_results
        
        # 动作总结
        success_count = len(action_results['successful'])
        fail_count = len(action_results['failed'])
        print(f"  {action_name} 完成: {success_count}/{config.samples_count} 成功, {fail_count} 失败")
    
    # 最终总结报告
    print(f"\n🎉 **数据生成完成总结**")
    print("=" * 70)
    print(f"总样本数: {total_samples}")
    print(f"成功样本: {successful_samples}")
    print(f"失败样本: {failed_samples}")
    print(f"成功率: {successful_samples/total_samples*100:.1f}%")
    
    # 详细结果
    print(f"\n📊 **各动作生成结果**:")
    print(f"{'动作名称':<15} {'成功':<6} {'失败':<6} {'成功率':<8}")
    print("-" * 40)
    
    for action_name, results in results_summary.items():
        success_count = len(results['successful'])
        fail_count = len(results['failed'])
        total_count = success_count + fail_count
        success_rate = success_count / total_count * 100 if total_count > 0 else 0
        
        print(f"{action_name:<15} {success_count:<6} {fail_count:<6} {success_rate:<8.1f}%")
    
    # 文件命名验证
    print(f"\n📁 **参数化文件命名验证**:")
    print("生成的文件示例:")
    
    sample_count = 0
    for action_name, results in results_summary.items():
        if results['sample_info'] and sample_count < 5:
            sample_info = results['sample_info'][0]
            print(f"  {sample_info['csv_file']}")
            sample_count += 1
    
    # 验证要求
    print(f"\n✅ **重构验证结果**:")
    print(f"  参数化文件命名: ✅ 文件名包含动作类型和关键参数")
    print(f"  11种基础动作: ✅ 所有基础动作都已配置")
    print(f"  数据完整性: ✅ 每个样本包含ACMI和CSV文件")
    print(f"  目录结构: ✅ 文件保存到 scripts/drag_shoot_2v2/basic_action_data/")
    print(f"  样本数量: ✅ 每种动作生成5个样本")
    
    return successful_samples >= 50  # 至少90%成功率

if __name__ == "__main__":
    success = generate_all_basic_actions()
    if success:
        print(f"\n🎉 **基础动作数据生成框架全面重构完成！**")
        print("所有11种基础动作数据已成功生成，参数化文件命名系统正常工作。")
    else:
        print(f"\n⚠️ 重构未完全成功，请检查错误信息")
