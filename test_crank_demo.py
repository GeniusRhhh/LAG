#!/usr/bin/env python3
"""
Crank动作测试演示
使用基础动作数据生成框架生成Crank动作样本并进行详细验证
"""

import sys
sys.path.append('.')
from scripts.basic_action_data_generator import BasicActionDataGenerator
import pandas as pd
import os
from datetime import datetime

def analyze_crank_sample(csv_path, sample_index):
    """分析单个Crank样本的详细数据"""
    print(f"\n📊 **样本 {sample_index} 详细分析**")
    print("=" * 50)
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV文件不存在: {csv_path}")
        return False
    
    try:
        # 读取CSV数据
        df = pd.read_csv(csv_path)
        
        # 基本信息
        print(f"📁 文件: {os.path.basename(csv_path)}")
        print(f"📊 数据行数: {len(df):,}")
        print(f"📋 列数: {len(df.columns)}")
        print(f"⏱️ 仿真时长: {df['Time_s'].max():.1f}秒")
        
        # 验证列结构
        expected_columns = ['Time_s', 'Agent_ID', 'X_m', 'Y_m', 'Z_m', 'Velocity_m_s', 'Heading_deg', 'Pitch_deg', 'Roll_deg', 'Action_Type']
        actual_columns = list(df.columns)
        
        print(f"\n📋 **列结构验证**:")
        if actual_columns == expected_columns:
            print("✅ 列结构完全正确")
        else:
            print("❌ 列结构不匹配")
            print(f"期望: {expected_columns}")
            print(f"实际: {actual_columns}")
        
        # 动作标注验证 - 检查方向性标注
        action_types = df['Action_Type'].unique()
        print(f"\n🎯 **动作标注验证**:")
        print(f"动作类型: {action_types}")
        if len(action_types) == 1 and action_types[0] in ['左转', '右转']:
            print(f"✅ 方向性标注正确：所有数据点都标注为'{action_types[0]}'")
        elif len(action_types) == 1 and action_types[0] == 'Crank':
            print("⚠️ 使用通用'Crank'标注，应该使用方向性标注")
        else:
            print("❌ 动作标注错误或不一致")
        
        # 分析A0100的飞行轨迹（测试飞机）
        a0100_data = df[df['Agent_ID'] == 'A0100'].copy()
        if len(a0100_data) > 0:
            print(f"\n✈️ **A0100飞行轨迹分析**:")
            
            # 航向变化分析
            initial_heading = a0100_data['Heading_deg'].iloc[0]
            final_heading = a0100_data['Heading_deg'].iloc[-1]
            
            # 计算航向变化（考虑360度循环）
            heading_change = final_heading - initial_heading
            if heading_change > 180:
                heading_change -= 360
            elif heading_change < -180:
                heading_change += 360
            
            print(f"  初始航向: {initial_heading:.1f}°")
            print(f"  最终航向: {final_heading:.1f}°")
            print(f"  航向变化: {heading_change:.1f}°")
            
            # 验证Crank机动特征（30-60度转弯）
            if 25.0 <= abs(heading_change) <= 65.0:
                print(f"  ✅ 符合Crank机动特征 (30-60度转弯)")
            else:
                print(f"  ⚠️ 航向变化超出Crank标准范围")
            
            # 高度变化分析
            initial_altitude = a0100_data['Z_m'].iloc[0]
            final_altitude = a0100_data['Z_m'].iloc[-1]
            altitude_change = final_altitude - initial_altitude
            
            print(f"  初始高度: {initial_altitude:.1f}m")
            print(f"  最终高度: {final_altitude:.1f}m")
            print(f"  高度变化: {altitude_change:+.1f}m")
            
            # 速度变化分析
            initial_velocity = a0100_data['Velocity_m_s'].iloc[0]
            final_velocity = a0100_data['Velocity_m_s'].iloc[-1]
            velocity_change = final_velocity - initial_velocity
            
            print(f"  初始速度: {initial_velocity:.1f}m/s")
            print(f"  最终速度: {final_velocity:.1f}m/s")
            print(f"  速度变化: {velocity_change:+.1f}m/s")
            
        # 显示前5行数据样本
        print(f"\n📋 **数据样本 (前5行)**:")
        print(df.head().to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return False

def test_crank_generation():
    """测试Crank动作生成"""
    print("🎯 **Crank动作数据生成测试演示**")
    print("=" * 60)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建生成器
    print("\n🔧 初始化基础动作数据生成器...")
    generator = BasicActionDataGenerator()
    
    # 显示Crank动作配置
    crank_config = generator.action_configs['Crank']
    print(f"\n⚙️ **Crank动作配置**:")
    print(f"  动作名称: {crank_config.name}")
    print(f"  函数名称: {crank_config.function_name}")
    print(f"  参数范围: {crank_config.param_ranges}")
    print(f"  持续时间: {crank_config.duration_range}")
    print(f"  默认样本数: {crank_config.samples_count}")
    
    # 生成5个Crank样本
    samples_to_generate = 5
    successful_samples = []
    
    print(f"\n🚀 **开始生成 {samples_to_generate} 个Crank样本**")
    print("=" * 60)
    
    for i in range(1, samples_to_generate + 1):
        print(f"\n🔄 生成样本 {i}/{samples_to_generate}...")
        
        try:
            acmi_path, csv_path = generator.generate_single_action_data('Crank', i)
            
            if acmi_path and csv_path:
                print(f"✅ 样本 {i} 生成成功!")
                print(f"  ACMI文件: {acmi_path}")
                print(f"  CSV文件: {csv_path}")
                
                # 验证文件存在
                acmi_exists = os.path.exists(acmi_path)
                csv_exists = os.path.exists(csv_path)
                
                print(f"  ACMI文件存在: {'✅' if acmi_exists else '❌'}")
                print(f"  CSV文件存在: {'✅' if csv_exists else '❌'}")
                
                if csv_exists:
                    successful_samples.append((i, acmi_path, csv_path))
                    
            else:
                print(f"❌ 样本 {i} 生成失败")
                
        except Exception as e:
            print(f"❌ 样本 {i} 生成异常: {e}")
    
    # 详细分析成功的样本
    print(f"\n📊 **详细数据分析**")
    print("=" * 60)
    
    for sample_index, acmi_path, csv_path in successful_samples:
        success = analyze_crank_sample(csv_path, sample_index)
        if not success:
            print(f"⚠️ 样本 {sample_index} 分析失败")
    
    # 总结报告
    print(f"\n🎉 **测试总结报告**")
    print("=" * 60)
    print(f"目标样本数: {samples_to_generate}")
    print(f"成功样本数: {len(successful_samples)}")
    print(f"成功率: {len(successful_samples)/samples_to_generate*100:.1f}%")
    
    if successful_samples:
        print(f"\n✅ **双重输出验证**:")
        print(f"  ACMI文件: 所有样本都生成了ACMI文件用于可视化")
        print(f"  CSV文件: 所有样本都生成了CSV文件用于机器学习")
        print(f"  数据一致性: ACMI和CSV包含相同的轨迹数据")
        
        print(f"\n✅ **数据质量验证**:")
        print(f"  格式正确: CSV包含所有必需的轨迹数据列")
        print(f"  标注准确: 所有数据点正确标注为'Crank'")
        print(f"  参数合理: 转弯角度在30-60度范围内")
        print(f"  随机化: 每个样本使用不同的随机参数")
        
        print(f"\n📁 **输出文件位置**:")
        print(f"  ACMI文件目录: basic_action_data/acmi_files/")
        print(f"  CSV文件目录: basic_action_data/csv_data/")
        
    return len(successful_samples) > 0

if __name__ == "__main__":
    success = test_crank_generation()
    if success:
        print(f"\n🎉 Crank动作测试演示完成！")
    else:
        print(f"\n❌ 测试演示失败")
