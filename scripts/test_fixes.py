#!/usr/bin/env python3
"""
测试修复效果的脚本
只生成关键样本来验证加速和俯冲问题的修复
"""

import sys
import os
import logging

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from scripts.basic_action_data_generator import BasicActionDataGenerator, ActionConfig

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_fixes.log')
        ]
    )

def test_accelerate_fix():
    """测试加速动作修复"""
    print("测试加速动作修复")

    generator = BasicActionDataGenerator()
    generator.create_action_directories()

    # 生成2个加速样本
    for i in range(1, 3):
        success = generator.generate_action_data("accelerate", i)
        if success:
            print(f"加速样本 {i} 生成成功")
        else:
            print(f"加速样本 {i} 生成失败")

def test_dive_fix():
    """测试俯冲动作修复"""
    print("测试俯冲动作修复")

    generator = BasicActionDataGenerator()
    generator.create_action_directories()

    # 生成2个俯冲样本
    for i in range(1, 3):
        success = generator.generate_action_data("dive", i)
        if success:
            print(f"俯冲样本 {i} 生成成功")
        else:
            print(f"俯冲样本 {i} 生成失败")

def analyze_results():
    """分析生成结果"""
    import pandas as pd

    print("分析生成结果")
    
    # 分析加速数据
    accelerate_dir = "scripts/drag_shoot_2v2/basic_action_data/accelerate"
    if os.path.exists(accelerate_dir):
        for file in os.listdir(accelerate_dir):
            if file.endswith('.csv'):
                csv_path = os.path.join(accelerate_dir, file)
                df = pd.read_csv(csv_path)
                
                initial_velocity = df['Velocity_m_s'].iloc[0]
                final_velocity = df['Velocity_m_s'].iloc[-1]
                velocity_change = final_velocity - initial_velocity
                
                print(f"加速文件 {file}:")
                print(f"   初始速度: {initial_velocity:.1f}m/s")
                print(f"   最终速度: {final_velocity:.1f}m/s")
                print(f"   速度变化: {velocity_change:+.1f}m/s")

                if velocity_change > 0:
                    print("   加速正常工作")
                else:
                    print("   加速仍然异常")
    
    # 分析俯冲数据
    dive_dir = "scripts/drag_shoot_2v2/basic_action_data/dive"
    if os.path.exists(dive_dir):
        for file in os.listdir(dive_dir):
            if file.endswith('.csv'):
                csv_path = os.path.join(dive_dir, file)
                df = pd.read_csv(csv_path)
                
                initial_altitude = df['Z_m'].iloc[0]
                final_altitude = df['Z_m'].iloc[-1]
                altitude_change = initial_altitude - final_altitude
                
                print(f"俯冲文件 {file}:")
                print(f"   初始高度: {initial_altitude:.1f}m")
                print(f"   最终高度: {final_altitude:.1f}m")
                print(f"   高度下降: {altitude_change:.1f}m")

                # 从文件名提取预期俯冲距离
                if "dive_" in file:
                    try:
                        expected_str = file.split("dive_")[1].split("m_")[0]
                        expected_loss = float(expected_str)
                        error_rate = abs(altitude_change - expected_loss) / expected_loss * 100

                        print(f"   预期俯冲: {expected_loss:.1f}m")
                        print(f"   误差率: {error_rate:.1f}%")

                        if error_rate < 5.0:
                            print("   俯冲精度达标")
                        else:
                            print("   俯冲误差过大")
                    except:
                        print("   无法解析预期俯冲距离")

def main():
    """主函数"""
    print("开始测试修复效果")

    # 测试加速修复
    test_accelerate_fix()

    # 测试俯冲修复
    test_dive_fix()

    # 分析结果
    analyze_results()

    print("测试完成")

if __name__ == "__main__":
    main()
