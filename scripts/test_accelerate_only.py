#!/usr/bin/env python3
"""测试加速动作效果"""

import os
import sys
import logging

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from scripts.basic_action_data_generator import BasicActionDataGenerator

def test_single_accelerate():
    """测试单个减速动作"""
    print("🛑 测试减速动作效果（修复后）")
    print("=" * 50)

    # 创建数据生成器
    generator = BasicActionDataGenerator()

    # 生成一个减速样本
    try:
        acmi_path, csv_path = generator.generate_single_action_data("decelerate", 1)
        
        print(f"ACMI文件: {acmi_path}")
        print(f"CSV文件: {csv_path}")
        
        # 检查文件是否真的存在
        if os.path.exists(csv_path):
            print("✅ CSV文件成功生成")
            
            # 读取并分析数据
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            initial_velocity = df['Velocity_m_s'].iloc[0]
            final_velocity = df['Velocity_m_s'].iloc[-1]
            velocity_change = final_velocity - initial_velocity
            
            print(f"初始速度: {initial_velocity:.1f} m/s")
            print(f"最终速度: {final_velocity:.1f} m/s")
            print(f"速度变化: {velocity_change:.1f} m/s")
            
            if velocity_change < -50:
                print("✅ 减速效果强度正常（目标<-50 m/s）")
            elif velocity_change < 0:
                print("⚠️ 减速效果偏弱（需要<-50 m/s）")
            else:
                print("❌ 减速效果异常（速度未减少）")
                
        else:
            print("❌ CSV文件未生成")
            print(f"预期路径: {csv_path}")
            print(f"绝对路径: {os.path.abspath(csv_path)}")
            
        if os.path.exists(acmi_path):
            print("✅ ACMI文件成功生成")
        else:
            print("❌ ACMI文件未生成")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_accelerate()
