#!/usr/bin/env python3
"""
测试仿真时间修复效果
"""

import sys
sys.path.append('.')
from scripts.basic_action_data_generator import BasicActionDataGenerator
import pandas as pd
import os

def test_single_action_timing():
    """测试单个动作的时间精确度"""
    print("🧪 **测试仿真时间修复效果**")
    print("=" * 50)
    
    # 创建生成器
    generator = BasicActionDataGenerator()
    
    # 测试level_flight动作
    action_name = "level_flight"
    sample_index = 999  # 使用特殊编号避免冲突
    
    print(f"测试动作: {action_name}")
    print(f"样本编号: {sample_index}")
    
    try:
        # 生成样本
        print("开始生成样本...")
        acmi_path, csv_path = generator.generate_single_action_data(action_name, sample_index)
        
        if csv_path and os.path.exists(csv_path):
            # 分析CSV数据
            df = pd.read_csv(csv_path)
            
            # 基本信息
            data_rows = len(df)
            time_min = df['Time_s'].min()
            time_max = df['Time_s'].max()
            time_range = time_max - time_min
            
            print(f"\n📊 **数据分析结果**:")
            print(f"CSV文件: {os.path.basename(csv_path)}")
            print(f"数据行数: {data_rows}")
            print(f"时间范围: {time_min:.2f}s - {time_max:.2f}s")
            print(f"时间跨度: {time_range:.2f}s")
            
            # 从文件名提取预期duration
            filename = os.path.basename(csv_path)
            if "_" in filename:
                parts = filename.split("_")
                for part in parts:
                    if part.endswith("s"):
                        try:
                            expected_duration = float(part[:-1])
                            print(f"预期时长: {expected_duration:.2f}s")
                            
                            # 计算精确度
                            time_diff = abs(time_range - expected_duration)
                            print(f"时间差异: {time_diff:.2f}s")
                            
                            if time_diff < 0.5:
                                print("✅ 时间精确度: 优秀 (误差 < 0.5s)")
                            elif time_diff < 1.0:
                                print("⚠️ 时间精确度: 良好 (误差 < 1.0s)")
                            else:
                                print("❌ 时间精确度: 需要改进 (误差 >= 1.0s)")
                            
                            break
                        except:
                            continue
            
            # 显示前5行和后5行数据
            print(f"\n📋 **数据样本 (前5行)**:")
            print(df.head().to_string(index=False))
            
            print(f"\n📋 **数据样本 (后5行)**:")
            print(df.tail().to_string(index=False))
            
            # 清理测试文件
            try:
                os.remove(csv_path)
                if acmi_path and os.path.exists(acmi_path):
                    os.remove(acmi_path)
                print(f"\n🧹 测试文件已清理")
            except:
                pass
            
            return True
            
        else:
            print("❌ 样本生成失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

if __name__ == "__main__":
    success = test_single_action_timing()
    if success:
        print(f"\n🎉 时间修复测试完成")
    else:
        print(f"\n⚠️ 测试失败")
