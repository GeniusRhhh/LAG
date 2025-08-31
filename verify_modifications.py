#!/usr/bin/env python3
"""验证基础动作数据生成框架的修改"""

import os
import sys
sys.path.append('.')

def verify_modifications():
    print("🔧 验证基础动作数据生成框架修改")
    print("=" * 50)
    
    try:
        from scripts.basic_action_data_generator import BasicActionDataGenerator
        
        # 1. 验证目录结构调整
        print("\n1️⃣ 验证目录结构调整:")
        generator = BasicActionDataGenerator()
        print(f"  输出目录: {generator.output_dir}")
        print(f"  ACMI目录: {generator.acmi_dir}")
        print(f"  CSV目录: {generator.csv_dir}")
        
        expected_base = "scripts/drag_shoot_2v2/basic_action_data"
        if expected_base in generator.output_dir:
            print("  ✅ 目录结构调整正确")
        else:
            print("  ❌ 目录结构调整失败")
        
        # 2. 验证Crank参数范围扩大
        print("\n2️⃣ 验证Crank参数范围扩大:")
        crank_config = generator.action_configs['Crank']
        print(f"  转弯角度范围: {crank_config.param_ranges['turn_angle']}")
        print(f"  转弯速率范围: {crank_config.param_ranges['turn_rate']}")
        
        angle_range = crank_config.param_ranges['turn_angle']
        rate_range = crank_config.param_ranges['turn_rate']
        
        if angle_range == (25.0, 70.0) and rate_range == (2.5, 6.0):
            print("  ✅ Crank参数范围扩大正确")
        else:
            print("  ❌ Crank参数范围扩大失败")
        
        # 3. 测试单个Crank样本生成
        print("\n3️⃣ 测试单个Crank样本生成:")
        try:
            acmi_path, csv_path = generator.generate_single_action_data('Crank', 1)
            
            if acmi_path and csv_path:
                print(f"  ✅ 样本生成成功")
                print(f"  ACMI: {os.path.basename(acmi_path)}")
                print(f"  CSV: {os.path.basename(csv_path)}")
                
                # 验证文件存在
                if os.path.exists(csv_path):
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    action_types = df['Action_Type'].unique()
                    print(f"  动作标注: {action_types}")
                    
                    if len(action_types) == 1 and action_types[0] in ['左转', '右转']:
                        print("  ✅ 方向性标注正确")
                    else:
                        print("  ❌ 方向性标注失败")
                else:
                    print("  ❌ CSV文件未生成")
            else:
                print("  ❌ 样本生成失败")
                
        except Exception as e:
            print(f"  ❌ 样本生成异常: {e}")
        
        print("\n🎉 修改验证完成!")
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
    except Exception as e:
        print(f"❌ 验证异常: {e}")

if __name__ == "__main__":
    verify_modifications()
