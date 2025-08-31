#!/usr/bin/env python3
"""测试左转动作生成"""

import sys
sys.path.append('.')
from scripts.basic_action_data_generator import BasicActionDataGenerator
import pandas as pd

def test_left_turn():
    print('🧪 测试左转动作数据生成')
    generator = BasicActionDataGenerator()

    print('测试左转动作...')
    acmi_path, csv_path = generator.generate_single_action_data('左转', 1)

    if acmi_path and csv_path:
        print(f'✅ 成功生成:')
        print(f'  ACMI: {acmi_path}')
        print(f'  CSV: {csv_path}')
        
        # 检查CSV内容
        df = pd.read_csv(csv_path)
        print(f'  数据行数: {len(df)}')
        action_types = df['Action_Type'].unique()
        print(f'  动作类型: {action_types}')
        
        # 检查A0100的航向变化（测试飞机）
        a0100_data = df[df['Agent_ID'] == 'A0100']
        initial_heading = a0100_data['Heading_deg'].iloc[0]
        final_heading = a0100_data['Heading_deg'].iloc[-1]
        print(f'  A0100航向变化: {initial_heading:.1f}° -> {final_heading:.1f}°')

        # 计算航向变化量（考虑360度循环）
        heading_change = final_heading - initial_heading
        if heading_change > 180:
            heading_change -= 360
        elif heading_change < -180:
            heading_change += 360

        print(f'  航向变化量: {heading_change:.1f}°')

        # 从0°到292°实际上是左转-68°
        if abs(heading_change + 68) < 10:  # 允许10度误差
            print('  ✅ 确认为左转 (符合预期的-68°左转)')
        elif heading_change < 0:
            print('  ✅ 确认为左转 (负角度变化)')
        else:
            print('  ❌ 错误：应该是左转但显示右转')
            
    else:
        print('❌ 生成失败')

if __name__ == "__main__":
    test_left_turn()
