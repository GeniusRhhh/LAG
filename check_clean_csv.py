#!/usr/bin/env python3
"""检查清理后的CSV文件是否为纯净轨迹数据"""

import pandas as pd
import os

def check_clean_csv():
    """检查清理后的CSV文件"""
    
    # 检查最新生成的CSV文件
    csv_file = 'scripts/drag_shoot_2v2/front_back_attack_results/front_back_attack_trajectory_20250830_114118.csv'
    
    print('🔍 检查清理后的CSV文件')
    print('=' * 60)
    
    if not os.path.exists(csv_file):
        print(f'❌ 文件不存在: {csv_file}')
        return
    
    try:
        df = pd.read_csv(csv_file)
        
        print(f'📁 文件: {os.path.basename(csv_file)}')
        print(f'📊 总行数: {len(df)}')
        print(f'📋 总列数: {len(df.columns)}')
        print()
        
        print('📋 所有列名:')
        for i, col in enumerate(df.columns, 1):
            print(f'  {i:2d}. {col}')
        
        print()
        
        # 检查是否包含动作标注列
        action_columns = ['Action_Type', 'Direction']
        has_action_columns = any(col in df.columns for col in action_columns)
        
        if has_action_columns:
            print('❌ 发现动作标注列:')
            for col in action_columns:
                if col in df.columns:
                    print(f'  - {col}')
            print('❌ 清理失败：仍包含动作标注列')
        else:
            print('✅ 确认：CSV文件不包含动作标注列 (Action_Type, Direction)')
            print('✅ 成功生成纯净轨迹数据')
        
        print()
        print('📊 数据样本 (前3行):')
        print(df.head(3))
        
        # 检查基础轨迹数据列
        expected_columns = ['Time_s', 'Agent_ID', 'X_m', 'Y_m', 'Z_m', 'Velocity_m_s', 'Heading_deg']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        print()
        if missing_columns:
            print(f'⚠️ 缺少基础轨迹数据列: {missing_columns}')
        else:
            print('✅ 包含所有基础轨迹数据列')
        
        return not has_action_columns
        
    except Exception as e:
        print(f'❌ 读取文件失败: {e}')
        return False

if __name__ == "__main__":
    success = check_clean_csv()
    print()
    print('=' * 60)
    if success:
        print('🎉 第一阶段清理工作完成！')
        print('✅ 成功移除动作标注系统，生成纯净轨迹数据')
    else:
        print('❌ 清理工作未完成，需要进一步修复')
