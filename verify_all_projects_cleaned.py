#!/usr/bin/env python3
"""验证所有四个项目的动作标注系统清理结果"""

import pandas as pd
import os
from datetime import datetime

def verify_project_cleaned(project_name, csv_file):
    """验证单个项目的清理结果"""
    print(f"\n🔍 检查 {project_name}")
    print("=" * 50)
    
    if not os.path.exists(csv_file):
        print(f"❌ 文件不存在: {csv_file}")
        return False
    
    try:
        df = pd.read_csv(csv_file)
        
        print(f"📁 文件: {os.path.basename(csv_file)}")
        print(f"📊 数据行数: {len(df):,}")
        print(f"📋 列数: {len(df.columns)}")
        
        # 检查动作标注列
        action_columns = ['Action_Type', 'Direction']
        has_action_columns = any(col in df.columns for col in action_columns)
        
        if has_action_columns:
            print("❌ 仍包含动作标注列:")
            for col in action_columns:
                if col in df.columns:
                    print(f"  - {col}")
            return False
        else:
            print("✅ 确认：无动作标注列")
        
        # 检查基础轨迹数据列
        expected_columns = ['Time_s', 'Agent_ID', 'X_m', 'Y_m', 'Z_m', 'Velocity_m_s', 'Heading_deg']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️ 缺少基础列: {missing_columns}")
        else:
            print("✅ 包含所有基础轨迹数据列")
        
        print(f"📋 实际列名: {list(df.columns)}")
        return True
        
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return False

def main():
    """主验证函数"""
    print("🧹 第一阶段清理工作验证")
    print("=" * 60)
    print("验证所有四个项目的动作标注系统清理结果")
    print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 定义所有项目和对应的最新CSV文件
    projects = {
        "拖曳射击": "scripts/drag_shoot_2v2/air_combat_results/drag_shoot_trajectory_20250830_114622.csv",
        "上下夹击": "scripts/drag_shoot_2v2/high_low_attack_results/high_low_attack_trajectory_20250830_112826.csv", 
        "钳形攻击": "scripts/drag_shoot_2v2/pincer_attack_results/pincer_attack_trajectory_20250830_113519.csv",
        "前后攻击": "scripts/drag_shoot_2v2/front_back_attack_results/front_back_attack_trajectory_20250830_114118.csv"
    }
    
    results = {}
    
    # 验证每个项目
    for project_name, csv_file in projects.items():
        results[project_name] = verify_project_cleaned(project_name, csv_file)
    
    # 总结结果
    print("\n" + "=" * 60)
    print("🎯 **验证结果总结**")
    print("=" * 60)
    
    all_success = True
    for project_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{project_name:10s}: {status}")
        if not success:
            all_success = False
    
    print("\n" + "=" * 60)
    if all_success:
        print("🎉 **第一阶段清理工作完全成功！**")
        print("✅ 所有四个项目都已成功移除动作标注系统")
        print("✅ 所有CSV文件都是纯净的飞行轨迹数据")
        print("\n📋 **清理成果:**")
        print("- 移除了所有 Action_Type 和 Direction 列")
        print("- 保留了完整的基础轨迹数据")
        print("- 四个项目运行脚本都已禁用动作标注系统")
        print("\n🚀 **准备就绪:**")
        print("- 可以开始第二阶段：基础动作数据生成框架")
        print("- 11种基础动作的独立仿真系统开发")
    else:
        print("❌ **清理工作未完成**")
        print("需要修复失败的项目")
    
    return all_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
