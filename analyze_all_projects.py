#!/usr/bin/env python3
"""全面分析所有四个战术项目的动作标注数据质量"""

import pandas as pd
from collections import Counter
import os

def analyze_project_data(project_name, csv_file):
    """分析单个项目的数据质量"""
    
    print(f"\n{'='*60}")
    print(f"📊 {project_name} 项目数据分析")
    print(f"{'='*60}")
    
    if not os.path.exists(csv_file):
        print(f"❌ 文件不存在: {csv_file}")
        return None
    
    try:
        df = pd.read_csv(csv_file)
        
        print(f"📈 基本统计:")
        print(f"  数据点总数: {len(df)}")
        print(f"  智能体数量: {df['Agent_ID'].nunique()}")
        print(f"  时间范围: {df['Time_s'].min():.1f}s - {df['Time_s'].max():.1f}s")
        
        # 检查是否有动作标注列
        if 'Action_Type' in df.columns:
            print(f"\n🎯 动作标注统计:")
            action_counts = Counter(df['Action_Type'].dropna())
            for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {action}: {count}次")
            
            if 'Direction' in df.columns:
                print(f"\n🧭 方向标注统计:")
                direction_counts = Counter(df['Direction'].dropna())
                for direction, count in sorted(direction_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {direction}: {count}次")
                
                # 分析Short Skate机动
                short_skate_data = df[df['Action_Type'] == 'Short skate']
                if len(short_skate_data) > 0:
                    print(f"\n⚡ Short Skate机动分析:")
                    print(f"  总数: {len(short_skate_data)}次")
                    
                    # 按智能体分析
                    for agent_id in short_skate_data['Agent_ID'].unique():
                        agent_short_skate = short_skate_data[short_skate_data['Agent_ID'] == agent_id]
                        agent_directions = Counter(agent_short_skate['Direction'])
                        print(f"  {agent_id}: {len(agent_short_skate)}次", end="")
                        if agent_directions:
                            direction_str = ", ".join([f"{d}:{c}" for d, c in agent_directions.items()])
                            print(f" ({direction_str})")
                        else:
                            print()
                else:
                    print(f"\n⚡ Short Skate机动: 未检测到")
            else:
                print(f"\n❌ 缺少Direction列")
        else:
            print(f"\n❌ 缺少Action_Type列 - 可能未集成动作标注系统")
        
        return {
            'total_points': len(df),
            'agents': df['Agent_ID'].nunique(),
            'has_action_annotation': 'Action_Type' in df.columns,
            'has_direction_annotation': 'Direction' in df.columns,
            'action_counts': Counter(df['Action_Type'].dropna()) if 'Action_Type' in df.columns else {},
            'direction_counts': Counter(df['Direction'].dropna()) if 'Direction' in df.columns else {}
        }
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None

def main():
    """主函数 - 分析所有四个战术项目"""
    
    print("🎯 全面验证所有四个战术项目的动作标注系统")
    print("=" * 80)
    
    # 定义项目和对应的最新CSV文件
    projects = {
        "拖曳射击": "scripts/drag_shoot_2v2/air_combat_results/drag_shoot_trajectory_20250829_234339.csv",
        "上下夹击": "scripts/drag_shoot_2v2/high_low_attack_results/high_low_attack_trajectory_20250829_222739.csv",
        "钳形攻击": "scripts/drag_shoot_2v2/pincer_attack_results/pincer_attack_trajectory_20250829_235128.csv",
        "前后攻击": "scripts/drag_shoot_2v2/front_back_attack_results/front_back_attack_trajectory_20250829_235819.csv"
    }
    
    results = {}
    
    # 分析每个项目
    for project_name, csv_file in projects.items():
        results[project_name] = analyze_project_data(project_name, csv_file)
    
    # 生成综合对比报告
    print(f"\n{'='*80}")
    print("📋 综合对比报告")
    print(f"{'='*80}")
    
    print(f"\n🔍 系统集成状态:")
    for project_name, result in results.items():
        if result:
            status = "✅ 已集成" if result['has_action_annotation'] else "❌ 未集成"
            direction_status = "✅ 完整" if result['has_direction_annotation'] else "❌ 缺失"
            print(f"  {project_name:8s}: 动作标注 {status}, 方向标注 {direction_status}")
        else:
            print(f"  {project_name:8s}: ❌ 数据文件缺失")
    
    print(f"\n📊 Short Skate机动对比:")
    for project_name, result in results.items():
        if result and result['has_action_annotation']:
            short_skate_count = result['action_counts'].get('Short skate', 0)
            print(f"  {project_name:8s}: {short_skate_count:4d}次")
        else:
            print(f"  {project_name:8s}: N/A")
    
    print(f"\n🎯 动作标注多样性对比:")
    for project_name, result in results.items():
        if result and result['has_action_annotation']:
            action_types = len(result['action_counts'])
            print(f"  {project_name:8s}: {action_types}种动作类型")
        else:
            print(f"  {project_name:8s}: N/A")
    
    print(f"\n🧭 方向标注多样性对比:")
    for project_name, result in results.items():
        if result and result['has_direction_annotation']:
            direction_types = len(result['direction_counts'])
            print(f"  {project_name:8s}: {direction_types}种方向类型")
        else:
            print(f"  {project_name:8s}: N/A")
    
    # 系统评估
    print(f"\n🏆 系统集成评估:")
    integrated_count = sum(1 for r in results.values() if r and r['has_action_annotation'])
    total_projects = len([r for r in results.values() if r is not None])
    
    if integrated_count == total_projects:
        print(f"  ✅ 完美集成: {integrated_count}/{total_projects} 个项目已成功集成动作标注系统")
    elif integrated_count > total_projects // 2:
        print(f"  ⚠️ 部分集成: {integrated_count}/{total_projects} 个项目已集成动作标注系统")
    else:
        print(f"  ❌ 集成不足: 仅 {integrated_count}/{total_projects} 个项目已集成动作标注系统")
    
    print(f"\n{'='*80}")
    print("分析完成!")

if __name__ == "__main__":
    main()
