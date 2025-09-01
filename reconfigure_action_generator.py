#!/usr/bin/env python3
"""
重新配置基础动作数据生成器
按照重新确认的11种机动动作列表重新配置和生成数据
"""

import sys
sys.path.append('.')
import os
import shutil
from datetime import datetime

def complete_data_cleanup():
    """完全清理现有数据"""
    print("🧹 **完全清理现有数据**")
    print("=" * 70)
    
    base_dir = "scripts/drag_shoot_2v2/basic_action_data"
    
    if os.path.exists(base_dir):
        total_files = 0
        # 统计现有文件
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith(('.csv', '.acmi')):
                    total_files += 1
        
        print(f"发现现有数据文件: {total_files} 个")
        
        # 删除整个目录
        try:
            shutil.rmtree(base_dir)
            print(f"✅ 完全删除目录: {base_dir}")
        except Exception as e:
            print(f"删除目录失败: {e}")
            return False
        
        print(f"✅ 数据清理完成，删除了 {total_files} 个文件")
    else:
        print("数据目录不存在，无需清理")
    
    return True

def verify_cleanup():
    """验证清理结果"""
    print("\n🔍 **验证清理结果**")
    print("=" * 50)
    
    base_dir = "scripts/drag_shoot_2v2/basic_action_data"
    
    if not os.path.exists(base_dir):
        print("✅ 目录已完全清理")
        return True
    else:
        remaining_files = []
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith(('.csv', '.acmi')):
                    remaining_files.append(os.path.join(root, file))
        
        if remaining_files:
            print(f"❌ 仍有 {len(remaining_files)} 个文件未清理:")
            for file in remaining_files[:5]:  # 只显示前5个
                print(f"  {file}")
            if len(remaining_files) > 5:
                print(f"  ... 还有 {len(remaining_files) - 5} 个文件")
            return False
        else:
            print("✅ 目录已完全清理")
            return True

def show_new_action_list():
    """显示重新确认的动作列表"""
    print("\n📋 **重新确认的11种机动动作列表**")
    print("=" * 70)
    
    new_actions = [
        ("1", "平飞", "level_flight", "基础动作"),
        ("2", "加速", "accelerate", "基础动作"),
        ("3", "减速", "decelerate", "基础动作"),
        ("4", "左转", "turn_left", "基础动作"),
        ("5", "右转", "turn_right", "基础动作"),
        ("6", "爬升", "climb", "基础动作"),
        ("7", "左爬升", "climb_left", "组合机动"),
        ("8", "右爬升", "climb_right", "组合机动"),
        ("9", "俯冲", "dive", "基础动作"),
        ("10", "左俯冲", "dive_left", "组合机动"),
        ("11", "右俯冲", "dive_right", "组合机动")
    ]
    
    print(f"{'序号':<4} {'中文名称':<8} {'英文名称':<12} {'类型':<8}")
    print("-" * 50)
    
    basic_count = 0
    combo_count = 0
    
    for num, chinese, english, type_name in new_actions:
        print(f"{num:<4} {chinese:<8} {english:<12} {type_name:<8}")
        if type_name == "基础动作":
            basic_count += 1
        else:
            combo_count += 1
    
    print("-" * 50)
    print(f"总计: {len(new_actions)} 种动作")
    print(f"基础动作: {basic_count} 种")
    print(f"组合机动: {combo_count} 种")
    print(f"预期样本数: {len(new_actions)} × 10 = {len(new_actions) * 10} 个")

def show_removed_actions():
    """显示被移除的动作"""
    print("\n❌ **被移除的动作列表**")
    print("=" * 50)
    
    removed_actions = [
        "turn (分离为turn_left和turn_right)",
        "Crank (战术动作)",
        "tactical_crank (战术动作)",
        "tactical_climb (战术动作)",
        "tactical_dive (战术动作)",
        "notch_back (战术动作)",
        "short_skate (战术动作)"
    ]
    
    for action in removed_actions:
        print(f"  - {action}")
    
    print(f"\n移除动作数: {len(removed_actions)} 种")

def show_added_actions():
    """显示新增的动作"""
    print("\n✅ **新增的动作列表**")
    print("=" * 50)
    
    added_actions = [
        ("turn_left", "左转", "从turn动作分离"),
        ("turn_right", "右转", "从turn动作分离"),
        ("climb_left", "左爬升", "爬升+左转组合机动"),
        ("climb_right", "右爬升", "爬升+右转组合机动"),
        ("dive_left", "左俯冲", "俯冲+左转组合机动"),
        ("dive_right", "右俯冲", "俯冲+右转组合机动")
    ]
    
    for english, chinese, description in added_actions:
        print(f"  + {english:<12} ({chinese}) - {description}")
    
    print(f"\n新增动作数: {len(added_actions)} 种")

def main():
    """主函数"""
    print("🚀 **基础动作数据生成框架重新配置**")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("目标: 按照重新确认的11种机动动作列表重新配置框架")
    
    # 显示动作变更信息
    show_new_action_list()
    show_removed_actions()
    show_added_actions()
    
    # 完全清理现有数据
    print(f"\n" + "=" * 80)
    cleanup_success = complete_data_cleanup()
    
    if not cleanup_success:
        print("❌ 数据清理失败，请检查错误信息")
        return False
    
    # 验证清理结果
    cleanup_verified = verify_cleanup()
    
    if not cleanup_verified:
        print("❌ 清理验证失败，仍有残留文件")
        return False
    
    print(f"\n🎉 **数据清理完成！**")
    print("✅ 所有现有数据已完全清理")
    print("✅ 目录结构已重置")
    print("✅ 准备进行框架重新配置")
    
    print(f"\n📋 **下一步操作**:")
    print("1. 修改basic_action_data_generator.py中的动作配置")
    print("2. 实现组合机动动作的生成逻辑")
    print("3. 重新生成11种动作的数据集")
    print("4. 验证110个样本的质量")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 **框架重新配置准备完成！**")
    else:
        print(f"\n⚠️ 重新配置准备失败，请检查错误信息")
