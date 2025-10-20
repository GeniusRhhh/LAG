#!/usr/bin/env python3
"""
敌方AI坠机问题自动修复脚本

修复unified_enemy_tactical_ai.py中的致命高度指令错误
"""

import os
import re
import shutil
from datetime import datetime

def backup_file(filepath):
    """备份原文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.backup_{timestamp}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ 已备份原文件: {backup_path}")
    return backup_path

def fix_altitude_commands(filepath):
    """修复高度指令错误"""
    print(f"\n{'='*80}")
    print(f"🔧 开始修复文件: {filepath}")
    print(f"{'='*80}\n")
    
    # 读取文件
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    fixes_applied = 0
    
    # 修复1: altitude_cmd = 0 # 爬升 → altitude_cmd = 9 # 温和爬升150m
    pattern1 = r'altitude_cmd\s*=\s*0\s*#\s*爬升'
    replacement1 = 'altitude_cmd = 9  # 温和爬升150m（已修复：原错误用0导致极度俯冲！）'
    new_content, count1 = re.subn(pattern1, replacement1, content)
    if count1 > 0:
        print(f"✅ 修复 {count1} 处致命错误: 'altitude_cmd = 0 # 爬升' → 'altitude_cmd = 9'")
        content = new_content
        fixes_applied += count1
    
    # 修复2: altitude_cmd = -1  # 温和俯冲 → altitude_cmd = 6  # 轻微俯冲50m
    pattern2 = r'altitude_cmd\s*=\s*-1\s*#\s*温和俯冲'
    replacement2 = 'altitude_cmd = 6  # 轻微俯冲50m（已修复：原错误用-1实际是极度爬升）'
    new_content, count2 = re.subn(pattern2, replacement2, content)
    if count2 > 0:
        print(f"✅ 修复 {count2} 处概念错误: 'altitude_cmd = -1 # 温和俯冲' → 'altitude_cmd = 6'")
        content = new_content
        fixes_applied += count2
    
    # 修复3: altitude_change = 1  # 强制爬升 → altitude_change = 9
    pattern3 = r'altitude_change\s*=\s*1\s*#\s*强制爬升'
    replacement3 = 'altitude_change = 9  # 强制温和爬升150m（已修复：原错误用1导致严重俯冲）'
    new_content, count3 = re.subn(pattern3, replacement3, content)
    if count3 > 0:
        print(f"✅ 修复 {count3} 处致命错误: 'altitude_change = 1 # 强制爬升' → 'altitude_change = 9'")
        content = new_content
        fixes_applied += count3
    
    # 修复4: altitude_change = random.choice([0, 1])  → altitude_change = random.choice([7, 8, 9])
    pattern4 = r'altitude_change\s*=\s*random\.choice\(\[0,\s*1\]\)\s*#\s*水平或爬升'
    replacement4 = 'altitude_change = random.choice([7, 8, 9])  # 保持高度或温和爬升（已修复：原错误用0,1都是俯冲）'
    new_content, count4 = re.subn(pattern4, replacement4, content)
    if count4 > 0:
        print(f"✅ 修复 {count4} 处致命错误: 'random.choice([0, 1])' → 'random.choice([7, 8, 9])'")
        content = new_content
        fixes_applied += count4
    
    # 修复5: altitude_cmd = random.choice([8, 9])  # 保持高度或轻微下降
    pattern5 = r'altitude_cmd\s*=\s*random\.choice\(\[8,\s*9\]\)\s*#\s*保持高度或轻微下降'
    replacement5 = 'altitude_cmd = random.choice([7, 8, 9])  # 保持高度或轻微/温和爬升（已修复注释）'
    new_content, count5 = re.subn(pattern5, replacement5, content)
    if count5 > 0:
        print(f"✅ 修复 {count5} 处注释错误: '保持高度或轻微下降' → '保持高度或轻微/温和爬升'")
        content = new_content
        fixes_applied += count5
    
    if fixes_applied == 0:
        print("⚠️  未找到需要修复的代码模式，文件可能已经修复过或格式不匹配")
        return False
    
    # 写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n{'='*80}")
    print(f"✅ 修复完成！共应用 {fixes_applied} 处修复")
    print(f"{'='*80}\n")
    
    return True

def verify_fixes(filepath):
    """验证修复结果"""
    print(f"\n{'='*80}")
    print("🔍 验证修复结果")
    print(f"{'='*80}\n")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否还有危险的高度指令
    dangerous_patterns = [
        (r'altitude_cmd\s*=\s*0([^0-9]|$)', "altitude_cmd = 0（极度俯冲1500m）"),
        (r'altitude_cmd\s*=\s*1([^0-9]|$)', "altitude_cmd = 1（严重俯冲1000m）"),
        (r'altitude_cmd\s*=\s*2([^0-9]|$)', "altitude_cmd = 2（大幅俯冲750m）"),
        (r'altitude_change\s*=\s*0([^0-9]|$)', "altitude_change = 0（极度俯冲1500m）"),
        (r'altitude_change\s*=\s*1([^0-9]|$)', "altitude_change = 1（严重俯冲1000m）"),
    ]
    
    warnings = []
    for pattern, desc in dangerous_patterns:
        matches = re.findall(pattern, content)
        if matches:
            warnings.append(f"⚠️  仍存在危险模式: {desc}")
    
    if warnings:
        print("⚠️  警告：发现以下潜在问题：")
        for warning in warnings:
            print(f"   {warning}")
        print("\n请手动检查这些位置是否确实需要大幅俯冲\n")
    else:
        print("✅ 验证通过！未发现明显的危险高度指令\n")
    
    # 统计正确的高度指令使用情况
    safe_patterns = [
        (r'altitude_cmd\s*=\s*7', "保持高度（索引7）"),
        (r'altitude_cmd\s*=\s*[89]', "温和爬升（索引8-9）"),
        (r'altitude_cmd\s*=\s*1[0-4]', "中等/大幅爬升（索引10-14）"),
        (r'altitude_cmd\s*=\s*[456]', "温和/小幅俯冲（索引4-6）"),
    ]
    
    print("📊 高度指令使用统计：")
    for pattern, desc in safe_patterns:
        count = len(re.findall(pattern, content))
        if count > 0:
            print(f"   {desc}: {count} 处")

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🚨 敌方AI坠机问题自动修复脚本")
    print("="*80)
    print()
    print("本脚本将修复 unified_enemy_tactical_ai.py 中的致命高度指令错误")
    print()
    print("修复内容：")
    print("  1. altitude_cmd = 0（原注释：爬升）→ 9（温和爬升150m）")
    print("  2. altitude_cmd = -1（原注释：俯冲）→ 6（轻微俯冲50m）")
    print("  3. altitude_change = 1（原注释：爬升）→ 9（温和爬升150m）")
    print("  4. random.choice([0, 1])（原注释：水平或爬升）→ [7, 8, 9]")
    print("  5. 修正错误的注释")
    print()
    
    # 获取文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_file = os.path.join(script_dir, "unified_enemy_tactical_ai.py")
    
    if not os.path.exists(target_file):
        print(f"❌ 错误：找不到文件 {target_file}")
        return 1
    
    print(f"目标文件: {target_file}")
    print()
    
    # 询问用户确认
    response = input("是否继续修复？(y/n): ")
    if response.lower() != 'y':
        print("已取消修复")
        return 0
    
    # 备份原文件
    backup_path = backup_file(target_file)
    
    # 应用修复
    success = fix_altitude_commands(target_file)
    
    if not success:
        print("\n❌ 修复失败或无需修复")
        # 恢复备份
        shutil.copy2(backup_path, target_file)
        print(f"已恢复原文件")
        return 1
    
    # 验证修复
    verify_fixes(target_file)
    
    print("="*80)
    print("✅ 修复完成！")
    print("="*80)
    print()
    print("下一步：")
    print("  1. 查看修复后的代码并确认")
    print("  2. 运行测试: python run_pincer_attack.py")
    print("  3. 观察敌方AI是否还会坠毁")
    print()
    print(f"备份文件位置: {backup_path}")
    print("如需恢复，请手动复制备份文件")
    print()
    
    return 0

if __name__ == "__main__":
    exit(main())
