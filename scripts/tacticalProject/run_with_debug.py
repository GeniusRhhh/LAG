#!/usr/bin/env python3
"""
一键运行 - 敌方AI下降异常诊断仿真
直接运行本脚本即可：启用调试 + 运行仿真 + 生成日志
"""

import os
import sys
import subprocess

# 设置环境变量（启用调试）
os.environ['ENEMY_DESCENT_DEBUG'] = '1'
os.environ['CAP_ROOTCAUSE_TRACE'] = '1'
os.environ['CAP_DEBUG_PRINT'] = '1'
os.environ['CAP_CONTROL_DEBUG'] = '1'

print("=" * 80)
print("【敌方AI下降异常诊断 - 一键运行】")
print("=" * 80)
print()
print("[1/3] 启用调试...")
print(f"  ✓ ENEMY_DESCENT_DEBUG = 1")
print(f"  ✓ CAP_ROOTCAUSE_TRACE = 1")
print()

# 创建输出目录
os.makedirs('cap_results', exist_ok=True)
print("[2/3] 准备输出目录...")
print(f"  ✓ cap_results/")
print()

print("[3/3] 运行仿真...")
print()
print("=" * 80)

# 运行仿真
try:
    # 文件在 cap/ 子目录中
    result = subprocess.run(
        [sys.executable, 'cap/run_cap_simulation_native.py'],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    print()
    print("=" * 80)
    
    if result.returncode == 0:
        print("✅ 仿真完成！")
        print()
        print("日志位置:")
        print("  📄 cap_results/descent_debug.log           (★ 异常下降信息)")
        print("  📄 cap_results/descent_diagnosis.log       (诊断信息)")
        print()
        print("查看日志:")
        print("  type cap_results\\descent_debug.log")
        print()
        print("搜索异常下降:")
        print("  Select-String 'is_descent=True' cap_results\\descent_debug.log")
        print("  Select-String 'B0200' cap_results\\descent_debug.log")
    else:
        print("❌ 仿真失败 (exit code: {})".format(result.returncode))
        
except Exception as e:
    print(f"❌ 错误: {e}")
    sys.exit(1)
