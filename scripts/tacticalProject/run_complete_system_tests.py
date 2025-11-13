#!/usr/bin/env python3
"""
完整智能战术系统测试脚本：运行10次仿真，验证所有功能
"""
import subprocess
import os
import sys
import time
from datetime import datetime

def run_single_test(test_num):
    """运行单次测试"""
    print(f"\n{'='*80}")
    print(f"🧪 测试 #{test_num}/10 - 完整智能战术系统")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # 运行仿真
    result = subprocess.run(
        [sys.executable, "scripts/tacticalProject/run_simulation.py"],
        cwd=os.getcwd(),
        capture_output=True,
        text=True
    )
    
    elapsed_time = time.time() - start_time
    
    # 检查输出中的关键信息
    output = result.stdout + result.stderr
    
    # 检查是否有错误
    has_error = "ERROR" in output or "Exception" in output or result.returncode != 0
    
    # 检查是否有战术选择
    has_tactic_selection = "战术选择完成" in output or "选定战术" in output
    
    # 检查是否有紧急中断
    has_emergency = "检测到导弹来袭" in output or "RWR告警" in output
    
    print(f"\n✅ 测试 #{test_num} 完成，耗时 {elapsed_time:.1f}秒")
    print(f"   返回码: {result.returncode}")
    print(f"   战术选择: {'✓' if has_tactic_selection else '✗'}")
    print(f"   紧急中断: {'✓' if has_emergency else '-'}")
    print(f"   错误: {'✗ 有错误' if has_error else '✓ 无错误'}")
    
    return {
        'success': result.returncode == 0 and not has_error,
        'has_tactic_selection': has_tactic_selection,
        'has_emergency': has_emergency,
        'elapsed_time': elapsed_time
    }

def main():
    """主函数"""
    num_tests = 10  # 运行10次测试
    
    print(f"{'='*80}")
    print(f"🚀 完整智能战术系统批量测试")
    print(f"{'='*80}")
    print(f"测试次数: {num_tests}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    results = []
    
    for i in range(1, num_tests + 1):
        result = run_single_test(i)
        results.append(result)
        
        # 每次测试之间暂停2秒
        if i < num_tests:
            time.sleep(2)
    
    # 统计结果
    success_count = sum(1 for r in results if r['success'])
    fail_count = num_tests - success_count
    tactic_selection_count = sum(1 for r in results if r['has_tactic_selection'])
    emergency_count = sum(1 for r in results if r['has_emergency'])
    avg_time = sum(r['elapsed_time'] for r in results) / num_tests
    
    print(f"\n{'='*80}")
    print(f"📊 批量测试完成")
    print(f"{'='*80}")
    print(f"总测试次数: {num_tests}")
    print(f"成功: {success_count} ({success_count/num_tests*100:.1f}%)")
    print(f"失败: {fail_count} ({fail_count/num_tests*100:.1f}%)")
    print(f"战术选择触发: {tactic_selection_count} ({tactic_selection_count/num_tests*100:.1f}%)")
    print(f"紧急中断触发: {emergency_count} ({emergency_count/num_tests*100:.1f}%)")
    print(f"平均耗时: {avg_time:.1f}秒")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # 返回成功率
    return success_count == num_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

