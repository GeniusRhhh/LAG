#!/usr/bin/env python3
"""
批量测试脚本：运行多次仿真，收集数据并生成报告
"""
import subprocess
import os
import sys
import time
from datetime import datetime

def run_single_test(test_num):
    """运行单次测试"""
    print(f"\n{'='*80}")
    print(f"开始测试 #{test_num}")
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
    
    print(f"\n测试 #{test_num} 完成，耗时 {elapsed_time:.1f}秒")
    print(f"返回码: {result.returncode}")
    
    return result.returncode == 0

def main():
    """主函数"""
    num_tests = 15  # 运行15次测试
    
    print(f"开始批量测试：共 {num_tests} 次")
    print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success_count = 0
    fail_count = 0
    
    for i in range(1, num_tests + 1):
        success = run_single_test(i)
        if success:
            success_count += 1
        else:
            fail_count += 1
        
        # 每次测试之间暂停2秒
        if i < num_tests:
            time.sleep(2)
    
    print(f"\n{'='*80}")
    print(f"批量测试完成")
    print(f"{'='*80}")
    print(f"总测试次数: {num_tests}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

