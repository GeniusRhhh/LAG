"""
批量测试脚本 - 完整智能战术选择系统
运行10次仿真测试，收集统计数据
"""
import subprocess
import os
import re
from datetime import datetime
from collections import Counter
import json

def run_single_test(test_num):
    """运行单次测试"""
    print(f"\n{'='*80}")
    print(f"开始测试 #{test_num}/10")
    print(f"{'='*80}")
    
    # 运行仿真
    result = subprocess.run(
        ['.venv/Scripts/python.exe', 'scripts/tacticalProject/run_simulation.py'],
        capture_output=True,
        text=True,
        cwd='.'
    )
    
    return result.returncode == 0

def analyze_log_file(log_file):
    """分析日志文件，提取关键信息"""
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取战术选择
    tactics = re.findall(r'选定战术: (\w+)', content)
    
    # 提取紧急中断
    missile_evasions = len(re.findall(r'导弹来袭.*TACTICAL_EVASION', content))
    high_threat_turns = len(re.findall(r'RWR告警.*TACTICAL_TURN', content))
    
    # 提取态势评估
    situations = re.findall(r'态势评估: (\w+)', content)
    
    # 提取威胁等级
    threat_levels = re.findall(r'威胁等级: ([\d.]+)', content)
    
    # 检查坠毁
    crashes = len(re.findall(r'坠毁|crashed|killed', content))
    
    # 检查边界违规
    boundary_violations = len(re.findall(r'CAP边界违规', content))
    
    return {
        'tactics': tactics,
        'missile_evasions': missile_evasions,
        'high_threat_turns': high_threat_turns,
        'situations': situations,
        'threat_levels': threat_levels,
        'crashes': crashes,
        'boundary_violations': boundary_violations
    }

def main():
    """主函数"""
    print("="*80)
    print("完整智能战术选择系统 - 批量测试")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试次数: 10")
    print("="*80)
    
    results = []
    all_tactics = []
    total_crashes = 0
    total_boundary_violations = 0
    
    # 运行10次测试
    for i in range(1, 11):
        success = run_single_test(i)
        
        if success:
            # 查找最新的日志文件
            log_dir = 'scripts/tacticalProject/tactical_simulation_results'
            log_files = [f for f in os.listdir(log_dir) if f.startswith('front_back_simulation_') and f.endswith('.log')]
            if log_files:
                latest_log = max([os.path.join(log_dir, f) for f in log_files], key=os.path.getmtime)
                
                # 分析日志
                analysis = analyze_log_file(latest_log)
                if analysis:
                    results.append(analysis)
                    all_tactics.extend(analysis['tactics'])
                    total_crashes += analysis['crashes']
                    total_boundary_violations += analysis['boundary_violations']
                    
                    print(f"✅ 测试 #{i} 完成")
                    print(f"   战术触发: {Counter(analysis['tactics'])}")
                else:
                    print(f"⚠️ 测试 #{i} 日志分析失败")
        else:
            print(f"❌ 测试 #{i} 运行失败")
    
    # 生成统计报告
    print("\n" + "="*80)
    print("测试统计报告")
    print("="*80)
    
    print(f"\n总测试次数: {len(results)}")
    print(f"总坠毁次数: {total_crashes}")
    print(f"总边界违规次数: {total_boundary_violations}")
    
    print(f"\n战术触发统计:")
    tactic_counts = Counter(all_tactics)
    for tactic, count in tactic_counts.most_common():
        percentage = (count / len(results)) * 100 if results else 0
        print(f"  {tactic}: {count}次 ({percentage:.1f}%)")
    
    # 保存统计结果
    report_file = f'scripts/tacticalProject/tactical_simulation_results/batch_test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_tests': len(results),
            'total_crashes': total_crashes,
            'total_boundary_violations': total_boundary_violations,
            'tactic_counts': dict(tactic_counts),
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细报告已保存: {report_file}")
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()

