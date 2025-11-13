"""
快速测试脚本 - 运行3次测试验证系统
"""
import subprocess
import os
import re
from datetime import datetime
from collections import Counter

def run_single_test(test_num):
    """运行单次测试"""
    print(f"\n{'='*80}")
    print(f"开始测试 #{test_num}/3")
    print(f"{'='*80}")
    
    result = subprocess.run(
        ['.venv/Scripts/python.exe', 'scripts/tacticalProject/run_simulation.py'],
        capture_output=False,  # 显示输出
        cwd='.'
    )
    
    return result.returncode == 0

def analyze_latest_log():
    """分析最新的日志文件"""
    log_dir = 'scripts/tacticalProject/tactical_simulation_results'
    log_files = [f for f in os.listdir(log_dir) if f.startswith('front_back_simulation_') and f.endswith('.log')]
    if not log_files:
        return None
    
    latest_log = max([os.path.join(log_dir, f) for f in log_files], key=os.path.getmtime)
    
    with open(latest_log, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取战术选择
    tactics = re.findall(r'选定战术: (\w+)', content)
    
    # 提取决策表查询
    decision_queries = re.findall(r'\[决策表\] (\w+)查询:', content)
    
    # 提取战术适应度评分
    fitness_scores = re.findall(r'战术 (\w+): 适应度 = ([\d.]+)', content)
    
    # 检查紧急中断
    missile_detections = len(re.findall(r'检测到来袭导弹', content))
    rwr_warnings = len(re.findall(r'RWR告警等级', content))
    
    # 检查坠毁和边界违规
    crashes = len(re.findall(r'坠毁|crashed', content))
    boundary_violations = len(re.findall(r'CAP边界违规', content))
    
    return {
        'tactics': tactics,
        'decision_queries': decision_queries,
        'fitness_scores': fitness_scores,
        'missile_detections': missile_detections,
        'rwr_warnings': rwr_warnings,
        'crashes': crashes,
        'boundary_violations': boundary_violations,
        'log_file': latest_log
    }

def main():
    """主函数"""
    print("="*80)
    print("完整智能战术选择系统 - 快速验证测试 (3次)")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    results = []
    
    # 运行3次测试
    for i in range(1, 4):
        success = run_single_test(i)
        
        if success:
            analysis = analyze_latest_log()
            if analysis:
                results.append(analysis)
                print(f"\n✅ 测试 #{i} 完成")
                print(f"   日志文件: {analysis['log_file']}")
                print(f"   战术触发: {Counter(analysis['tactics'])}")
                print(f"   决策表查询: {Counter(analysis['decision_queries'])}")
                print(f"   坠毁: {analysis['crashes']}, 边界违规: {analysis['boundary_violations']}")
            else:
                print(f"⚠️ 测试 #{i} 日志分析失败")
        else:
            print(f"❌ 测试 #{i} 运行失败")
    
    # 汇总统计
    print("\n" + "="*80)
    print("测试汇总")
    print("="*80)
    
    all_tactics = []
    total_crashes = 0
    total_boundary_violations = 0
    
    for r in results:
        all_tactics.extend(r['tactics'])
        total_crashes += r['crashes']
        total_boundary_violations += r['boundary_violations']
    
    print(f"\n总测试次数: {len(results)}")
    print(f"总坠毁次数: {total_crashes}")
    print(f"总边界违规次数: {total_boundary_violations}")
    
    print(f"\n战术触发统计:")
    tactic_counts = Counter(all_tactics)
    for tactic, count in tactic_counts.most_common():
        print(f"  {tactic}: {count}次")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()

