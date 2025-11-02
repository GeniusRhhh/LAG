#!/usr/bin/env python3
"""
机动时间偏差分析工具
分析现有验证结果中的时间偏差问题
"""

import os
import csv
import json
from datetime import datetime

def analyze_csv_duration(csv_path):
    """分析CSV文件的实际持续时间"""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return 0.0
            
            # 获取最后一行的时间
            last_time = float(rows[-1]['time_s'])
            return last_time
    except Exception as e:
        print(f"读取CSV文件错误: {e}")
        return 0.0

def get_expected_durations():
    """获取预期持续时间"""
    return {
        # 基础机动 - 根据你提供的表格
        "level_flight": 30.0,      # 平飞机动 30s
        "turn_level": 30.0,        # 转弯机动 30s  
        "pull_up": 20.0,           # 拉起机动 20s
        "dive": 20.0,              # 俯冲机动 20s
        "accelerate": 20.0,        # 加速机动 20s
        "decelerate": 20.0,        # 减速机动 20s
        "crank": 30.0,             # 战术偏置转向 30s
        "diagonal_flight": 30.0,   # 战术转弯机动 30s
        
        # 复合机动
        "turn_pull_up": 60.0,      # 复合机动通常更长
        "turn_dive": 60.0,
        "short_skate_tactical": 60.0,
    }

def analyze_latest_results():
    """分析最新的验证结果"""
    results_dir = "air_combat_results/4_4_3"
    
    # 找到最新的结果目录
    if not os.path.exists(results_dir):
        print(f"结果目录不存在: {results_dir}")
        return
    
    subdirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if not subdirs:
        print("没有找到结果子目录")
        return
    
    latest_dir = os.path.join(results_dir, sorted(subdirs)[-1])
    print(f"分析目录: {latest_dir}")
    
    # 读取summary文件
    summary_path = os.path.join(latest_dir, "summary_metrics.csv")
    if not os.path.exists(summary_path):
        print(f"Summary文件不存在: {summary_path}")
        return
    
    expected_durations = get_expected_durations()
    results = []
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['name']
            csv_path = row['csv_path']
            
            if not os.path.exists(csv_path):
                print(f"CSV文件不存在: {csv_path}")
                continue
            
            # 分析实际持续时间
            actual_duration = analyze_csv_duration(csv_path)
            expected_duration = expected_durations.get(name, 30.0)
            
            deviation = actual_duration - expected_duration
            deviation_percent = (deviation / expected_duration * 100) if expected_duration > 0 else 0
            
            result = {
                "name": name,
                "kind": row['kind'],
                "expected_duration": expected_duration,
                "actual_duration": actual_duration,
                "deviation": deviation,
                "deviation_percent": deviation_percent,
                "anomaly_cut_time": row.get('anomaly_cut_time_s', ''),
            }
            results.append(result)
    
    return results

def print_analysis_table(results):
    """打印分析表格"""
    print(f"\n{'='*100}")
    print("机动时间偏差分析表")
    print(f"{'='*100}")
    print(f"{'机动类型':<20} {'类别':<10} {'预期时间(s)':<12} {'实际时间(s)':<12} {'偏差(s)':<10} {'偏差(%)':<10} {'异常截断':<10}")
    print(f"{'-'*100}")
    
    for result in results:
        anomaly = "是" if result['anomaly_cut_time'] else "否"
        print(f"{result['name']:<20} {result['kind']:<10} {result['expected_duration']:<12.1f} "
              f"{result['actual_duration']:<12.1f} {result['deviation']:<10.1f} "
              f"{result['deviation_percent']:<10.1f} {anomaly:<10}")

def analyze_time_control_issues(results):
    """分析时间控制问题"""
    print(f"\n{'='*60}")
    print("时间控制问题分析")
    print(f"{'='*60}")
    
    # 统计偏差情况
    large_deviations = [r for r in results if abs(r['deviation_percent']) > 15]
    small_deviations = [r for r in results if abs(r['deviation_percent']) <= 15]
    
    print(f"总机动数量: {len(results)}")
    print(f"大偏差机动 (>15%): {len(large_deviations)}")
    print(f"小偏差机动 (≤15%): {len(small_deviations)}")
    
    if large_deviations:
        print(f"\n大偏差机动详情:")
        for r in large_deviations:
            print(f"  {r['name']}: {r['deviation_percent']:+.1f}% ({r['deviation']:+.1f}s)")
    
    # 分析偏差模式
    print(f"\n偏差模式分析:")
    positive_deviations = [r for r in results if r['deviation'] > 0]
    negative_deviations = [r for r in results if r['deviation'] < 0]
    
    print(f"时间超出预期: {len(positive_deviations)} 个机动")
    print(f"时间少于预期: {len(negative_deviations)} 个机动")
    
    if positive_deviations:
        avg_positive = sum(r['deviation'] for r in positive_deviations) / len(positive_deviations)
        print(f"平均超出时间: {avg_positive:.1f}s")
    
    if negative_deviations:
        avg_negative = sum(r['deviation'] for r in negative_deviations) / len(negative_deviations)
        print(f"平均缺少时间: {avg_negative:.1f}s")

def identify_root_causes():
    """识别根本原因"""
    print(f"\n{'='*60}")
    print("根本原因分析")
    print(f"{'='*60}")
    
    causes = [
        "1. 机动完成检测不完整",
        "   - 只有部分机动类型有明确的完成状态检测",
        "   - 转弯、拉起等机动缺乏完成判据",
        "",
        "2. 额外稳定时间影响",
        "   - 每个机动完成后额外飞行5秒稳定时间",
        "   - 这会系统性地增加总时间",
        "",
        "3. 时间控制逻辑不统一",
        "   - 不同机动类型使用不同的时间控制方法",
        "   - 基于状态检测 vs 基于固定时间",
        "",
        "4. 底层物理模型的响应延迟",
        "   - JSBSim物理仿真的惯性和延迟",
        "   - 控制指令到实际响应的时间差",
        "",
        "5. 机动参数与实际执行的差异",
        "   - 预期参数可能与实际可达参数不匹配",
        "   - 安全约束可能限制机动执行",
    ]
    
    for cause in causes:
        print(cause)

def propose_solutions():
    """提出解决方案"""
    print(f"\n{'='*60}")
    print("解决方案建议")
    print(f"{'='*60}")
    
    solutions = [
        "1. 统一机动完成检测机制",
        "   - 为所有机动类型定义明确的完成状态",
        "   - 基于目标达成度而非固定时间判断完成",
        "",
        "2. 优化时间控制策略",
        "   - 减少或取消额外稳定时间",
        "   - 使用自适应时间控制",
        "",
        "3. 改进机动参数校准",
        "   - 根据实际测试结果调整预期时间",
        "   - 考虑物理模型的响应特性",
        "",
        "4. 增加实时监控和调整",
        "   - 实时监控机动进度",
        "   - 动态调整控制参数",
        "",
        "5. 建立机动性能基准",
        "   - 为每种机动建立性能基准数据库",
        "   - 基于历史数据预测执行时间",
    ]
    
    for solution in solutions:
        print(solution)

def main():
    """主函数"""
    print("机动时间偏差分析工具")
    print("="*60)
    
    # 分析最新结果
    results = analyze_latest_results()
    if not results:
        print("没有找到有效的分析结果")
        return
    
    # 打印分析表格
    print_analysis_table(results)
    
    # 分析时间控制问题
    analyze_time_control_issues(results)
    
    # 识别根本原因
    identify_root_causes()
    
    # 提出解决方案
    propose_solutions()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"time_deviation_analysis_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n分析结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
