"""
专项测试脚本 - 测试紧急中断机制
测试目标：
1. TACTICAL_EVASION在导弹来袭时触发
2. TACTICAL_TURN在高威胁时触发
3. 所有7种战术模板都能触发
"""

import os
import sys
import logging
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, '..', '..'))

from run_simulation import run_simulation


def setup_logging():
    """设置日志"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f'tactical_simulation_results/specialized_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
        ]
    )


def analyze_log_file(log_file):
    """分析日志文件，统计战术触发情况"""
    tactics_count = {
        'SIDE_BY_SIDE': 0,
        'PINCER_ATTACK': 0,
        'HIGH_LOW_ATTACK': 0,
        'DRAG_SHOOT': 0,
        'SEQUENTIAL_ATTACK': 0,
        'TACTICAL_EVASION': 0,
        'TACTICAL_TURN': 0
    }
    
    emergency_interrupts = {
        'missile_detected': 0,
        'rwr_high': 0
    }
    
    crashes = 0
    boundary_violations = 0
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 统计战术触发
                for tactic in tactics_count.keys():
                    if f'选定战术: {tactic}' in line or f'战术模板: {tactic}' in line:
                        tactics_count[tactic] += 1
                
                # 统计紧急中断
                if '导弹威胁' in line or '导弹来袭' in line:
                    emergency_interrupts['missile_detected'] += 1
                if 'RWR-4级' in line or 'RWR-5级' in line:
                    emergency_interrupts['rwr_high'] += 1
                
                # 统计坠毁和边界违规
                if '坠毁' in line or 'crashed' in line:
                    crashes += 1
                if '边界违规' in line or 'boundary violation' in line:
                    boundary_violations += 1
    
    except Exception as e:
        logging.error(f"分析日志文件失败: {e}")
    
    return tactics_count, emergency_interrupts, crashes, boundary_violations


def run_specialized_tests(num_tests=5):
    """
    运行专项测试
    
    Args:
        num_tests: 测试次数
    """
    setup_logging()
    
    logging.info("=" * 80)
    logging.info("🧪 开始专项测试 - 紧急中断机制验证")
    logging.info("=" * 80)
    
    total_tactics = {
        'SIDE_BY_SIDE': 0,
        'PINCER_ATTACK': 0,
        'HIGH_LOW_ATTACK': 0,
        'DRAG_SHOOT': 0,
        'SEQUENTIAL_ATTACK': 0,
        'TACTICAL_EVASION': 0,
        'TACTICAL_TURN': 0
    }
    
    total_interrupts = {
        'missile_detected': 0,
        'rwr_high': 0
    }
    
    total_crashes = 0
    total_violations = 0
    
    for i in range(num_tests):
        logging.info(f"\n{'='*80}")
        logging.info(f"🧪 测试 {i+1}/{num_tests}")
        logging.info(f"{'='*80}")
        
        # 运行仿真
        log_file = run_simulation()
        
        if log_file:
            # 分析日志
            tactics, interrupts, crashes, violations = analyze_log_file(log_file)
            
            # 累加统计
            for tactic, count in tactics.items():
                total_tactics[tactic] += count
            
            for interrupt_type, count in interrupts.items():
                total_interrupts[interrupt_type] += count
            
            total_crashes += crashes
            total_violations += violations
            
            # 打印本次测试结果
            logging.info(f"\n📊 测试 {i+1} 结果:")
            logging.info(f"  战术触发: {tactics}")
            logging.info(f"  紧急中断: {interrupts}")
            logging.info(f"  坠毁: {crashes}, 边界违规: {violations}")
    
    # 打印总结
    logging.info("\n" + "=" * 80)
    logging.info("📊 专项测试总结")
    logging.info("=" * 80)
    logging.info(f"总测试次数: {num_tests}")
    logging.info(f"总坠毁次数: {total_crashes}")
    logging.info(f"总边界违规次数: {total_violations}")
    logging.info("\n战术触发统计:")
    for tactic, count in total_tactics.items():
        logging.info(f"  {tactic}: {count}次")
    
    logging.info("\n紧急中断统计:")
    logging.info(f"  导弹检测: {total_interrupts['missile_detected']}次")
    logging.info(f"  高RWR告警: {total_interrupts['rwr_high']}次")
    
    # 验证清单
    logging.info("\n✅ 验证清单:")
    triggered_tactics = sum(1 for count in total_tactics.values() if count > 0)
    logging.info(f"  [ {'✓' if triggered_tactics == 7 else '✗'} ] 7/7战术模板都能触发 (当前: {triggered_tactics}/7)")
    logging.info(f"  [ {'✓' if total_tactics['TACTICAL_EVASION'] > 0 else '✗'} ] TACTICAL_EVASION能触发 (触发{total_tactics['TACTICAL_EVASION']}次)")
    logging.info(f"  [ {'✓' if total_tactics['TACTICAL_TURN'] > 0 else '✗'} ] TACTICAL_TURN能触发 (触发{total_tactics['TACTICAL_TURN']}次)")
    logging.info(f"  [ {'✓' if total_crashes == 0 else '✗'} ] 零坠毁率 (坠毁{total_crashes}次)")
    logging.info(f"  [ {'✓' if total_violations == 0 else '✗'} ] 零边界违规 (违规{total_violations}次)")


if __name__ == '__main__':
    run_specialized_tests(num_tests=5)

