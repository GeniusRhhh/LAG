#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版战术仿真测试脚本 - 支持战术多样性和完整性测试

功能特点：
1. 支持随机初始战术选择
2. 支持不同意图组合测试
3. 完整的两轮攻击验证
4. 清晰的战术切换日志
"""

import os
import sys
import random
import logging
import argparse
from pathlib import Path

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s.%(msecs)03d] %(levelname)-7s - %(message)s',
    datefmt='%H:%M:%S'
)

# 禁用JSBSim的冗余输出
os.environ['JSBSIM_DEBUG'] = '0'

def run_tactical_test(
    our_intent='CONSERVATIVE_CLEAR',
    initial_tactic=None,
    enemy_mode='MIXED',
    max_steps=1650,
    seed=None
):
    """
    运行战术测试
    
    Args:
        our_intent: 我方意图 (AGGRESSIVE_CLEAR/CONSERVATIVE_CLEAR/DEFENSIVE)
        initial_tactic: 初始战术 (None表示自动选择)
        enemy_mode: 敌方模式 (AGGRESSIVE/DEFENSIVE/MIXED)
        max_steps: 最大仿真步数
        seed: 随机种子
    """
    
    # 设置随机种子
    if seed is not None:
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)
    
    # 导入必要的模块
    from envs.JSBSim.envs import MultipleCombatEnv
    from tactical_task import TacticalTask
    
    try:
        # 创建环境
        env = MultipleCombatEnv('tactical_bvr')
        env.max_steps = max_steps
        
        # 创建战术任务（自动选择模式）
        tactical_task = TacticalTask(env.config, my_intent_type=our_intent)
        env.task = tactical_task
        
        # 如果指定了初始战术，强制设置
        if initial_tactic:
            tactical_task.selected_tactic = initial_tactic
            logging.info(f"🎯 强制初始战术: {initial_tactic}")
        
        # 设置敌方模式
        if hasattr(env, 'enemy_ai'):
            if enemy_mode == 'AGGRESSIVE':
                env.enemy_ai.force_tactical_mode('B0100', 'aggressive')
                env.enemy_ai.force_tactical_mode('B0200', 'aggressive')
            elif enemy_mode == 'DEFENSIVE':
                env.enemy_ai.force_tactical_mode('B0100', 'defensive')
                env.enemy_ai.force_tactical_mode('B0200', 'defensive')
            # MIXED模式让AI自动决定
        
        # 重置环境
        obs = env.reset()
        
        # 仿真主循环
        print("\n" + "="*80)
        print("🎬 战术多样性测试开始")
        print(f"   我方意图: {our_intent}")
        print(f"   初始战术: {initial_tactic or '自动选择'}")
        print(f"   敌方模式: {enemy_mode}")
        print("="*80 + "\n")
        
        # 运行状态跟踪
        done = False
        step_count = 0
        tactic_changes = []
        last_tactic = tactical_task.selected_tactic
        enemy_casualties = 0
        our_casualties = 0
        
        # 主循环
        while not done and step_count < max_steps:
            # 获取动作
            actions = tactical_task.get_actions(env)
            
            # 执行动作
            obs, rewards, dones, info = env.step(actions)
            
            # 检查战术变化
            current_tactic = tactical_task.selected_tactic
            if current_tactic != last_tactic:
                time_s = step_count * env.time_interval
                tactic_changes.append({
                    'time': time_s,
                    'from': last_tactic,
                    'to': current_tactic
                })
                print(f"\n🔄 [战术切换] {time_s:.1f}s: {last_tactic} → {current_tactic}\n")
                last_tactic = current_tactic
            
            # 检查伤亡
            for agent_id in ['A0100', 'A0200']:
                if agent_id in env.agents and not env.agents[agent_id].is_alive:
                    if agent_id not in ['A0100', 'A0200']:
                        continue
                    our_casualties += 1
                    
            for agent_id in ['B0100', 'B0200']:
                if agent_id in env.agents and not env.agents[agent_id].is_alive:
                    enemy_casualties += 1
            
            # 状态输出（每30步）
            if step_count % 300 == 0:
                time_s = step_count * env.time_interval
                distance = tactical_task._calculate_distance(env) / 1000
                print(f"[{time_s:6.1f}s] 距离:{distance:5.1f}km 战术:{current_tactic:20s} 我方:{2-our_casualties}/2 敌方:{2-enemy_casualties}/2")
            
            done = all(dones.values())
            step_count += 1
        
        # 输出结果
        print("\n" + "="*80)
        print("📊 战术测试结果")
        print("="*80)
        print(f"总时长: {step_count * env.time_interval:.1f}秒")
        print(f"我方存活: {2-our_casualties}/2")
        print(f"敌方存活: {2-enemy_casualties}/2")
        print(f"战术切换次数: {len(tactic_changes)}")
        
        if tactic_changes:
            print("\n战术切换记录:")
            for change in tactic_changes:
                print(f"  {change['time']:6.1f}s: {change['from']:20s} → {change['to']}")
        
        # 判定结果
        if our_casualties < enemy_casualties:
            print("\n🎖️ 结果: 我方胜利!")
        elif our_casualties > enemy_casualties:
            print("\n❌ 结果: 敌方胜利")
        else:
            print("\n🤝 结果: 平局")
        
        # 清理
        env.close()
        
        return {
            'success': True,
            'our_casualties': our_casualties,
            'enemy_casualties': enemy_casualties,
            'tactic_changes': len(tactic_changes),
            'final_tactic': current_tactic
        }
        
    except Exception as e:
        logging.error(f"仿真失败: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def run_comprehensive_test():
    """运行综合测试 - 测试所有意图和战术组合"""
    
    print("\n" + "="*80)
    print("🔬 战术系统综合测试")
    print("="*80)
    
    # 测试配置
    intents = ['AGGRESSIVE_CLEAR', 'CONSERVATIVE_CLEAR', 'DEFENSIVE']
    tactics = [None, 'DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SIDE_BY_SIDE']
    enemy_modes = ['AGGRESSIVE', 'DEFENSIVE', 'MIXED']
    
    results = []
    
    # 运行测试组合（简化版，选择代表性组合）
    test_cases = [
        # 意图测试
        ('AGGRESSIVE_CLEAR', None, 'AGGRESSIVE'),
        ('CONSERVATIVE_CLEAR', None, 'MIXED'),
        ('DEFENSIVE', None, 'DEFENSIVE'),
        
        # 战术测试
        ('CONSERVATIVE_CLEAR', 'DRAG_SHOOT', 'MIXED'),
        ('CONSERVATIVE_CLEAR', 'PINCER_ATTACK', 'MIXED'),
        ('CONSERVATIVE_CLEAR', 'HIGH_LOW_ATTACK', 'MIXED'),
        ('CONSERVATIVE_CLEAR', 'SIDE_BY_SIDE', 'MIXED'),
        
        # 对抗测试
        ('AGGRESSIVE_CLEAR', None, 'DEFENSIVE'),
        ('DEFENSIVE', None, 'AGGRESSIVE'),
    ]
    
    for i, (intent, tactic, enemy_mode) in enumerate(test_cases, 1):
        print(f"\n📝 测试 {i}/{len(test_cases)}: 意图={intent}, 战术={tactic or 'AUTO'}, 敌方={enemy_mode}")
        print("-" * 40)
        
        result = run_tactical_test(
            our_intent=intent,
            initial_tactic=tactic,
            enemy_mode=enemy_mode,
            max_steps=800,  # 缩短测试时间
            seed=42 + i  # 每个测试用不同种子
        )
        
        results.append({
            'intent': intent,
            'tactic': tactic or 'AUTO',
            'enemy_mode': enemy_mode,
            **result
        })
    
    # 输出汇总结果
    print("\n" + "="*80)
    print("📈 测试汇总")
    print("="*80)
    
    success_count = sum(1 for r in results if r.get('success', False))
    win_count = sum(1 for r in results if r.get('our_casualties', 2) < r.get('enemy_casualties', 0))
    
    print(f"成功运行: {success_count}/{len(results)}")
    print(f"胜利场次: {win_count}/{success_count}")
    print(f"平均战术切换: {sum(r.get('tactic_changes', 0) for r in results) / len(results):.1f}次")
    
    print("\n详细结果:")
    print(f"{'意图':15s} {'初始战术':15s} {'敌方':10s} {'结果':6s} {'切换':4s}")
    print("-" * 60)
    
    for r in results:
        if r.get('success'):
            our_cas = r.get('our_casualties', 0)
            enemy_cas = r.get('enemy_casualties', 0)
            if our_cas < enemy_cas:
                result = '胜利'
            elif our_cas > enemy_cas:
                result = '失败'
            else:
                result = '平局'
            
            print(f"{r['intent']:15s} {r['tactic']:15s} {r['enemy_mode']:10s} {result:6s} {r.get('tactic_changes', 0):4d}")
        else:
            print(f"{r['intent']:15s} {r['tactic']:15s} {r['enemy_mode']:10s} {'错误':6s}    -")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='战术系统测试')
    parser.add_argument('--mode', choices=['single', 'comprehensive'], default='single',
                        help='测试模式')
    parser.add_argument('--intent', choices=['AGGRESSIVE_CLEAR', 'CONSERVATIVE_CLEAR', 'DEFENSIVE'],
                        default='CONSERVATIVE_CLEAR', help='我方意图')
    parser.add_argument('--tactic', choices=['AUTO', 'DRAG_SHOOT', 'PINCER_ATTACK', 
                                            'HIGH_LOW_ATTACK', 'SIDE_BY_SIDE'],
                        default='AUTO', help='初始战术')
    parser.add_argument('--enemy', choices=['AGGRESSIVE', 'DEFENSIVE', 'MIXED'],
                        default='MIXED', help='敌方模式')
    parser.add_argument('--steps', type=int, default=1650, help='最大步数')
    parser.add_argument('--seed', type=int, help='随机种子')
    
    args = parser.parse_args()
    
    if args.mode == 'comprehensive':
        run_comprehensive_test()
    else:
        # 单次测试
        initial_tactic = None if args.tactic == 'AUTO' else args.tactic
        result = run_tactical_test(
            our_intent=args.intent,
            initial_tactic=initial_tactic,
            enemy_mode=args.enemy,
            max_steps=args.steps,
            seed=args.seed
        )
        
        if result['success']:
            print("\n✅ 测试完成")
        else:
            print(f"\n❌ 测试失败: {result.get('error', 'Unknown error')}")
