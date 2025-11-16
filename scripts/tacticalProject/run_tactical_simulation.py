#!/usr/bin/env python3
"""
战术决策系统完整仿真运行脚本
F-16 (我方) vs SU-27 (敌方) 2v2空战仿真
集成新的战术决策系统与JSBSim环境
"""

import os
import sys
import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

# 导入JSBSim环境
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.tasks.multiplecombat_task import HierarchicalMultipleCombatTask
from envs.JSBSim.utils.utils import get_root_dir

# 导入战术适配器
from simulation.tactical_adapter import TacticalAdapter


def setup_logging(output_dir: str) -> str:
    """设置日志系统"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"tactical_simulation_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("=" * 80)
    logging.info("[LAUNCH] 战术决策系统仿真开始")
    logging.info("=" * 80)
    logging.info(f"📝 日志文件: {log_file}")
    return log_file


def print_simulation_header():
    """打印仿真标题"""
    print("\n" + "=" * 80)
    print("🎯 战术决策系统完整仿真")
    print("=" * 80)
    print("我方: F-16C Block 50/52 x2 (蓝方)")
    print("  - 长机: A0100")
    print("  - 僚机: A0200")
    print("  - 雷达: AN/APG-68(V)9")
    print("  - 导弹: AIM-120C AMRAAM")
    print()
    print("敌方: SU-27 Flanker x2 (红方)")
    print("  - 敌机1: B0100")
    print("  - 敌机2: B0200")
    print("  - 雷达: N001VE")
    print("  - 导弹: R-27ER")
    print()
    print("战术系统:")
    print("  - 威胁值计算")
    print("  - 战术选择 (5种进攻战术)")
    print("  - 控制距离节点 (NLT/MELD/MTR/LR/TR/DOR/DR/MAR)")
    print("  - 协同决策")
    print("=" * 80)


def print_tactical_phases():
    """打印战术阶段说明"""
    print("\n📋 控制距离节点说明:")
    print("  NLT  (120km): 转换为战斗队形")
    print("  MELD (100km): 战术选择 + 目标分配")
    print("  MTR  (80km):  前往攻击占位点")
    print("  LR   (78km):  发射导弹 + 中制导")
    print("  TR   (75km):  中制导结束 + 规避")
    print("  DOR  (70km):  规避 + 预决策下一战术")
    print("  DR   (65km):  Beam转侧对 + 决策")
    print("  MAR  (40km):  紧急脱离")
    print()


def calculate_distance(agent1, agent2):
    """计算两个智能体之间的距离"""
    pos1 = agent1.get_position()
    pos2 = agent2.get_position()
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)


def get_min_distance(env) -> float:
    """获取双方最小距离"""
    min_distance = float('inf')
    
    for friendly_id in ["A0100", "A0200"]:
        if friendly_id not in env.agents or not env.agents[friendly_id].is_alive:
            continue
        for enemy_id in ["B0100", "B0200"]:
            if enemy_id not in env.agents or not env.agents[enemy_id].is_alive:
                continue
            distance = calculate_distance(env.agents[friendly_id], env.agents[enemy_id])
            min_distance = min(min_distance, distance)
    
    return min_distance / 1000.0  # 转换为km


def print_status(env, decisions: Dict, current_time: float, step: int):
    """打印当前状态"""
    min_dist = get_min_distance(env)
    
    # 统计存活飞机
    friendly_alive = sum(1 for aid in ["A0100", "A0200"] 
                        if aid in env.agents and env.agents[aid].is_alive)
    enemy_alive = sum(1 for aid in ["B0100", "B0200"] 
                     if aid in env.agents and env.agents[aid].is_alive)
    
    print(f"\n[步骤 {step:4d}] 时间: {current_time:6.1f}s | 距离: {min_dist:6.1f}km | "
          f"我方: {friendly_alive}/2 | 敌方: {enemy_alive}/2")
    
    # 打印战术信息
    if decisions:
        print(f"  阶段: {decisions.get('phase', 'N/A'):12s} | "
              f"战术: {decisions.get('tactic', 'N/A')}")
        
        if decisions.get('lead'):
            lead = decisions['lead']
            print(f"  长机: 节点={lead.get('node', 'N/A'):8s} | "
                  f"行动={lead.get('action', 'N/A'):10s} | "
                  f"威胁={lead.get('threat', {}).get('total', 0):.3f}")
        
        if decisions.get('wingman'):
            wingman = decisions['wingman']
            print(f"  僚机: 节点={wingman.get('node', 'N/A'):8s} | "
                  f"行动={wingman.get('action', 'N/A'):10s} | "
                  f"威胁={wingman.get('threat', {}).get('total', 0):.3f}")


def check_termination(env) -> tuple:
    """检查仿真是否应该终止"""
    # 检查存活飞机
    friendly_alive = [aid for aid in ["A0100", "A0200"] 
                     if aid in env.agents and env.agents[aid].is_alive]
    enemy_alive = [aid for aid in ["B0100", "B0200"] 
                  if aid in env.agents and env.agents[aid].is_alive]
    
    if len(friendly_alive) == 0:
        return True, "我方全部被击落"
    
    if len(enemy_alive) == 0:
        return True, "敌方全部被击落"
    
    # 检查距离
    min_dist = get_min_distance(env)
    if min_dist < 5.0:  # 小于5km
        return True, "距离过近，交战结束"
    
    return False, ""


def run_simulation(
    our_intent_type: str = 'conservative_clear',
    max_steps: int = 3000,
    output_dir: str = './tactical_simulation_results'
):
    """
    运行完整仿真
    
    Args:
        our_intent_type: 我方意图类型
        max_steps: 最大步数
        output_dir: 输出目录
    """
    # 设置日志
    log_file = setup_logging(output_dir)
    
    # 打印标题
    print_simulation_header()
    print_tactical_phases()
    
    # 创建环境
    print("\n🔧 初始化仿真环境...")
    
    try:
        # 使用正确的配置文件名（能够发射导弹的配置）
        env = MultipleCombatEnv("2v2/ShootMissile/HierarchySelfplay")
        env.max_steps = max_steps
        
        print("✓ JSBSim环境创建成功")
    except Exception as e:
        logging.error(f"环境创建失败: {e}")
        print(f"❌ 环境创建失败: {e}")
        return
    
    # 创建战术任务
    print(f"🔧 初始化战术任务系统 (意图: {our_intent_type})...")
    from tactical_task import TacticalTask
    from core import TacticalDecisionManager
    
    # 创建决策管理器（可以传入自定义的）
    decision_manager = TacticalDecisionManager(our_intent_type=our_intent_type)
    
    # 创建战术任务
    tactical_task = TacticalTask(env.config, decision_manager=decision_manager)
    env.task = tactical_task
    
    # 重置环境
    print("🔧 重置环境...")
    obs = env.reset()
    print("✓ 战术系统初始化完成")
    
    print("\n" + "=" * 80)
    print("🚀 仿真开始")
    print("=" * 80)
    
    # 仿真循环
    current_time = 0.0
    dt = 1/12  # 12Hz
    step = 0
    last_print_time = 0.0
    print_interval = 5.0  # 每5秒打印一次
    
    decisions = None
    
    try:
        while step < max_steps:
            step += 1
            current_time = step * dt
            
            # 环境步进（任务类会处理所有决策和控制）
            num_agents = len(env.agents)
            dummy_actions = np.zeros((1, num_agents, 4))
            obs, share_obs, rewards, dones, info = env.step(dummy_actions)
            
            # 定期打印状态
            if current_time - last_print_time >= print_interval:
                # 从任务类获取决策信息
                decisions = {}
                print_status(env, decisions, current_time, step)
                last_print_time = current_time
            
            # 检查终止条件
            should_terminate, reason = check_termination(env)
            if should_terminate:
                print(f"\n{'=' * 80}")
                print(f"🏁 仿真结束: {reason}")
                print(f"{'=' * 80}")
                break
            
            # 检查环境done
            if isinstance(dones, dict):
                if all(dones.values()):
                    logging.info("所有飞机已终止")
                    break
            else:
                # dones是数组
                if dones.all():
                    logging.info("所有飞机已终止")
                    print(f"\n{'=' * 80}")
                    print("🏁 仿真结束: 环境终止")
                    print(f"{'=' * 80}")
                    break
    
    except KeyboardInterrupt:
        print(f"\n{'=' * 80}")
        print("⚠️  仿真被用户中断")
        print(f"{'=' * 80}")
    
    except Exception as e:
        logging.error(f"仿真错误: {e}", exc_info=True)
        print(f"\n❌ 仿真错误: {e}")
    
    finally:
        # 打印最终状态
        print_status(env, decisions, current_time, step)
        
        # 统计结果
        friendly_alive = sum(1 for aid in ["A0100", "A0200"] 
                            if aid in env.agents and env.agents[aid].is_alive)
        enemy_alive = sum(1 for aid in ["B0100", "B0200"] 
                         if aid in env.agents and env.agents[aid].is_alive)
        
        print(f"\n{'=' * 80}")
        print("📊 仿真统计")
        print(f"{'=' * 80}")
        print(f"总步数: {step}")
        print(f"总时间: {current_time:.1f}秒")
        print(f"我方存活: {friendly_alive}/2")
        print(f"敌方存活: {enemy_alive}/2")
        # 计算已发射导弹数 = 初始导弹数 - 剩余导弹数
        missiles_fired = 0
        for agent_id in ['A0100', 'A0200']:
            initial = 2  # 初始导弹数
            remaining = tactical_task.state_manager.missiles_remaining.get(agent_id, 2)
            missiles_fired += (initial - remaining)
        print(f"发射导弹: {missiles_fired}枚")
        
        if friendly_alive > enemy_alive:
            print("\n🎉 我方获胜！")
        elif enemy_alive > friendly_alive:
            print("\n😞 敌方获胜")
        else:
            print("\n🤝 平局")
        
        print(f"\n📝 详细日志: {log_file}")
        print(f"{'=' * 80}\n")
        
        # 关闭环境
        env.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='战术决策系统仿真')
    parser.add_argument('--intent', type=str, default='conservative_clear',
                       choices=['aggressive_clear', 'conservative_clear', 'defensive'],
                       help='我方意图类型')
    parser.add_argument('--steps', type=int, default=3000,
                       help='最大仿真步数')
    parser.add_argument('--output', type=str, default='./tactical_simulation_results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    run_simulation(
        our_intent_type=args.intent,
        max_steps=args.steps,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
