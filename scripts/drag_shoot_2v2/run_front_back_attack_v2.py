#!/usr/bin/env python3
"""
前后攻击战术仿真V2 - 基于拖曳射击成功架构
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# 导入必要模块
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.core.catalog import Catalog as c
from front_back_attack_tactical_task_v2 import FrontBackAttackTacticalTaskV2

def run_front_back_attack_simulation():
    """运行前后攻击战术仿真V2 - 完全基于拖曳射击架构"""
    print("=" * 60)
    print("前后攻击战术仿真V2 (Front-Back Attack V2)")
    print("=" * 60)
    print("基于拖曳射击的成功架构，确保系统稳定运行")
    print("=" * 60)

    try:
        # 创建环境 - 完全复制拖曳射击的方式
        print("创建仿真环境...")
        env = MultipleCombatEnv("drag_shoot_tactical")

        # 强制设置2500步（500秒）- 适合前后攻击战术
        env.max_steps = 2500

        # 替换任务为前后攻击任务V2 - 完全复制拖曳射击的方式
        env.task = FrontBackAttackTacticalTaskV2(env.config)

        # 重置环境
        obs = env.reset()

        print("仿真环境初始化完成")

        # 仿真数据收集
        trajectory_data = []
        missile_events = []

        # 运行仿真
        step_count = 0
        max_steps = env.max_steps

        print(f"仿真参数: 最大步数={max_steps}, 时间间隔={env.time_interval}s")
        print("前后攻击战术目标:")
        print("   1. 僚机建立长机后方3海里队形")
        print("   2. 长机先行发射导弹")
        print("   3. 僚机延迟8秒后方攻击")
        print("   4. 形成前后夹击态势")
        print("开始仿真...")

        while step_count < max_steps:
            # 执行步骤
            actions = {}
            for agent_id in env.agents.keys():
                if env.agents[agent_id].is_alive:
                    # 使用战术任务的normalize_action
                    action = env.task.normalize_action(env, agent_id, None)
                    actions[agent_id] = action
                else:
                    actions[agent_id] = np.array([0.0, 0.0, 0.0, 0.0])

            # 环境步进
            obs, rewards, dones, infos = env.step(actions)

            # 收集轨迹数据
            current_time = step_count * env.time_interval
            for agent_id in env.agents.keys():
                if env.agents[agent_id].is_alive:
                    agent = env.agents[agent_id]
                    pos = agent.get_position()

                    trajectory_data.append({
                        'Time': current_time,
                        'Agent_ID': agent_id,
                        'X': pos[0],
                        'Y': pos[1],
                        'Z': pos[2],
                        'Heading': np.rad2deg(agent.get_property_value(c.attitude_psi_rad)),
                        'Velocity': agent.get_property_value(c.velocities_vc_mps),
                        'Altitude': agent.get_property_value(c.position_h_sl_m),
                        'Status': 'ALIVE',
                        'Phase': env.task.current_phase.value,
                        'Missiles': agent.num_missiles
                    })

            # 检查终止条件
            if all(dones.values()):
                print("所有智能体终止，仿真结束")
                break

            step_count += 1

            # 每100步打印进度
            if step_count % 100 == 0:
                alive_count = sum(1 for agent in env.agents.values() if agent.is_alive)
                print(f"t={current_time:.1f}s, 步数={step_count}, 存活={alive_count}/4, 阶段={env.task.current_phase.value}")

        # 仿真结束
        final_time = step_count * env.time_interval
        print(f"仿真结束: 步数={step_count}, 时间={final_time:.1f}s")

        # 保存轨迹数据
        if trajectory_data:
            df = pd.DataFrame(trajectory_data)

            # 创建结果目录
            os.makedirs('front_back_attack_v2_results', exist_ok=True)

            # 保存CSV文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f'front_back_attack_v2_results/front_back_attack_v2_trajectory_{timestamp}.csv'
            df.to_csv(csv_filename, index=False)
            print(f"轨迹数据已保存: {csv_filename}")

            # 生成统计报告
            generate_simulation_report(env.task, env, final_time, csv_filename)

        env.close()
        print("前后攻击战术仿真V2完成")

    except Exception as e:
        print(f"仿真过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def generate_simulation_report(task, env, final_time, csv_filename):
    """生成仿真统计报告"""
    print("=" * 60)
    print("前后攻击战术仿真V2结果统计")
    print("=" * 60)

    # 存活统计
    alive_agents = []
    dead_agents = []

    for agent_id, agent in env.agents.items():
        if agent.is_alive:
            alive_agents.append(agent_id)
            print(f"   ✅ {agent_id}: 存活")
        else:
            dead_agents.append(agent_id)
            print(f"   ❌ {agent_id}: 被击落")

    print(f"最终存活: {len(alive_agents)}/4")

    # 导弹发射统计
    print("导弹发射统计:")
    total_launches = 0
    for agent_id, count in task.missile_launch_count.items():
        print(f"   {agent_id}: {count}发")
        total_launches += count

    if total_launches == 0:
        print("   无导弹发射")

    # 仿真时长
    print(f"仿真时长: {final_time:.1f}秒")

    # 最终阶段
    print(f"最终阶段: {task.current_phase.value}")

    # 前后攻击战术评估
    formation_success = task.formation_states["A0200"]["formation_established"]
    leader_attacked = task.attack_timeline["leader_missile_launched"]
    wingman_ready = task.attack_timeline["wingman_attack_ready"]

    if formation_success:
        print("✅ 僚机后方队形建立成功")
    else:
        print("❌ 僚机后方队形建立失败")

    if leader_attacked:
        print("✅ 长机先行攻击成功")
    else:
        print("❌ 长机先行攻击失败")

    if wingman_ready:
        print("✅ 僚机后方攻击就绪")
    else:
        print("❌ 僚机后方攻击未就绪")

    print("=" * 60)
    print("前后攻击战术仿真V2完成")

if __name__ == "__main__":
    run_front_back_attack_simulation()
